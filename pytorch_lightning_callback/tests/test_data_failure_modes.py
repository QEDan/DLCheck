"""Tests for Data & Pipeline Integrity Failure Modes (Section 3 of failure_modes_todos.md)"""

import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class DataIntegrityDataset(Dataset):
    def __init__(self, behavior="normal", size=100):
        self.behavior = behavior
        self.data = torch.rand(size, 10)
        self.labels = torch.randint(0, 2, (size,))
        
        if behavior == "bad_scaling":
            # Very large values
            self.data = torch.randn(size, 10) * 500 + 1000
        elif behavior == "leakage":
            # Feature is just the label
            self.data = self.labels.float().unsqueeze(1).repeat(1, 10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class DataIntegrityModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 2)

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TestDataFailureModes:
    @pytest.fixture
    def setup(self):
        self.callback = DLCheckCallback()
        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=1,
            callbacks=[self.callback],
            logger=False,
            enable_checkpointing=False
        )

    def test_input_scaling_check(self, setup):
        """Test that poorly scaled inputs are detected."""
        dataset = DataIntegrityDataset(behavior="bad_scaling")
        loader = DataLoader(dataset, batch_size=10)
        model = DataIntegrityModel()
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_input_scaling(model)
        assert not passed

    def test_label_leakage_detection(self, setup):
        """Test that potential label leakage is detected via suspicious convergence."""
        dataset = DataIntegrityDataset(behavior="leakage")
        loader = DataLoader(dataset, batch_size=10)
        model = DataIntegrityModel()
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_label_leakage(self.trainer)
        assert not passed

    def test_label_noise_detection(self, setup):
        """Test that noisy/mislabeled samples are detected."""
        dataset = DataIntegrityDataset(size=10)
        # Manually corrupt one sample to have high loss
        dataset.data[0].fill_(0.5)
        dataset.labels[0] = 0 # Assume this conflicts with model expectation
        
        loader = DataLoader(dataset, batch_size=5)
        model = DataIntegrityModel()
        
        # Train for a few epochs to see sample loss consistency
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_label_noise(self.trainer)
        assert not passed
