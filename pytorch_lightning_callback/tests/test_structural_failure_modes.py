"""Tests for Structural & Architectural Failure Modes (Section 2 of failure_modes_todos.md)"""

import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class SimpleDataset(Dataset):
    def __init__(self, size=16):
        self.data = torch.rand(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class StructuralModel(L.LightningModule):
    def __init__(self, behavior="normal"):
        super().__init__()
        self.behavior = behavior
        self.l1 = nn.Linear(10, 20)
        
        if behavior == "dying_relu":
            # Force negative bias to kill ReLUs
            self.l1.bias.data.fill_(-10.0)
            self.act = nn.ReLU()
        elif behavior == "saturated":
            # Force large weights to saturate Sigmoid
            self.l1.weight.data.fill_(100.0)
            self.act = nn.Sigmoid()
        elif behavior == "untrained":
            self.l2 = nn.Linear(20, 2)
            # We will manually prevent l2 from being trained in the optimizer config
            self.act = nn.ReLU()
        else:
            self.act = nn.ReLU()
            self.l2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.act(self.l1(x))
        if hasattr(self, 'l2'):
            x = self.l2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = nn.functional.cross_entropy(out, y)
        return loss

    def configure_optimizers(self):
        if self.behavior == "untrained":
            # Only optimize l1
            return torch.optim.Adam(self.l1.parameters(), lr=0.01)
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TestStructuralFailureModes:
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
        self.loader = DataLoader(SimpleDataset(), batch_size=4)

    def test_dying_relu_detection(self, setup):
        """Test that dying ReLUs are detected."""
        model = StructuralModel(behavior="dying_relu")
        self.trainer.fit(model, self.loader)
        
        passed = self.callback.check_dying_relu(model)
        assert not passed

    def test_activation_saturation_detection(self, setup):
        """Test that saturated activations are detected."""
        model = StructuralModel(behavior="saturated")
        self.trainer.fit(model, self.loader)
        
        passed = self.callback.check_activation_saturation(model)
        assert not passed

    def test_untrained_layer_detection_extended(self, setup):
        """Test that untrained layers are detected via checksums."""
        model = StructuralModel(behavior="untrained")
        self.trainer.fit(model, self.loader)
        
        passed = self.callback.check_untrained_layers_extended(model)
        assert not passed
