"""Tests for Augmentation Sanity failure modes (Section 5 of failure_modes_todos.md)"""

import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class AugmentationDataset(Dataset):
    def __init__(self, behavior="normal", size=100):
        self.behavior = behavior
        self.size = size
        # Normal data: centered around 0, std 1
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

        if behavior == "out_of_range":
            # Simulation of augmentation that produces massive values
            self.data = self.data * 1000 + 5000
        elif behavior == "constant":
            # Simulation of augmentation that zero-out everything (broken cutout)
            self.data = torch.zeros_like(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 2)

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        return nn.functional.cross_entropy(out, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TestAugmentationFailureModes:
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

    def test_augmentation_out_of_range_detection(self, setup):
        """Test that augmentations producing extreme values are detected."""
        dataset = AugmentationDataset(behavior="out_of_range")
        loader = DataLoader(dataset, batch_size=10)
        model = SimpleModel()
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_augmentation_sanity(self.trainer)
        assert not passed

    def test_augmentation_constant_output_detection(self, setup):
        """Test that augmentations producing constant (e.g. all zero) outputs are detected."""
        dataset = AugmentationDataset(behavior="constant")
        loader = DataLoader(dataset, batch_size=10)
        model = SimpleModel()
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_augmentation_sanity(self.trainer)
        assert not passed
