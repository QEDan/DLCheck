"""Tests for Learning Rate failure modes (Phase 5 of refactoring_todos.md)"""

import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class SimpleDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LRVanishingModel(L.LightningModule):
    def __init__(self, target_lr=1e-9):
        super().__init__()
        self.l1 = nn.Linear(10, 2)
        self.target_lr = target_lr

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        # Constant high loss
        return torch.tensor(1.0, requires_grad=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.target_lr)
        return optimizer


class TestLRFailureModes:
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

    def test_lr_vanishing_detection(self, setup):
        """Test that low LR with high loss is detected."""
        dataset = SimpleDataset()
        loader = DataLoader(dataset, batch_size=5)
        model = LRVanishingModel(target_lr=1e-10)
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_lr_health(self.trainer)
        assert not passed
