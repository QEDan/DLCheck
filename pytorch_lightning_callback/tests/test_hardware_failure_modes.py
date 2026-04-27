"""Tests for Hardware & Resource Usage Failure Modes (Section 4 of failure_modes_todos.md)"""

import time
import lightning.pytorch as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from unittest.mock import patch

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class SlowDataset(Dataset):
    def __init__(self, delay=0.1, size=10):
        self.delay = delay
        self.size = size
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate slow data loading
        time.sleep(self.delay)
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


class TestHardwareFailureModes:
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

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_stats")
    def test_gpu_memory_fragmentation_detection(self, mock_stats, mock_cuda, setup):
        """Test that high GPU memory fragmentation is detected."""
        # Simulate high fragmentation: high reserved vs allocated
        # fragmentation = (reserved - allocated) / reserved
        # e.g. reserved=10GB, allocated=1GB -> 90% fragmentation
        mock_stats.return_value = {
            "reserved_bytes.all.current": 10 * 1024**3,
            "allocated_bytes.all.current": 1 * 1024**3
        }
        
        model = SimpleModel()
        # We don't necessarily need to fit for this if we can call the check directly,
        # but fit() initializes the callback.
        dataset = SlowDataset(delay=0, size=5)
        loader = DataLoader(dataset, batch_size=5)
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_gpu_memory_fragmentation(self.trainer)
        assert not passed

    def test_cpu_gpu_bottleneck_detection(self, setup):
        """Test that data loading bottlenecks are detected."""
        # Use a slow dataset and single worker to ensure bottleneck
        dataset = SlowDataset(delay=0.2, size=10)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        model = SimpleModel()
        
        self.trainer.fit(model, loader)
        
        passed = self.callback.check_cpu_gpu_bottleneck(self.trainer)
        assert not passed
