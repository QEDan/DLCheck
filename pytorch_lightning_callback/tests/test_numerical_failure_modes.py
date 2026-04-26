"""Tests for Numerical & Optimization Failure Modes (Section 1 of failure_modes_todos.md)"""

import lightning.pytorch as L
import pytest
import torch
from torch.nn import functional as F
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


class SimpleModel(L.LightningModule):
    def __init__(self, behavior="normal"):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 2)
        self.behavior = behavior

    def forward(self, x):
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        
        if self.behavior == "nan":
            # Induce NaN in loss
            loss = F.cross_entropy(out, y) * torch.tensor(float('nan'))
        elif self.behavior == "exploding":
            # Induce exploding gradients
            loss = F.cross_entropy(out, y) * 1e20
        elif self.behavior == "constant":
            # Loss doesn't depend on weights
            loss = torch.tensor(1.0, requires_grad=True)
        else:
            loss = F.cross_entropy(out, y)
            
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class TestNumericalFailureModes:
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

    def test_gradient_exploding_detection(self, setup):
        """Test that exploding gradients are detected."""
        model = SimpleModel(behavior="exploding")
        self.trainer.fit(model, self.loader)
        
        passed = self.callback.check_gradient_health(model)
        assert not passed

    def test_gradient_vanishing_detection(self, setup):
        """Test that vanishing gradients are detected."""
        model = SimpleModel()
        # Mocking zero gradients to simulate extreme vanishing
        self.trainer.fit(model, self.loader)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        passed = self.callback.check_gradient_health(model)
        assert not passed

    def test_nan_propagation_tracking(self, setup):
        """Test that NaN propagation source is tracked."""
        model = SimpleModel(behavior="nan")
        # We expect this to either catch during backward or after
        try:
            self.trainer.fit(model, self.loader)
        except Exception:
            pass # Trainer might crash on NaN depending on config
            
        source_layer = self.callback.track_nan_propagation(model)
        assert source_layer is not None

    def test_loss_inconsistency_check(self, setup):
        """Test that constant loss with non-zero gradients is flagged."""
        model = SimpleModel(behavior="constant")
        self.trainer.fit(model, self.loader)
        
        passed = self.callback.check_loss_consistency(self.trainer)
        assert not passed
