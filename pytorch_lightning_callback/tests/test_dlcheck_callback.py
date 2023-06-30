"""Test the DLCheckCallback class"""

import lightning.pytorch as L
import pytest
import torch
from torch.nn import functional as F

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class ModelForTests(L.LightningModule):
    """Lightning Module for Testing"""
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class TestDLCheckCallback:
    """Test the DLCheckCallback class"""
    def setup_method(self):
        """Setup tests"""
        self.model = ModelForTests()
        self.callback = DLCheckCallback()
        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3,
            callbacks=[self.callback]
)

    @pytest.fixture
    def init(self):
        """Initialize the callback"""
        self.callback.init_callback(self.model)

    def test_check_weight_initialization(self, init):
        """Check normal weight initialization"""
        passed = self.callback.check_weight_initialization()
        assert passed

    def test_bad_weight_initialization(self):
        """Check weight initilization with zeroed weights"""
        for param in self.model.parameters():
            param.data = torch.zeros(param.shape)
        self.callback.init_callback(self.model)
        passed = self.callback.check_weight_initialization()
        assert not passed
