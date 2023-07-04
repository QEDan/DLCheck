"""Test the DLCheckCallback class"""

import lightning.pytorch as L
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import get_graph_node_names

from pytorch_lightning_callback.dlcheck_callback import DLCheckCallback


class DatasetForTests(Dataset):
    def __init__(self, shape=(8, 28, 28)):
        self.shape = shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return torch.rand((1,) + self.shape[1:]), 0


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
        self.data_loader = DataLoader(DatasetForTests(), batch_size=4)
        self.trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=3
        )

    @pytest.fixture
    def init(self):
        """Initialize the callback"""
        self.model = ModelForTests()
        self.callback = DLCheckCallback()
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

    def test_check_untrained_params_fail(self, init):
        """Test that params are not updated with no training"""
        passed = self.callback.check_untrained_params(self.model)
        assert not passed

    def test_check_untrained_params(self, init):
        """Test that params are updated with training"""
        self.trainer.fit(self.model, self.data_loader)
        passed = self.callback.check_untrained_params(self.model)
        assert passed

    def test_check_diverging_params(self):
        """Test that freshly initialized model passes diverging parameters check"""
        passed = self.callback.check_diverging_params(self.model)
        assert passed

    def test_check_diverging_params_small(self):
        """Test that model with small parameters fails the diverging parameters check"""
        for param in self.model.parameters():
            param.data = 1.0e2 * self.callback.divergence_threshold_min * param.data
        passed = self.callback.check_diverging_params(self.model)
        assert not passed

    def test_check_diverging_params_large(self):
        """Test that model with large parameters fails the diverging parameters check"""
        for param in self.model.parameters():
            param.data = 1.0e2 * self.callback.divergence_threshold_max * param.data
        passed = self.callback.check_diverging_params(self.model)
        assert not passed

    def test_check_unstable_learning(self, init):
        """Test that learning is not unstable."""
        self.trainer.fit(self.model, self.data_loader)
        passed = self.callback.check_unstable_learning(self.trainer, self.model)
        assert passed

    def test_get_parameter_outputs(self, init):
        outs = self.callback.get_parameter_outputs(self.model, self.data_loader.__iter__().__next__()[0])
        node_names = get_graph_node_names(self.model)[0]
        assert any(isinstance(outs[nn], torch.Tensor) for nn in node_names)
