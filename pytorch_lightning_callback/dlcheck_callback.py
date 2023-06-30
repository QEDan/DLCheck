"""DLCheck. A PyTorch Lightning callback for discovering and debugging training issues.

# References

* Houssem Ben Braiek, Foutse Khomh, [TFCheck : A TensorFlow Library for Detecting
    Training Issues in Neural Network Programs](https://arxiv.org/abs/1909.02562)
* Mohammad Wardat, Breno Dantas Cruz, Wei Le, Hridesh Rajan [DeepDiagnosis:
    Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep
    Learning Programs](https://arxiv.org/abs/2112.04036).
    [DeepDiagnosis is on Github](https://github.com/DeepDiagnosis/ICSE2022).
"""
import copy
import logging

import torch

import lightning.pytorch as L

logger = logging.getLogger("lightning.pytorch.core")


class DLCheckCallback(L.Callback):
    """A PyTorch Lightning callback for discovering and debugging training issues"""
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.weight_init_eps = config.get('weight_init_eps', 1.0e-6)
        self.divergence_threshold_min = config.get('divergence_threshold_min', 1.0e3)
        self.divergence_threshold_max = config.get('divergence_threshold_max', -1.0e3)
        self.prev_weights = None
        self.prev_params = None
        self.neuron_output_history = None
        self.lowest_loss_value = None
        self.model_params = None
        self.model_modules = None
        self.initial_model = None
        self.loss_history = None

    def check_weight_initialization(self):
        """Check the weight initialization."""
        passed = True
        for m_name, m_params in self.model_params.items():
            for i, param in enumerate(m_params):
                std_p = torch.std(param)
                if std_p < self.weight_init_eps:
                    passed = False
                    logger.warning(f"\nWarning: Module {m_name} has very small "
                                   f"weight variance in parameter {i} at initialization. "
                                   f"std_p={std_p} < {self.weight_init_eps}. ")
        return passed

    def check_untrained_params(self, model: L.LightningModule) -> bool:
        """Compare model weights with previous epoch."""
        passed = True
        curr_params = [var for var in model.parameters() if var.requires_grad]
        for i, curr_p, prev_p in enumerate(zip(curr_params, self.prev_params)):
            if torch.equal(curr_p, prev_p):
                passed = False
                logging.warning(f"Parameter {i} {curr_p.name} was not updated with training. "
                                f"Confirm that this layer is properly connected.")
        self.prev_weights = [copy.deepcopy(p) for p in curr_params]
        return passed

    def check_diverging_params(self, model: L.LightningModule) -> bool:
        """Check the upper and lower quartiles of the weights"""
        passed = True
        curr_params = [var for var in model.parameters() if var.requires_grad]
        for i, curr_p in enumerate(curr_params):
            percentile_25, percentile_75 = torch.quantile(curr_p, torch.tensor([0.25, 0.75]))
            if percentile_25 < self.divergence_threshold_min:
                passed = False
                logger.warning(f"Parameter {i} {curr_p.name} has 25% of weights "
                               f"less than {self.divergence_threshold_min}. Confirm "
                               f"that this layer is learning properly.")
            if percentile_75 > self.divergence_threshold_max:
                passed = False
                logger.warning(f"Parameter {i} {curr_p.name} has 25% of weights "
                               f"greater than {self.divergence_threshold_max}. Confirm "
                               f"that this layer is learning properly.")
        return passed

    def init_callback(self, model: L.LightningModule) -> None:
        """Initialize the callback using the initial state of the model"""
        self.initial_model = copy.deepcopy(model)
        self.model_modules = [m for m in self.initial_model.modules()
                              if any(p.requires_grad for p in m.parameters())]
        # pylint: disable=protected-access
        self.model_params = {m._get_name(): list(m.parameters())
                             for m in self.initial_model.modules()
                             if any(p.requires_grad for p in m.parameters())}
        self.loss_history = []
        self.lowest_loss_value = None
        self.neuron_output_history = None
        self.prev_params = [var for var in self.initial_model.parameters() if var.requires_grad]

    def on_train_start(self, trainer, pl_module):
        """Callback at start of training"""
        self.init_callback(pl_module)
        logging.info("Training with DLCheck Callback.")
        self.check_weight_initialization()

    def on_train_epoch_end(self, trainer, pl_module):
        """Callback at end of training epoch"""
        self.check_untrained_params(pl_module)
        self.check_diverging_params(pl_module)
