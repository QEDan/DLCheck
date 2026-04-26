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
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

logger = logging.getLogger("lightning.pytorch.core")
EPS = 1.0e-32


class DLCheckCallback(L.Callback):
    """A PyTorch Lightning callback for discovering and debugging training issues"""
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.weight_init_eps = config.get('weight_init_eps', 1.0e-6)
        self.divergence_threshold_min = config.get('divergence_threshold_min', -1.0e4)
        self.divergence_threshold_max = config.get('divergence_threshold_max', 1.0e4)
        self.unstable_learning_threshold_min = config.get('unstable_learning_threshold_min', -8.0)
        self.unstable_learning_threshold_max = config.get('unstable_learning_threshold_max', 1.0)
        self.prev_param_stats = None
        self.neuron_output_history = None
        self.lowest_loss_value = None
        self.loss_history = None

    @staticmethod
    def get_parameter_outputs(model, input_tensor):
        """Gets output tensors for each parameter."""
        node_names = get_graph_node_names(model)[0]  # First element is the training node names list
        model_with_outs = create_feature_extractor(model, node_names)
        outs = model_with_outs(input_tensor)
        return outs

    @staticmethod
    def _get_param_stats(model):
        """Returns a list of statistics for each trainable parameter."""
        stats = []
        for p in model.parameters():
            if p.requires_grad:
                # Store sum and L2 norm as a basic "fingerprint" of the weights
                stats.append((torch.sum(p).item(), torch.norm(p).item()))
        return stats

    def check_weight_initialization(self, model: L.LightningModule):
        """Check the weight initialization."""
        passed = True
        for m_name, m in model.named_modules():
            for i, param in enumerate(m.parameters(recurse=False)):
                if not param.requires_grad:
                    continue
                std_p = torch.std(param)
                if std_p < self.weight_init_eps:
                    passed = False
                    logger.warning(f"\nModule {m_name} has very small "
                                   f"weight variance in parameter {i} at initialization. "
                                   f"std_p={std_p} < {self.weight_init_eps}. "
                                   f"Check model initialization.")
        return passed

    def check_untrained_params(self, model: L.LightningModule) -> bool:
        """Compare model weights with previous epoch using statistics."""
        passed = True
        curr_stats = self._get_param_stats(model)
        if self.prev_param_stats is None:
            self.prev_param_stats = curr_stats
            return True

        for i, (curr, prev) in enumerate(zip(curr_stats, self.prev_param_stats)):
            if curr == prev:
                passed = False
                logging.warning(f"Parameter {i} was not updated with training. "
                                f"Confirm that this layer is properly connected.")
        self.prev_param_stats = curr_stats
        return passed

    def check_diverging_params(self, model: L.LightningModule) -> bool:
        """Check the upper and lower quartiles of the weights"""
        passed = True
        curr_params = [var for var in model.parameters() if var.requires_grad]
        for i, curr_p in enumerate(curr_params):
            percentile_25, percentile_75 = torch.quantile(
                curr_p,
                torch.tensor([0.25, 0.75], device=curr_p.device))
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

    def check_unstable_learning(self, trainer, model):
        """Report issues found during the epoch and reset reports."""
        passed = True
        batch_count = self.epoch_reports['batch_count']
        if batch_count == 0:
            return True

        # Check zero gradients
        for name in self.epoch_reports['zero_grads']:
            passed = False
            logging.warning(f"Parameter {name} had zero gradients during one or more batches. "
                            "Check that layer is connected and learning.")
        
        # Check update ratios
        for name, running_ratio in self.epoch_reports['update_ratios'].items():
            mean_ratio = running_ratio / batch_count
            log_mean_ratio = torch.log10(torch.tensor(mean_ratio + EPS)).item()

            if log_mean_ratio < self.unstable_learning_threshold_min:
                passed = False
                logging.warning(f"Parameter {name} has a mean update ratio of 10^{log_mean_ratio:.2f}, "
                                f"which is less than 10^{self.unstable_learning_threshold_min}. "
                                f"Learning may be slow.")
            
            if log_mean_ratio > self.unstable_learning_threshold_max:
                passed = False
                logging.warning(f"Parameter {name} has a mean update ratio of 10^{log_mean_ratio:.2f}, "
                                f"which is greater than 10^{self.unstable_learning_threshold_max}. "
                                f"Learning may be unstable.")
        
        # Reset for next epoch
        self.epoch_reports = {
            'zero_grads': set(),
            'update_ratios': {},
            'batch_count': 0
        }
        return passed

    def check_dead_activations(self):
        passed = True
        param_outputs = self.get_parameter_outputs()

    def init_callback(self, model: L.LightningModule) -> None:
        """Initialize the callback using the initial state of the model"""
        self.loss_history = []
        self.lowest_loss_value = None
        self.neuron_output_history = None
        self.prev_param_stats = self._get_param_stats(model)
        # Registry to track health statistics during the epoch
        self.epoch_reports = {
            'zero_grads': set(),
            'update_ratios': {},  # {param_name: RunningSum}
            'batch_count': 0
        }

    def on_after_backward(self, trainer, pl_module):
        """Check gradients immediately after backward pass and update running stats."""
        self.epoch_reports['batch_count'] += 1
        
        # Support for multiple optimizers: use the max LR as a heuristic or check all
        # For now, we'll associate the first optimizer's LR with all params
        # TODO: Map parameters to specific optimizers for perfect accuracy
        lrs = [pg['lr'] for opt in trainer.optimizers for pg in opt.param_groups]
        max_lr = max(lrs) if lrs else 0.0

        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.grad is None or (param.grad == 0).all():
                self.epoch_reports['zero_grads'].add(name)
                continue

            mean_abs_grad = torch.mean(torch.abs(param.grad)).item()
            mean_abs_val = torch.mean(torch.abs(param.data)).item()
            
            if mean_abs_val > 0:
                ratio = (max_lr * mean_abs_grad) / (mean_abs_val + EPS)
                if name not in self.epoch_reports['update_ratios']:
                    self.epoch_reports['update_ratios'][name] = 0.0
                self.epoch_reports['update_ratios'][name] += ratio

    def on_train_start(self, trainer, pl_module):
        """Callback at start of training"""
        self.init_callback(pl_module)
        logging.info("Training with DLCheck Callback.")
        self.check_weight_initialization(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        """Callback at end of training epoch"""
        self.check_untrained_params(pl_module)
        self.check_diverging_params(pl_module)
        self.check_unstable_learning(trainer, pl_module)
        # self.check_dead_activations()  # TODO: implement with hooks


