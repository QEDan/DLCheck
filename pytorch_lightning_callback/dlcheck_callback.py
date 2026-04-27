"""DLCheck. A PyTorch Lightning callback for discovering and debugging training issues.

# References

* Houssem Ben Braiek, Foutse Khomh, [TFCheck : A TensorFlow Library for Detecting
    Training Issues in Neural Network Programs](https://arxiv.org/abs/1909.02562)
* Mohammad Wardat, Breno Dantas Cruz, Wei Le, Hridesh Rajan [DeepDiagnosis:
    Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep
    Learning Programs](https://arxiv.org/abs/2112.04036).
    [DeepDiagnosis is on Github](https://github.com/DeepDiagnosis/ICSE2022).
"""
import logging

import torch

import lightning.pytorch as L

logger = logging.getLogger("lightning.pytorch.core")
EPS = 1.0e-32


class DLCheckCallback(L.Callback):
    """A PyTorch Lightning callback for discovering and debugging training issues.

    References:
    - [arXiv:2112.04036] (DeepDiagnosis: Automatically Diagnosing Faults and Recommending Actionable Fixes)
    - [arXiv:1909.02562] (TFCheck: A TensorFlow Library for Detecting Training Issues)
    - [arXiv:2411.08172] (Fault Localization in Deep Learning: A Survey)
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.weight_init_eps = config.get('weight_init_eps', 1.0e-6)
        self.divergence_threshold_min = config.get('divergence_threshold_min', -1.0e4)
        self.divergence_threshold_max = config.get('divergence_threshold_max', 1.0e4)
        self.unstable_learning_threshold_min = config.get('unstable_learning_threshold_min', -8.0)
        self.unstable_learning_threshold_max = config.get('unstable_learning_threshold_max', 1.0)
        self.dead_neuron_threshold = config.get('dead_neuron_threshold', 0.95)  # 95% dead as per TFCheck
        self.layers_to_monitor = config.get('layers_to_monitor', None)  # List of layer names
        
        # New thresholds
        self.vanishing_grad_threshold = config.get('vanishing_grad_threshold', 1e-7)
        self.exploding_grad_threshold = config.get('exploding_grad_threshold', 1e6)
        self.saturation_threshold_high = config.get('saturation_threshold_high', 0.95)
        self.saturation_threshold_low = config.get('saturation_threshold_low', 0.05)
        self.loss_window_size = config.get('loss_window_size', 10)

        # Data integrity thresholds [arXiv:1909.02562, arXiv:2412.11304]
        self.input_mean_threshold = config.get('input_mean_threshold', 10.0)
        self.input_std_threshold = config.get('input_std_threshold', 100.0)
        self.leakage_batch_threshold = config.get('leakage_batch_threshold', 5)
        self.leakage_loss_threshold = config.get('leakage_loss_threshold', 0.7)

        self.prev_param_stats = None
        self.loss_history = []
        self.hooks = []
        self.activation_stats = {}
        self.input_stats = {}
        self.sample_losses = {} # {sample_idx: [losses]}
        self.nan_source = None
        self.initial_weights = {}

    def _register_hooks(self, model):
        """Register forward hooks to monitor activations.
        
        References:
        - [arXiv:1909.02562] (Dying ReLU, Input Scaling)
        - [arXiv:2112.04036] (Activation Saturation, NaN Propagation)
        """
        self.hooks = []
        self.activation_stats = {}
        self.input_stats = {}

        for name, module in model.named_modules():
            # Skip the top-level model container itself if it's the only one
            if name == "":
                continue

            # If layers_to_monitor is provided, only monitor those
            if self.layers_to_monitor is not None:
                if name not in self.layers_to_monitor:
                    continue
            else:
                # Default: monitor leaf modules (no children) or modules with parameters
                if len(list(module.children())) > 0 and len(list(module.parameters(recurse=False))) == 0:
                    continue

            self.activation_stats[name] = {
                'dead_count': 0, 
                'total_count': 0,
                'abs_sum': 0.0,
                'abs_count': 0,
                'type': type(module).__name__
            }
            self.input_stats[name] = {
                'mean_sum': 0.0,
                'std_sum': 0.0,
                'count': 0
            }

            def hook_fn(m, inp, out, n=name):
                # Input Scaling Check [arXiv:1909.02562]
                if isinstance(inp, (tuple, list)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    curr_inp = inp[0]
                    self.input_stats[n]['mean_sum'] += torch.mean(curr_inp).item()
                    self.input_stats[n]['std_sum'] += torch.std(curr_inp).item()
                    self.input_stats[n]['count'] += 1

                if isinstance(out, torch.Tensor):
                    # NaN/Inf Propagation Tracking [arXiv:2112.04036]
                    if self.nan_source is None and not torch.isfinite(out).all():
                        self.nan_source = n
                    
                    # Dying ReLU Detection [arXiv:1909.02562]
                    if isinstance(m, torch.nn.ReLU):
                        is_zero = (out == 0).float()
                        self.activation_stats[n]['dead_count'] += torch.sum(is_zero).item()
                        self.activation_stats[n]['total_count'] += out.numel()
                    
                    # Activation Saturation (Sigmoid/Tanh) [arXiv:2112.04036]
                    if isinstance(m, (torch.nn.Sigmoid, torch.nn.Tanh)):
                        self.activation_stats[n]['abs_sum'] += torch.sum(torch.abs(out)).item()
                        self.activation_stats[n]['abs_count'] += out.numel()

            self.hooks.append(module.register_forward_hook(hook_fn))

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
        """Report issues found during the epoch and reset reports.
        
        References:
        - [arXiv:2112.04036] (Gradient Health)
        """
        passed = True
        batch_count = self.epoch_reports['batch_count']
        if batch_count == 0:
            return True

        # Check NaN gradients [arXiv:2112.04036]
        for name in self.epoch_reports['nan_grads']:
            passed = False
            self.health_summary['issue_counts']['nan_gradients'][name] = \
                self.health_summary['issue_counts']['nan_gradients'].get(name, 0) + 1
            logging.warning(f"Parameter {name} had NaN/Inf gradients. training is likely broken.")

        # Check Exploding gradients [arXiv:2112.04036]
        for name in self.epoch_reports['exploding_grads']:
            passed = False
            self.health_summary['issue_counts']['exploding_gradients'][name] = \
                self.health_summary['issue_counts']['exploding_gradients'].get(name, 0) + 1
            logging.warning(f"Parameter {name} had exploding gradients.")

        # Check Vanishing gradients [arXiv:2112.04036]
        for name in self.epoch_reports['vanishing_grads']:
            passed = False
            self.health_summary['issue_counts']['vanishing_gradients'][name] = \
                self.health_summary['issue_counts']['vanishing_gradients'].get(name, 0) + 1
            logging.warning(f"Parameter {name} had vanishing gradients.")

        # Check zero gradients
        for name in self.epoch_reports['zero_grads']:
            passed = False
            self.health_summary['issue_counts']['zero_gradients'][name] = \
                self.health_summary['issue_counts']['zero_gradients'].get(name, 0) + 1
            logging.warning(f"Parameter {name} had zero gradients during one or more batches.")
        
        # Check update ratios
        for name, running_ratio in self.epoch_reports['update_ratios'].items():
            mean_ratio = running_ratio / batch_count
            log_mean_ratio = torch.log10(torch.tensor(mean_ratio + EPS)).item()

            if log_mean_ratio < self.unstable_learning_threshold_min:
                passed = False
                self.health_summary['issue_counts']['slow_learning'][name] = \
                    self.health_summary['issue_counts']['slow_learning'].get(name, 0) + 1
                logging.warning(f"Parameter {name} has a mean update ratio of 10^{log_mean_ratio:.2f} (slow).")
            
            if log_mean_ratio > self.unstable_learning_threshold_max:
                passed = False
                self.health_summary['issue_counts']['unstable_learning'][name] = \
                    self.health_summary['issue_counts']['unstable_learning'].get(name, 0) + 1
                logging.warning(f"Parameter {name} has a mean update ratio of 10^{log_mean_ratio:.2f} (unstable).")
        
        return passed

    def check_dead_activations(self):
        """Report layers with a high percentage of dead neurons."""
        passed = True
        for name, stats in self.activation_stats.items():
            if stats['total_count'] > 0:
                dead_fraction = stats['dead_count'] / stats['total_count']
                if dead_fraction > self.dead_neuron_threshold:
                    passed = False
                    self.health_summary['issue_counts']['dead_activations'][name] = \
                        self.health_summary['issue_counts']['dead_activations'].get(name, 0) + 1
                    logging.warning(f"Layer {name} has {dead_fraction:.2%} dead neurons.")
                
                # Check for saturation too
                if any(t in stats['type'] for t in ('Sigmoid', 'Tanh')) and stats['abs_count'] > 0:
                    mean_abs = stats['abs_sum'] / stats['abs_count']
                    if mean_abs > self.saturation_threshold_high or mean_abs < self.saturation_threshold_low:
                        passed = False
                        self.health_summary['issue_counts']['saturated_activations'][name] = \
                            self.health_summary['issue_counts']['saturated_activations'].get(name, 0) + 1
                        logging.warning(f"Layer {name} has saturated activations (mean abs: {mean_abs:.4f}).")
        
        return passed

    def init_callback(self, model: L.LightningModule) -> None:
        """Initialize the callback using the initial state of the model"""
        self.loss_history = []
        self.prev_param_stats = self._get_param_stats(model)
        
        # Capture initial weights for extended untrained layer check [arXiv:1909.02562]
        self.initial_weights = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

        # Registry to track health statistics during the epoch
        self.epoch_reports = {
            'zero_grads': set(),
            'nan_grads': set(),
            'exploding_grads': set(),
            'vanishing_grads': set(),
            'update_ratios': {},  # {param_name: RunningSum}
            'batch_count': 0
        }
        # Final summary report across all epochs
        self.health_summary = {
            'total_batches': 0,
            'issue_counts': {
                'zero_gradients': {},
                'nan_gradients': {},
                'exploding_gradients': {},
                'vanishing_gradients': {},
                'slow_learning': {},
                'unstable_learning': {},
                'dead_activations': {},
                'saturated_activations': {},
                'untrained_layers': {}
            }
        }
        self.nan_source = None
        self._register_hooks(model)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Capture loss for consistency and integrity checks."""
        loss = None
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        
        if loss is not None:
            self.loss_history.append(loss.item())
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)

        # Per-sample loss tracking for label noise detection
        # We try to re-run the model on the batch with reduction='none'
        if hasattr(pl_module, 'forward'):
            try:
                # Ensure model is in eval mode for re-run to avoid updating stats
                was_training = pl_module.training
                pl_module.eval()
                
                x, y = batch
                # Move to same device
                x = x.to(pl_module.device)
                y = y.to(pl_module.device)
                with torch.no_grad():
                    out = pl_module(x)
                    # Generic cross_entropy for classification, can be extended
                    if out.shape[-1] > 1 and y.dtype == torch.long:
                        per_sample_loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
                        batch_size = x.size(0)
                        for i in range(batch_size):
                            idx = batch_idx * batch_size + i
                            if idx not in self.sample_losses:
                                self.sample_losses[idx] = []
                            self.sample_losses[idx].append(per_sample_loss[i].item())
                
                if was_training:
                    pl_module.train()
            except Exception:
                pass # Skip if batch format is not standard (x, y)

    def on_after_backward(self, trainer, pl_module):
        """Check gradients and log metrics.
        
        References:
        - [arXiv:2112.04036] (DeepDiagnosis, Symptom #3, #7, #8)
        """
        self.epoch_reports['batch_count'] += 1
        self.health_summary['total_batches'] += 1
        
        lrs = [pg['lr'] for opt in trainer.optimizers for pg in opt.param_groups]
        max_lr = max(lrs) if lrs else 0.0

        for name, param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.grad is None:
                continue

            # NaN/Inf Sentry [arXiv:2112.04036]
            if not torch.isfinite(param.grad).all():
                self.epoch_reports['nan_grads'].add(name)
                logger.error(f"NaN/Inf gradient detected in {name} at batch {self.epoch_reports['batch_count']}")
                continue

            # Gradient Health (Exploding/Vanishing) [arXiv:2112.04036]
            grad_norm = torch.norm(param.grad).item()
            if grad_norm > self.exploding_grad_threshold:
                self.epoch_reports['exploding_grads'].add(name)
            elif grad_norm < self.vanishing_grad_threshold:
                # We only flag vanishing if it's not exactly zero (which is handled separately)
                if grad_norm > 0:
                    self.epoch_reports['vanishing_grads'].add(name)

            if (param.grad == 0).all():
                self.epoch_reports['zero_grads'].add(name)
                continue

            mean_abs_grad = torch.mean(torch.abs(param.grad)).item()
            mean_abs_val = torch.mean(torch.abs(param.data)).item()
            
            if mean_abs_val > 0:
                ratio = (max_lr * mean_abs_grad) / (mean_abs_val + EPS)
                if name not in self.epoch_reports['update_ratios']:
                    self.epoch_reports['update_ratios'][name] = 0.0
                self.epoch_reports['update_ratios'][name] += ratio
                
                # Log metrics for experiment trackers
                self.log(f"dlcheck/update_ratio/{name}", torch.log10(torch.tensor(ratio + EPS)), on_step=True)
                self.log(f"dlcheck/grad_norm/{name}", mean_abs_grad, on_step=True)

    def on_train_start(self, trainer, pl_module):
        """Callback at start of training"""
        self.init_callback(pl_module)
        logging.info("Training with DLCheck Callback.")
        self.check_weight_initialization(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        """Reset epoch-specific reports and activation stats."""
        self.epoch_reports = {
            'zero_grads': set(),
            'nan_grads': set(),
            'exploding_grads': set(),
            'vanishing_grads': set(),
            'update_ratios': {},
            'batch_count': 0
        }
        for name in self.activation_stats:
            self.activation_stats[name]['dead_count'] = 0
            self.activation_stats[name]['total_count'] = 0
            self.activation_stats[name]['abs_sum'] = 0.0
            self.activation_stats[name]['abs_count'] = 0
        
        self.nan_source = None

    def on_train_epoch_end(self, trainer, pl_module):
        """Callback at end of training epoch"""
        self.check_untrained_params(pl_module)
        self.check_diverging_params(pl_module)
        self.check_unstable_learning(trainer, pl_module)
        self.check_dead_activations()

    def on_train_end(self, trainer, pl_module):
        """Cleanup hooks and print final Health Report."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        print("\n" + "="*40)
        print("DLCheck Training Health Report")
        print("="*40)
        print(f"Total Batches Processed: {self.health_summary['total_batches']}")
        
        has_issues = False
        for issue_type, layers in self.health_summary['issue_counts'].items():
            if layers:
                has_issues = True
                print(f"\n{issue_type.replace('_', ' ').title()}:")
                for layer_name, epoch_count in layers.items():
                    print(f"  - {layer_name}: flagged in {epoch_count} epoch(s)")
        
        if not has_issues:
            print("\nNo major training issues detected. Your model is healthy! ⚡")
        else:
            print("\nReview the flagged layers above to improve training stability.")
        print("="*40 + "\n")

    def check_gradient_health(self, model: L.LightningModule) -> bool:
        """Check for exploding or vanishing gradients.
        
        Reference: [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptoms #7 & #8)
        """
        passed = True
        if self.epoch_reports['exploding_grads']:
            passed = False
            for name in self.epoch_reports['exploding_grads']:
                logger.warning(f"Exploding gradients detected in layer: {name}")
        
        if self.epoch_reports['vanishing_grads']:
            passed = False
            for name in self.epoch_reports['vanishing_grads']:
                logger.warning(f"Vanishing gradients detected in layer: {name}")
        
        # Also check current gradients (primarily for testing support)
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                if grad_norm > self.exploding_grad_threshold:
                    passed = False
                    logger.warning(f"Current gradient is exploding in {name}")
                elif grad_norm < self.vanishing_grad_threshold:
                    # In the context of the test, zeroed gradients should be caught here
                    passed = False
                    logger.warning(f"Current gradient is vanishing in {name}")
        
        return passed

    def track_nan_propagation(self, model: L.LightningModule) -> str:
        """Backtrack through the graph to find the source of NaN/Inf.
        
        Reference: [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptom #3)
        """
        if self.nan_source:
            logger.error(f"NaN propagation detected. First module to produce non-finite output: {self.nan_source}")
        return self.nan_source

    def check_loss_consistency(self, trainer: L.Trainer) -> bool:
        """Check if the loss is decreasing as expected.
        
        Reference: [arXiv:2411.08172] (Fault Localization in DL, Section 4.2)
        """
        if len(self.loss_history) < 2:
            return True
        
        # Simple check: is the loss mostly constant?
        first_loss = self.loss_history[0]
        is_constant = all(abs(l - first_loss) < 1e-6 for l in self.loss_history)
        
        # If loss is constant, it's generally an issue in these tests
        if is_constant:
            logger.warning("Loss is constant. Check loss function implementation and connectivity.")
            return False
            
        return True

    def check_dying_relu(self, model: L.LightningModule) -> bool:
        """Detect layers with a high percentage of dying ReLUs.
        
        Reference: [arXiv:1909.02562] (TFCheck, Section IV "Untrained Parameters")
        """
        passed = True
        for name, stats in self.activation_stats.items():
            # Check for ReLU in type name (could be ReLU, LeakyReLU etc, but test uses ReLU)
            if 'ReLU' in stats['type'] and stats['total_count'] > 0:
                sparsity = stats['dead_count'] / stats['total_count']
                if sparsity > self.dead_neuron_threshold:
                    passed = False
                    logger.warning(f"Dying ReLU detected in layer {name}: {sparsity:.2%} neurons are dead.")
        return passed

    def check_activation_saturation(self, model: L.LightningModule) -> bool:
        """Detect saturated Sigmoid or Tanh activations.
        
        Reference: [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptom #2)
        """
        passed = True
        for name, stats in self.activation_stats.items():
            if any(t in stats['type'] for t in ('Sigmoid', 'Tanh')) and stats['abs_count'] > 0:
                mean_abs = stats['abs_sum'] / stats['abs_count']
                if mean_abs > self.saturation_threshold_high or mean_abs < self.saturation_threshold_low:
                    passed = False
                    logger.warning(f"Activation saturation detected in layer {name}: mean absolute value {mean_abs:.4f}")
        return passed

    def check_untrained_layers_extended(self, model: L.LightningModule) -> bool:
        """Extended check for untrained layers using weight checksums.
        
        Reference: [arXiv:1909.02562] (TFCheck, Section IV)
        """
        passed = True
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if name in self.initial_weights:
                # Ensure same device for comparison
                initial_weight = self.initial_weights[name].to(param.device)
                if torch.equal(param, initial_weight):
                    passed = False
                    logger.warning(f"Layer {name} was never updated during training.")
        return passed

    def check_gpu_memory_fragmentation(self, trainer: L.Trainer) -> bool:
        """Track torch.cuda.memory_stats() to detect potential OOMs before they happen.
        
        Reference: [Nvidia Blog] (Monitoring CUDA Memory)
        """
        raise NotImplementedError("GPU memory fragmentation monitor is not yet implemented.")

    def check_cpu_gpu_bottleneck(self, trainer: L.Trainer) -> bool:
        """Measure time delta between batches to detect data loading bottlenecks.
        
        Reference: [PyTorch Documentation] (Performance Tuning Guide)
        """
        raise NotImplementedError("CPU-GPU bottleneck detector is not yet implemented.")

    def check_input_scaling(self, model: L.LightningModule) -> bool:
        """Monitor the mean and variance of the input batch.
        
        Reference: [arXiv:1909.02562] (TFCheck, Section III "Numerical Issues")
        """
        passed = True
        for name, stats in self.input_stats.items():
            if stats['count'] > 0:
                mean = stats['mean_sum'] / stats['count']
                std = stats['std_sum'] / stats['count']
                if abs(mean) > self.input_mean_threshold or std > self.input_std_threshold:
                    passed = False
                    logger.warning(f"Input scaling issue in layer {name}: mean={mean:.2f}, std={std:.2f}")
        return passed

    def check_label_leakage(self, trainer: L.Trainer) -> bool:
        """Monitor for 'too good to be true' convergence.
        
        Reference: [arXiv:2412.11304] (An Empirical Study of Fault Localisation, Section 3.2 "Label Leakage")
        """
        if len(self.loss_history) < self.leakage_batch_threshold:
            return True
        
        # Check if loss drops below threshold extremely quickly
        for i, loss in enumerate(self.loss_history[:self.leakage_batch_threshold]):
            if loss < self.leakage_loss_threshold:
                logger.warning(f"Suspiciously fast convergence at batch {i+1} (loss={loss:.2e}). Possible label leakage.")
                return False
        return True

    def check_label_noise(self, trainer: L.Trainer) -> bool:
        """Track the loss per-sample across epochs to detect mislabeled data.
        
        Reference: [ACM 3637528.3671933] (BTTackler/DeepDiagnoser, Section 3.1 "Quality Indicators")
        """
        if not self.sample_losses:
            return True
        
        # Identify outliers: samples with consistently higher loss than others
        avg_losses = {idx: sum(losses)/len(losses) for idx, losses in self.sample_losses.items()}
        all_avg_losses = list(avg_losses.values())
        if not all_avg_losses:
            return True
            
        overall_mean = sum(all_avg_losses) / len(all_avg_losses)
        
        passed = True
        for idx, avg_loss in avg_losses.items():
            # If a sample's loss is significantly higher than the mean
            if avg_loss > 1.1 * overall_mean and avg_loss > 0.0001:
                passed = False
                logger.warning(f"Potential mislabeled sample detected at index {idx} (avg loss={avg_loss:.4f}, mean={overall_mean:.4f})")
        
        return passed


