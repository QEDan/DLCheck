# DLCheck: Advanced Failure Modes Todo List

This document tracks the implementation of complex diagnostic checks for deep learning training, based on recent research in automated DL debugging.

## 1. Numerical & Optimization Issues
*Goal: Detect mathematical instabilities and gradient health.*

- [ ] **Gradient Exploding/Vanishing Detection**
  - **Protocol:** Monitor the L2 norm of gradients per layer. Flag layers where the norm exceeds a threshold (Exploding) or drops below $1e-7$ (Vanishing) for more than $N$ consecutive batches.
  - **Reference:** [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptoms #7 & #8)
- [ ] **NaN/Inf Propagation Tracking**
  - **Protocol:** When a NaN is detected in gradients, backtrack through the graph to identify the first module that produced a non-finite output.
  - **Reference:** [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptom #3)
- [ ] **Loss Inconsistency Check**
  - **Protocol:** Verify that the loss is decreasing over a window of batches. If loss is constant while gradients are non-zero, flag potential loss function implementation errors (e.g., missing `reduction='mean'`).
  - **Reference:** [arXiv:2411.08172] (Fault Localization in DL, Section 4.2 "Loss Function Faults")

## 2. Structural & Architectural Faults
*Goal: Identify layers that are poorly configured or disconnected.*

- [ ] **Dying ReLU Detection**
  - **Protocol:** Using forward hooks, calculate the sparsity of ReLU outputs. Flag layers where >95% of the neurons are consistently zero across a full epoch.
  - **Reference:** [arXiv:1909.02562] (TFCheck, Section IV "Untrained Parameters")
- [ ] **Activation Saturation (Sigmoid/Tanh)**
  - **Protocol:** Monitor the mean absolute value of activations for Sigmoid/Tanh layers. Flag if they are consistently $>0.95$ (saturated high) or $<0.05$ (saturated low).
  - **Reference:** [arXiv:2112.04036] (DeepDiagnosis, Section 3.2, Symptom #2)
- [ ] **Untrained Layer Detection (Extended)**
  - **Protocol:** Compare weight checksums at the start and end of training. Flag any layer with `requires_grad=True` that has exactly the same weights, indicating it was never reached by the optimizer.
  - **Reference:** [arXiv:1909.02562] (TFCheck, Section IV)

## 3. Data & Pipeline Integrity
*Goal: Detect issues in the data flow and label quality.*

- [ ] **Input Scaling Check**
  - **Protocol:** Monitor the mean and variance of the input batch (at the first layer's input hook). Flag if inputs are not roughly normalized (e.g., mean > 10 or std > 100).
  - **Reference:** [arXiv:1909.02562] (TFCheck, Section III "Numerical Issues")
- [ ] **Label Leakage Detection (Proxy)**
  - **Protocol:** Monitor for "too good to be true" convergence (e.g., loss drops to near zero in <5 batches on a complex task). Flag for potential inclusion of labels in features.
  - **Reference:** [arXiv:2412.11304] (An Empirical Study of Fault Localisation, Section 3.2 "Label Leakage")
- [ ] **Label Noise/Inconsistency**
  - **Protocol:** Track the loss per-sample across epochs. Samples that consistently have high loss while others converge may indicate mislabeled data.
  - **Reference:** [ACM 3637528.3671933] (BTTackler/DeepDiagnoser, Section 3.1 "Quality Indicators")

## 4. Metadata & Configuration
*Goal: Prevent environment and setup-related failures.*

- [ ] **Learning Rate Scheduler Plateau Check**
  - **Protocol:** If the LR has been reduced to its minimum value but the validation loss is still significantly higher than the training loss, flag potential overfitting or a learning rate that was reduced too aggressively.
  - **Reference:** [arXiv:2411.08172] (Fault Localization in DL, Section 3.1 "Dynamic Training Symptoms")
- [ ] **Weight Initialization Variance Check**
  - **Protocol:** Verify that the variance of initialized weights follows the expected scale for the layer type (e.g., Xavier/Kaiming).
  - **Reference:** [arXiv:1909.02562] (TFCheck, Section IV "Unbreaking Symmetry")

## 5. Learning Dynamics & Silent Bugs
*Goal: Detect high-level training pathologies from TheDeepChecker.*

- [ ] **Loss Oscillation Detection**
  - **Protocol:** Monitor the derivative of the smoothed loss curve. Flag if the sign of the derivative changes more than $K$ times within a small window.
  - **Reference:** [TOSEM 2023] (TheDeepChecker, Section 4.2 "Dynamic Properties")
- [ ] **Label Imbalance Sentry**
  - **Protocol:** Aggregate the count of each class index during the first epoch. Flag if the ratio between class counts exceeds a threshold (e.g., >10x).
  - **Reference:** [TOSEM 2023] (TheDeepChecker, Section 4.1 "Data Quality")
- [ ] **Training-Validation Divergence (Overfitting)**
  - **Protocol:** Monitor the gap between training and validation loss. Flag if validation loss increases while training loss continues to decrease.
  - **Reference:** [TOSEM 2023] (TheDeepChecker, Section 4.3 "Model Performance")
- [ ] **Data Loader / Augmentation Sanity Check**
  - **Protocol:** Compare statistics (mean/std) of batches. Flag if augmentations produce values outside of expected input ranges.
  - **Reference:** [TOSEM 2023] (TheDeepChecker, Section 4.1 "Preprocessing Defects")
