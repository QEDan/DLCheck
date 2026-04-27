# DLCheck
Health checks for training issues in deep learning models

This library implements a suite of health checks for training deep learning models. These methods provide feedback during training that can help machine learning developers quickly identify and correct common training issues like disconnected layers, unstable parameters, exploding gradients, vanishing gradients, zero loss, saturated neurons, dead neurons, etc.

## References

*   **TFCheck**: Houssem Ben Braiek, Foutse Khomh, [TFCheck: A TensorFlow Library for Detecting Training Issues in Neural Network Programs](https://arxiv.org/abs/1909.02562) [arXiv:1909.02562]
*   **DeepDiagnosis**: Mohammad Wardat, Breno Dantas Cruz, Wei Le, Hridesh Rajan [DeepDiagnosis: Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep Learning Programs](https://arxiv.org/abs/2112.04036) [arXiv:2112.04036]. [DeepDiagnosis is on Github](https://github.com/DeepDiagnosis/ICSE2022).
*   **Fault Localization Survey**: [Fault Localization in Deep Learning: A Survey](https://arxiv.org/abs/2411.08172) [arXiv:2411.08172]
*   **Empirical Study**: [An Empirical Study of Fault Localisation in Deep Learning Programs](https://arxiv.org/abs/2412.11304) [arXiv:2412.11304]
*   **TheDeepChecker**: [TheDeepChecker: An Automated Testing Framework for Preprocessing Defects in Deep Learning Systems](https://doi.org/10.1145/3637528.3671933) [TOSEM 2023 / ACM 3637528.3671933]
*   **BTTackler/DeepDiagnoser**: Referenced for Label Noise detection [ACM 3637528.3671933]
*   **Nvidia Blog**: [Monitoring CUDA Memory](https://developer.nvidia.com/blog/monitoring-cuda-memory-usage-and-fragmentation/)
*   **PyTorch Documentation**: [Performance Tuning Guide](https://pytorch.org/docs/stable/notes/cuda.html#best-practices)
