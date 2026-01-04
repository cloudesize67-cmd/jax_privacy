# Privacy Accounting Module

This module provides tools for computing differential privacy guarantees for machine learning algorithms, particularly DP-SGD (Differentially Private Stochastic Gradient Descent).

## Overview

Privacy accounting is essential for determining the privacy budget consumed during training. This module helps you:

- **Compute privacy guarantees**: Calculate epsilon (ε) and delta (δ) values for your training process
- **Calibrate parameters**: Find optimal noise levels, batch sizes, or training steps given privacy constraints
- **Track privacy consumption**: Monitor privacy budget usage throughout training

## Key Components

### Accountants

The module supports multiple privacy accounting methods:

- **RDP (Rényi Differential Privacy) Accountant**: Fast and tight bounds using Rényi divergence
- **PLD (Privacy Loss Distribution) Accountant**: More accurate bounds using privacy loss distributions
- **DP Accountant**: Base interface for privacy accounting

Available configurations:
- `RdpAccountantConfig`: Configure RDP accountant with custom orders
- `PldAccountantConfig`: Configure PLD accountant with discretization settings
- `DpAccountantConfig`: Base configuration interface

### Training Accountants

Track privacy consumption over the course of training:

- `DpsgdTrainingAccountant`: Standard DP-SGD accounting
- `DpsgdTrainingUserLevelAccountant`: User-level DP-SGD accounting
- `DpTrainingAccountant`: General DP training accounting interface
- `SingleReleaseTrainingAccountant`: For single model release scenarios
- `CachedExperimentAccountant`: Caches accounting results for efficiency

### Calibration Functions

Find optimal hyperparameters given privacy constraints:

- `calibrate_noise_multiplier()`: Determine noise level given (ε, δ) and other parameters
- `calibrate_batch_size()`: Find batch size given privacy budget
- `calibrate_num_updates()`: Compute maximum training steps within budget

### Supporting Types

- `DpParams`: Container for differential privacy parameters (epsilon, delta)
- `SamplingMethod`: Enum for different sampling strategies
- `BatchingScaleSchedule`: Schedule for adjusting batch sizes during training

## Usage Example

```python
from jax_privacy.accounting import (
    calibrate_noise_multiplier,
    DpsgdTrainingAccountant,
    RdpAccountantConfig,
    DpParams,
)

# Calibrate noise multiplier for target privacy budget
noise_multiplier = calibrate_noise_multiplier(
    epsilon=3.0,
    delta=1e-5,
    num_samples=50000,
    batch_size=256,
    num_updates=10000,
)

# Create an accountant to track privacy during training
accountant = DpsgdTrainingAccountant(
    num_samples=50000,
    batch_size=256,
    noise_multiplier=noise_multiplier,
    accountant_config=RdpAccountantConfig(),
)

# Get current privacy guarantee after some training steps
epsilon = accountant.compute_epsilon(
    num_updates=5000,
    delta=1e-5,
)
print(f"After 5000 steps: ε = {epsilon:.2f}")
```

## Privacy Budgets

Differential privacy guarantees are specified using two parameters:

- **Epsilon (ε)**: Privacy loss parameter. Smaller is better (more private)
  - ε = 0: Perfect privacy (unachievable in practice)
  - ε = ln(3) ≈ 1.1: Common target for strong privacy
  - ε = 10: Weaker privacy, better utility

- **Delta (δ)**: Failure probability. Should be much smaller than 1/dataset_size
  - Typical values: 1e-5, 1e-6, 1e-7

## References

For more details on differential privacy and accounting methods, see:
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Rényi Differential Privacy](https://arxiv.org/abs/1702.07476)
- [Privacy Loss Distributions](https://arxiv.org/abs/2106.08567)
