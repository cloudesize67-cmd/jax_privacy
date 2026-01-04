# Privacy Auditing Module

This module provides tools for **empirical privacy auditing** to validate and estimate the actual privacy guarantees of trained models.

## Overview

While privacy accounting provides theoretical guarantees, privacy auditing empirically measures privacy leakage through controlled attacks. This helps:

- **Validate theoretical bounds**: Check if actual privacy matches theoretical guarantees
- **Detect implementation errors**: Find bugs in privacy implementations
- **Estimate privacy empirically**: Measure privacy when theoretical analysis is unavailable

## What is Privacy Auditing?

Privacy auditing uses **canary attacks** to measure information leakage:

1. **Canaries**: Special data points (real or synthetic) inserted into the training set
2. **Attack**: Train models with and without each canary
3. **Scoring**: Measure how well an attacker can distinguish whether a canary was in the training set
4. **Estimation**: Convert attack scores into privacy estimates

## Key Components

### Canary Score Auditor

The main component is `canary_score_auditor.py`, which provides:

- **Statistical estimation**: Estimate epsilon (ε) from attack scores
- **Bootstrap confidence intervals**: Quantify uncertainty in privacy estimates
- **Hypothesis testing**: Test whether privacy guarantees are satisfied
- **Multiple attack strategies**: Support for various membership inference attacks

### Key Functions

- `estimate_privacy()`: Estimate epsilon from canary attack scores
- `compute_confidence_interval()`: Bootstrap confidence intervals
- `test_privacy_hypothesis()`: Test if model satisfies (ε, δ)-DP

### Configuration

- `BootstrapParams`: Configure bootstrap resampling for uncertainty quantification
  - `num_samples`: Number of bootstrap samples
  - `quantiles`: Quantiles for confidence intervals (e.g., 0.025, 0.975 for 95% CI)
  - `seed`: Random seed for reproducibility

## Usage Example

```python
from jax_privacy.auditing import canary_score_auditor
import numpy as np

# Attack scores for canaries included in training (higher = more confident)
scores_in = np.array([0.8, 0.9, 0.85, 0.88, 0.92])

# Attack scores for canaries excluded from training (lower = more confident)
scores_out = np.array([0.3, 0.25, 0.35, 0.28, 0.32])

# Estimate privacy from attack scores
bootstrap_params = canary_score_auditor.BootstrapParams(
    num_samples=1000,
    quantiles=(0.025, 0.975),  # 95% confidence interval
    seed=42,
)

epsilon_estimate, confidence_interval = canary_score_auditor.estimate_privacy(
    scores_in=scores_in,
    scores_out=scores_out,
    delta=1e-5,
    bootstrap_params=bootstrap_params,
)

print(f"Estimated ε: {epsilon_estimate:.2f}")
print(f"95% CI: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
```

## How It Works

The auditor uses attack scores to empirically estimate privacy:

1. **Collect scores**: Run membership inference attacks on held-in and held-out canaries
2. **Compute advantage**: Measure attacker's ability to distinguish training vs. non-training data
3. **Convert to epsilon**: Transform attack advantage into ε estimate
4. **Bootstrap**: Resample to compute confidence intervals

## Best Practices

- **Multiple canaries**: Use many canaries for reliable estimates (typically 100+)
- **Representative attacks**: Use realistic attack strategies
- **Compare with theory**: Check if empirical ε matches theoretical guarantees
- **Confidence intervals**: Always report uncertainty in estimates
- **Diverse canaries**: Include various types of data points

## Interpretation

- **ε_empirical < ε_theory**: Good! Actual privacy may be better than guaranteed
- **ε_empirical ≈ ε_theory**: Expected behavior, theory matches practice
- **ε_empirical > ε_theory**: Warning! May indicate implementation bugs or invalid assumptions

## References

For more details on privacy auditing:
- [Auditing Differentially Private Machine Learning](https://arxiv.org/abs/2006.07709)
- [Canary in the Coal Mine](https://arxiv.org/abs/2101.05836)
- [Membership Inference Attacks](https://arxiv.org/abs/1610.05820)

## Related Modules

- `accounting`: Theoretical privacy guarantees
- `dp_sgd`: Implementation to audit
