# DP-SGD Module

This module provides the core implementation of **Differentially Private Stochastic Gradient Descent (DP-SGD)**, including gradient clipping, noise addition, and private gradient computation.

## Overview

DP-SGD is the fundamental algorithm for training machine learning models with differential privacy. This module contains low-level primitives and utilities that power the higher-level APIs in the `keras` module.

## What is DP-SGD?

DP-SGD modifies standard SGD to provide formal privacy guarantees:

1. **Per-example gradients**: Compute gradient for each example separately
2. **Gradient clipping**: Clip each gradient to bound sensitivity
3. **Noise addition**: Add calibrated Gaussian noise to clipped gradients
4. **Privacy accounting**: Track cumulative privacy loss

## Key Components

### Gradient Computation (`gradients.py`)

Core classes for computing differentially private gradients:

- `GradientComputer`: Abstract base class for DP gradient computation
- `DpSgdGradientComputer`: Full DP-SGD with clipping and noise
- `ClippedGradientComputer`: Clipping only (no noise)
- `NonPrivateGradientComputer`: Standard gradient computation

Features:
- Per-example gradient computation
- Automatic gradient clipping
- Integration with noise addition via privatizers
- Support for both sample-level and user-level privacy

### Gradient Clipping (`grad_clipping.py`)

Utilities for clipping gradients to control sensitivity:

- `global_clipping()`: Clip entire gradient vector by L2 norm
- `per_layer_clipping()`: Clip each layer separately
- `adaptive_clipping()`: Automatically adjust clipping threshold
- Efficient implementations using JAX vectorization

Supporting utilities in `grad_clipping_utils.py`:
- Norm computation
- Clipping statistics tracking
- Memory-efficient implementations

### Optimizers (`optim.py`)

Privacy-aware optimizer utilities:

- Integration with Optax optimizers
- Privacy-preserving learning rate schedules
- Gradient accumulation support

### Type Definitions (`typing.py`)

Type aliases for improved code clarity:
- `InputsT`: Input data type
- `ParamsT`: Model parameters type
- `ModelStateT`: Model state type
- `NoiseStateT`: Noise state type

### Metrics Accumulator (`metrics_accumulator.py`)

Track training metrics while preserving privacy:
- Accumulate metrics across batches
- Privacy-safe aggregation
- Integration with training loops

### Wrapped Privatizer (`wrapped_privatizer.py`)

Bridge between gradient computation and noise addition:
- Wraps Optax GradientTransformations
- Manages noise state
- Coordinates clipping and noise

## Usage Example

```python
from jax_privacy.dp_sgd import gradients
from jax_privacy.noise_addition import gaussian_privatizer
import jax
import jax.numpy as jnp

# Define a simple model and loss
def model(params, x):
    return jnp.dot(x, params)

def loss_fn(params, batch):
    x, y = batch
    predictions = jax.vmap(lambda xi: model(params, xi))(x)
    return jnp.mean((predictions - y) ** 2)

# Create a gradient computer with DP-SGD
privatizer = gaussian_privatizer(
    stddev=1.0,  # Noise level
    prng_key=jax.random.key(0),
)

grad_computer = gradients.DpSgdGradientComputer(
    clipping_norm=1.0,  # Clip gradients to L2 norm of 1.0
    privatizer=privatizer,
)

# Initialize
params = jnp.zeros(10)
model_state = None
noise_state = privatizer.init(params)

# Compute DP gradient
batch = (jnp.ones((32, 10)), jnp.ones(32))
loss, grad, model_state, noise_state, metrics = grad_computer.compute_gradients(
    loss_fn=loss_fn,
    inputs=batch,
    params=params,
    model_state=model_state,
    noise_state=noise_state,
    rng_key=jax.random.key(1),
)

print(f"Loss: {loss}")
print(f"Clipped gradient norm: {jnp.linalg.norm(grad)}")
```

## Implementation Details

### Per-Example Gradients

DP-SGD requires computing gradients for each training example individually:

```python
# Automatic vectorization over batch
per_example_grads = jax.vmap(
    jax.grad(loss_fn),
    in_axes=(None, 0)
)(params, batch)
```

### Gradient Clipping

Clip each per-example gradient to bounded sensitivity:

```python
# Clip to max L2 norm
def clip_gradient(grad, max_norm):
    grad_norm = jnp.linalg.norm(grad)
    return grad * jnp.minimum(1.0, max_norm / grad_norm)
```

### Privacy Guarantees

The combination of clipping and noise provides (ε, δ)-differential privacy:
- **Clipping norm** (C): Controls sensitivity
- **Noise multiplier** (σ): Ratio of noise stddev to clipping norm
- Higher σ → more privacy, less utility

## Advanced Features

### Sharded Training

Support for distributed training:
- Gradient clipping across devices
- Noise generation with JAX sharding
- See `grad_clipping_sharded_test.py` for examples

### Adaptive Clipping

Automatically adjust clipping threshold:
- Track gradient norm statistics
- Adapt to maintain target quantile
- Reduces hyperparameter tuning

## Best Practices

1. **Clipping norm**: Start with 1.0, adjust based on gradient magnitudes
2. **Noise multiplier**: Use accounting to find appropriate value
3. **Batch size**: Larger batches improve privacy-utility tradeoff
4. **Learning rate**: May need to increase compared to non-private training
5. **Architecture**: Some architectures work better with DP (e.g., GroupNorm vs BatchNorm)

## References

- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- [Scalable Private Learning with PATE](https://arxiv.org/abs/1802.08908)
- [Large Scale Private Learning via Low-rank Reparametrization](https://arxiv.org/abs/2106.09352)

## Related Modules

- `accounting`: Compute privacy guarantees
- `noise_addition`: Privatizer implementations
- `keras`: High-level API using this module
- `experimental`: Experimental optimizations
