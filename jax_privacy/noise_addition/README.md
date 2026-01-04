# Noise Addition Module

This module implements **privatizers** - gradient transformations that add carefully calibrated noise to gradients to achieve differential privacy.

## Overview

Privatizers are the final step in DP-SGD: after gradients are clipped and aggregated, noise is added to ensure privacy. This module provides implementations of various noise-addition schemes as Optax `GradientTransformation` objects.

## What are Privatizers?

Privatizers are **stateful** transformations that:

1. Take clipped + aggregated gradients as input
2. Add calibrated noise to provide privacy
3. Return noisy gradients and updated state
4. Track internal state (random keys, noise history, etc.)

The stateful nature allows for sophisticated noise-addition strategies beyond simple i.i.d. Gaussian noise.

## Key Components

### Gaussian Privatizer

The standard privatizer using i.i.d. Gaussian noise:

```python
from jax_privacy.noise_addition import gaussian_privatizer
import jax

privatizer = gaussian_privatizer(
    stddev=1.0,              # Noise standard deviation
    prng_key=jax.random.key(0),  # Random seed
)
```

Features:
- Adds independent Gaussian noise to each gradient
- State contains only the PRNG key
- Simplest and most commonly used privatizer

### Matrix Factorization Privatizer

Advanced privatizer using correlated noise:

```python
from jax_privacy.noise_addition import matrix_factorization_privatizer
import jax

privatizer = matrix_factorization_privatizer(
    stddev=1.0,
    prng_key=jax.random.key(0),
    strategy='streaming',     # Noise correlation strategy
)
```

Features:
- Adds **correlated** noise across time steps
- Can improve privacy-utility tradeoff
- Richer state representation
- More computationally intensive

Supported strategies (see `SupportedStrategies` enum):
- `'streaming'`: Streaming matrix factorization
- `'banded'`: Banded matrix structure
- `'toeplitz'`: Toeplitz matrix structure

## Usage Pattern

All privatizers follow the same Optax `GradientTransformation` interface:

```python
from jax_privacy.noise_addition import gaussian_privatizer
import jax
import jax.numpy as jnp

# Create privatizer
privatizer = gaussian_privatizer(
    stddev=1.0,
    prng_key=jax.random.key(0),
)

# Initialize state
model_params = jnp.zeros(10)
noise_state = privatizer.init(model_params)

# Use in training loop
for step in range(100):
    # Compute clipped gradient (from your training code)
    sum_of_clipped_grads = jnp.ones(10)  # Example

    # Add noise
    noisy_grad, noise_state = privatizer.update(
        sum_of_clipped_grads,
        noise_state,
    )

    # Apply noisy gradient to model
    model_params = model_params - 0.01 * noisy_grad
```

## Integration with Optax

Since privatizers are `GradientTransformation` objects, they compose naturally with Optax:

```python
import optax
from jax_privacy.noise_addition import gaussian_privatizer

# Create privatizer and optimizer
privatizer = gaussian_privatizer(stddev=1.0, prng_key=jax.random.key(0))
optimizer = optax.adam(learning_rate=0.001)

# Chain them together
combined = optax.chain(
    privatizer,      # First add noise
    optimizer,       # Then apply optimizer
)

# Use combined transformation
opt_state = combined.init(params)
updates, opt_state = combined.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

**Important**: By the post-processing property of differential privacy, chaining privatizers with other transformations **preserves privacy guarantees**.

## Complete Example

```python
from jax_privacy.noise_addition import gaussian_privatizer
import jax
import jax.numpy as jnp
import optax

# Setup
prng_key = jax.random.key(42)
model_params = jnp.zeros(10)

# Create privatizer
privatizer = gaussian_privatizer(
    stddev=1.0,
    prng_key=prng_key,
)

# Create optimizer
optimizer = optax.adam(learning_rate=0.01)

# Combine privatizer and optimizer
combined_transform = optax.chain(privatizer, optimizer)

# Initialize states
combined_state = combined_transform.init(model_params)

# Training step
def train_step(params, state, clipped_grads):
    # Privatizer adds noise, then optimizer applies update
    updates, new_state = combined_transform.update(
        clipped_grads,
        state,
    )
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state

# Simulate training
for step in range(10):
    # Assume we computed clipped gradients
    clipped_grads = jnp.ones(10) * 0.1

    model_params, combined_state = train_step(
        model_params,
        combined_state,
        clipped_grads,
    )

    print(f"Step {step}: params norm = {jnp.linalg.norm(model_params):.3f}")
```

## Noise Calibration

The `stddev` parameter controls the privacy-utility tradeoff:

- **Noise Multiplier** (σ): `stddev / clipping_norm`
- **Higher stddev**: More privacy, less utility
- **Lower stddev**: Less privacy, more utility

Use the `accounting` module to find the right noise level:

```python
from jax_privacy.accounting import calibrate_noise_multiplier

noise_multiplier = calibrate_noise_multiplier(
    epsilon=3.0,
    delta=1e-5,
    num_samples=50000,
    batch_size=256,
    num_updates=10000,
)

# Convert to stddev
clipping_norm = 1.0
stddev = noise_multiplier * clipping_norm

privatizer = gaussian_privatizer(stddev=stddev, prng_key=key)
```

## Advanced: Matrix Factorization

Matrix factorization privatizers add **correlated** noise across training steps:

```python
from jax_privacy.noise_addition import (
    matrix_factorization_privatizer,
    SupportedStrategies,
)

privatizer = matrix_factorization_privatizer(
    stddev=1.0,
    prng_key=jax.random.key(0),
    strategy=SupportedStrategies.STREAMING,
)
```

Benefits:
- Can achieve better privacy-utility tradeoffs
- Particularly useful for longer training runs
- Implements ideas from matrix factorization mechanism

Tradeoffs:
- More complex state management
- Higher computational cost
- More memory usage

See `jax_privacy/matrix_factorization/` for implementation details.

## Implementation Details

### State Management

Privatizer state typically contains:
- **PRNG key**: For generating random noise
- **Step counter**: Track training progress
- **Noise history**: For correlated noise strategies
- **Auxiliary variables**: Strategy-specific state

### Noise Generation

Gaussian privatizer generates noise as:
```python
noise = stddev * jax.random.normal(key, shape=gradient.shape)
noisy_gradient = gradient + noise
```

### Sharding Support

Privatizers work with JAX sharding for distributed training:
- Noise generation respects device placement
- See `sharding_utils.py` for utilities
- `distributed_noise_generation_test.py` for examples

## Best Practices

1. **Use gaussian_privatizer** unless you have specific needs
2. **Set PRNG seed** for reproducibility
3. **Calibrate noise** using accounting module
4. **Compose with Optax** for full training pipeline
5. **Monitor noise magnitude** relative to gradients

## Common Patterns

### With Manual Gradient Clipping

```python
# Clip gradients manually
clipped_grads = clip_gradients(raw_grads, max_norm=1.0)

# Add noise
noisy_grads, noise_state = privatizer.update(clipped_grads, noise_state)

# Apply optimizer
updates, opt_state = optimizer.update(noisy_grads, opt_state)
params = optax.apply_updates(params, updates)
```

### With DP-SGD Module

The `dp_sgd` module handles clipping and noise together:

```python
from jax_privacy.dp_sgd import DpSgdGradientComputer

grad_computer = DpSgdGradientComputer(
    clipping_norm=1.0,
    privatizer=privatizer,  # Privatizer from this module
)
```

## Testing and Validation

Verify your privatizer:

```python
# Check noise is actually added
grad_no_noise = jnp.zeros(10)
noisy_grad, _ = privatizer.update(grad_no_noise, noise_state)
noise_magnitude = jnp.linalg.norm(noisy_grad)
print(f"Noise magnitude: {noise_magnitude:.3f}")
# Should be approximately stddev * sqrt(dimension)
```

## References

- [DP-SGD Paper](https://arxiv.org/abs/1607.00133)
- [Matrix Factorization Mechanism](https://arxiv.org/abs/1906.03853)
- [Optax Documentation](https://optax.readthedocs.io/)

## Related Modules

- `dp_sgd`: Uses privatizers for DP training
- `accounting`: Calibrate noise levels
- `matrix_factorization`: Advanced noise strategies
- `keras`: High-level API using privatizers
