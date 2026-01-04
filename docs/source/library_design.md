# Library Design

## Overview

JAX Privacy is designed with a modular, layered architecture that provides both high-level convenience APIs and low-level primitives for differential privacy in machine learning. The library emphasizes functional purity, immutability, and composability—core principles of the JAX ecosystem.

## Design Principles

### 1. Functional and Immutable

Following JAX conventions, JAX Privacy embraces:
- **Pure functions**: No hidden state or side effects
- **Immutability**: All state is explicit and passed through function calls
- **Functional transformations**: Compatible with `jax.jit`, `jax.vmap`, `jax.grad`

### 2. Layered Architecture

The library provides three levels of abstraction:

#### High-Level APIs (Keras)
- **Module**: `jax_privacy.keras`
- **Target users**: Practitioners who want privacy with minimal code changes
- **Characteristics**: Simple configuration, automatic parameter calibration, familiar Keras interface
- **Use cases**: Fine-tuning LLMs, standard model training

#### Mid-Level APIs (DP-SGD + Noise Addition)
- **Modules**: `jax_privacy.dp_sgd`, `jax_privacy.noise_addition`
- **Target users**: Researchers and engineers who need flexibility
- **Characteristics**: Explicit control over clipping, noise, and gradients
- **Use cases**: Custom training loops, novel architectures, research experiments

#### Low-Level Primitives (Experimental)
- **Module**: `jax_privacy.experimental`
- **Target users**: Advanced researchers implementing new DP algorithms
- **Characteristics**: Maximum control, research-oriented, may have rough edges
- **Use cases**: Algorithm development, performance optimization, cutting-edge research

### 3. Composability with Optax

JAX Privacy integrates seamlessly with Optax (JAX's optimization library):
- Privatizers implement the `optax.GradientTransformation` interface
- Can be composed with any Optax optimizer using `optax.chain()`
- Privacy guarantees preserved by post-processing property

### 4. Modular Components

The library is organized into focused, single-responsibility modules:

```
jax_privacy/
├── accounting/     # Privacy budget tracking and calibration
├── auditing/       # Empirical privacy validation
├── dp_sgd/         # Core DP-SGD implementation
├── keras/          # High-level Keras API
├── noise_addition/ # Privatizers (noise mechanisms)
├── matrix_factorization/  # Advanced noise strategies
└── experimental/   # Research components
```

Each module can be used independently or combined as needed.

## Core Abstractions

### Privatizers

**Location**: `jax_privacy.noise_addition`

Privatizers are stateful transformations that add calibrated noise to gradients:

```python
class Privatizer(optax.GradientTransformation):
    def init(self, params) -> NoiseState:
        """Initialize noise state."""

    def update(self, gradients, state) -> (noisy_gradients, new_state):
        """Add noise to gradients and update state."""
```

**Key implementations**:
- `gaussian_privatizer`: I.i.d. Gaussian noise (standard DP-SGD)
- `matrix_factorization_privatizer`: Correlated noise for better privacy-utility

**Design rationale**: By implementing the Optax interface, privatizers compose naturally with existing optimization workflows while encapsulating privacy-specific logic.

### Gradient Computers

**Location**: `jax_privacy.dp_sgd.gradients`

Gradient computers handle per-example gradient computation, clipping, and noise integration:

```python
class GradientComputer:
    def compute_gradients(
        self,
        loss_fn,
        inputs,
        params,
        model_state,
        noise_state,
        rng_key
    ) -> (loss, grads, model_state, noise_state, metrics):
        """Compute differentially private gradients."""
```

**Key implementations**:
- `DpSgdGradientComputer`: Full DP-SGD with clipping and noise
- `ClippedGradientComputer`: Clipping only (no privacy)
- `NonPrivateGradientComputer`: Standard gradients (baseline)

**Design rationale**: Separating gradient computation from noise addition allows flexible composition and makes it easy to disable privacy for debugging or comparison.

### Accountants

**Location**: `jax_privacy.accounting`

Accountants track privacy budget consumption during training:

```python
class DpTrainingAccountant:
    def compute_epsilon(self, num_updates, delta) -> epsilon:
        """Compute privacy spent after num_updates steps."""
```

**Key implementations**:
- `DpsgdTrainingAccountant`: Standard DP-SGD accounting
- `DpsgdTrainingUserLevelAccountant`: User-level privacy
- `CachedExperimentAccountant`: Efficient caching for repeated queries

**Design rationale**: Separating accounting from training logic keeps privacy guarantees explicit and verifiable, and supports multiple accounting methods (RDP, PLD).

## State Management

JAX Privacy follows JAX's explicit state management:

### Types of State

1. **Model Parameters** (`params`): Trainable weights
2. **Model State** (`model_state`): Non-trainable state (e.g., batch norm statistics)
3. **Optimizer State** (`opt_state`): Optimizer accumulators (momentum, etc.)
4. **Noise State** (`noise_state`): PRNG keys and noise history
5. **Accounting State**: Privacy budget tracking (typically stateless functions)

### State Flow

```python
# Typical training step
def train_step(params, model_state, noise_state, opt_state, batch, rng):
    # Compute DP gradients
    loss, grads, model_state, noise_state, metrics = grad_computer.compute_gradients(
        loss_fn, batch, params, model_state, noise_state, rng
    )

    # Apply optimizer
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, model_state, noise_state, opt_state, metrics
```

All state is explicitly threaded through functions—no hidden global state.

## Privacy Guarantees

### Composition

Privacy guarantees compose over training:
- Each DP-SGD step consumes privacy budget
- Accountants track cumulative privacy loss
- Different accounting methods (RDP, PLD) provide different tightness

### Post-Processing

Privacy is preserved under post-processing:
- Applying non-private transformations after privatizer preserves privacy
- `optax.chain(privatizer, optimizer)` maintains privacy guarantees
- Any function of DP outputs is also DP

## Performance Considerations

### Vectorization

JAX Privacy leverages JAX's vectorization for efficient per-example gradients:

```python
# Automatic vectorization over batch
per_example_grads = jax.vmap(
    jax.grad(loss_fn),
    in_axes=(None, 0)  # Broadcast params, vectorize over batch
)(params, batch)
```

### JIT Compilation

All core functions are JIT-compatible:
- Gradient computation, clipping, and noise addition can be compiled
- Entire training steps compile to efficient XLA code
- Use `jax.jit` for maximum performance

### Sharding

Multi-device training is supported:
- Gradient clipping across devices
- Distributed noise generation
- Compatible with JAX's `jax.sharding` API

## Extension Points

### Custom Privatizers

Implement `optax.GradientTransformation` for custom noise strategies:

```python
def my_custom_privatizer(stddev, prng_key):
    def init_fn(params):
        return {'key': prng_key}

    def update_fn(gradients, state):
        key, new_key = jax.random.split(state['key'])
        # Custom noise generation logic
        noise = my_custom_noise(key, gradients.shape)
        return gradients + stddev * noise, {'key': new_key}

    return optax.GradientTransformation(init_fn, update_fn)
```

### Custom Accountants

Extend base accountant classes for new accounting methods:

```python
class MyAccountantConfig(DpAccountantConfig):
    def create_accountant(self):
        # Return your custom dp_accounting.PrivacyAccountant
        return my_custom_accountant
```

### Custom Clipping

Implement custom clipping strategies in `dp_sgd.grad_clipping`:

```python
def my_clipping_fn(grads, max_norm):
    # Custom clipping logic
    return clipped_grads, clipping_stats
```

## Testing and Validation

### Unit Tests

- Each module has comprehensive `*_test.py` files
- Tests use `hypothesis` for property-based testing
- Isolated tests use `chex.set_n_cpu_devices()` for multi-device simulation

### Privacy Auditing

The `auditing` module enables empirical validation:
- Canary attacks measure actual privacy leakage
- Compare empirical privacy with theoretical guarantees
- Detect implementation bugs

### Type Checking

- Extensive use of type annotations
- Validated with `pytype`
- Generic types for flexibility with strong typing

## External Contributions

We welcome contributions that align with our design principles:

- **Bug fixes**: Small patches for correctness issues
- **Documentation**: Improvements to docs, examples, and guides
- **Performance**: Optimizations that preserve correctness
- **Research**: Novel DP algorithms in `experimental/`

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines, including:
- Code style (follows Google Python Style Guide)
- Docstring requirements (public APIs need full docs + doctests)
- Type annotation standards
- Review process

**Note**: We prioritize stability and correctness over feature breadth. Major architectural changes require discussion before implementation.

## Related Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Optax Documentation](https://optax.readthedocs.io/)
- [Differential Privacy Theory](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [DP-SGD Paper](https://arxiv.org/abs/1607.00133)

## Design Evolution

JAX Privacy has evolved from research-focused code to a production-ready library:

- **v1.0+**: Production APIs (Keras, simplified interfaces)
- **v0.x**: Research components (experiments, matrix factorization)
- **Future**: Additional DP algorithms (DP-FTRL, etc.)

The modular design allows both stability in production APIs and flexibility in research components.
