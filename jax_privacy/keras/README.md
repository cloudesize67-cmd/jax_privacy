# Keras API Module

This module provides a **high-level API for adding differential privacy to Keras models**, making it easy to train models with formal privacy guarantees using familiar Keras workflows.

## Overview

The Keras API is the **easiest way** to add differential privacy to your machine learning models. It handles all the complexity of DP-SGD while maintaining the simple Keras interface you know.

Perfect for:
- Fine-tuning Large Language Models (LLMs) with privacy
- Training standard neural networks with DP
- Rapid prototyping of private ML models
- Users who want privacy without low-level details

## Quick Start

```python
from jax_privacy.keras import DPKerasConfig, DPModel
import keras

# 1. Define your model as usual
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

# 2. Configure differential privacy
dp_config = DPKerasConfig(
    epsilon=3.0,           # Privacy budget
    delta=1e-5,            # Failure probability
    clipping_norm=1.0,     # Gradient clipping threshold
)

# 3. Wrap your model
dp_model = DPModel(model, dp_config)

# 4. Train as usual!
dp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

dp_model.fit(x_train, y_train, epochs=10, batch_size=256)
```

That's it! Your model now trains with (ε=3.0, δ=1e-5)-differential privacy.

## Key Components

### DPKerasConfig

Configuration for differential privacy parameters:

```python
@dataclass
class DPKerasConfig:
    epsilon: float              # Privacy budget (smaller = more private)
    delta: float               # Failure probability (typically 1e-5)
    clipping_norm: float       # Gradient clipping threshold
    num_samples: int           # Total training examples
    batch_size: int            # Training batch size
    num_updates: int           # Total training steps (optional)
    noise_multiplier: float    # Noise level (auto-calibrated if not set)
    accountant_config: ...     # Privacy accounting method (RDP/PLD)
```

### DPModel

Wrapper that adds DP-SGD to any Keras model:

- **Automatic noise calibration**: Finds the right noise level for your (ε, δ) budget
- **Privacy accounting**: Tracks privacy consumption during training
- **Seamless integration**: Works with standard Keras APIs
- **Gradient clipping**: Automatically clips gradients
- **Batch handling**: Manages per-example gradient computation

### Functions

- `make_dp_model_fn()`: Create a privacy-preserving model function
- `compute_privacy_spent()`: Check privacy budget during training
- `calibrate_dp_params()`: Auto-tune DP parameters

## Complete Example

```python
from jax_privacy.keras import DPKerasConfig, DPModel
from jax_privacy.accounting import RdpAccountantConfig
import keras
import numpy as np

# Prepare data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Define model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax'),
])

# Configure DP with automatic noise calibration
dp_config = DPKerasConfig(
    epsilon=3.0,                           # Target privacy budget
    delta=1e-5,                            # Target delta
    clipping_norm=1.0,                     # Clip gradients
    num_samples=len(x_train),              # Dataset size
    batch_size=256,                        # Batch size
    epochs=10,                             # Number of epochs
    accountant_config=RdpAccountantConfig(), # Use RDP accounting
)

# Create DP model
dp_model = DPModel(model, dp_config)

# Compile with optimizer and loss
dp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train with privacy
history = dp_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=256,
    validation_data=(x_test, y_test),
)

# Check final privacy spent
epsilon_spent = dp_model.compute_epsilon()
print(f"Privacy spent: ε = {epsilon_spent:.2f}")
```

## Fine-Tuning LLMs with Privacy

The Keras API excels at private fine-tuning of large language models:

```python
from jax_privacy.keras import DPKerasConfig, DPModel
import keras_nlp

# Load pre-trained model
model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

# Configure DP for fine-tuning
dp_config = DPKerasConfig(
    epsilon=8.0,              # Privacy budget for fine-tuning
    delta=1e-6,
    clipping_norm=0.1,        # Lower for LLMs
    num_samples=10000,        # Fine-tuning dataset size
    batch_size=8,             # Smaller batch for LLMs
)

# Apply DP
dp_model = DPModel(model, dp_config)

# Fine-tune with privacy
dp_model.fit(fine_tuning_data, epochs=3)
```

See `examples/dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb` for a complete tutorial.

## Privacy Parameters Guide

### Epsilon (ε)

Privacy budget - smaller is more private:
- **ε < 1**: Very strong privacy, significant utility loss
- **ε ≈ 1-3**: Good privacy-utility balance
- **ε ≈ 3-10**: Moderate privacy, better utility
- **ε > 10**: Weak privacy guarantees

### Delta (δ)

Failure probability - should be ≪ 1/dataset_size:
- For 10K samples: δ = 1e-5
- For 100K samples: δ = 1e-6
- For 1M samples: δ = 1e-7

### Clipping Norm

Controls gradient sensitivity:
- **Start with 1.0** for standard models
- **Use 0.1-0.5** for large models (LLMs)
- **Monitor gradient norms** during training
- **Too low**: Slow convergence
- **Too high**: Weak privacy guarantees

### Batch Size

Larger batches improve privacy-utility tradeoff:
- **Minimum**: 256 for image models
- **Recommended**: 512-2048
- **LLMs**: Often limited by memory (8-32)

## Advanced Usage

### Custom Privacy Accounting

```python
from jax_privacy.accounting import PldAccountantConfig

dp_config = DPKerasConfig(
    epsilon=3.0,
    delta=1e-5,
    accountant_config=PldAccountantConfig(
        discretization=1e-4,  # Higher accuracy
    ),
    ...
)
```

### Manual Noise Calibration

```python
# If you already know the noise multiplier
dp_config = DPKerasConfig(
    noise_multiplier=1.2,  # Manually set noise
    clipping_norm=1.0,
    # epsilon and delta used for tracking only
    ...
)
```

### Privacy Callbacks

```python
class PrivacyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epsilon = self.model.compute_epsilon()
        print(f"Epoch {epoch}: ε = {epsilon:.2f}")

dp_model.fit(
    x_train, y_train,
    callbacks=[PrivacyCallback()],
)
```

## Best Practices

1. **Start with examples**: Use provided notebooks as templates
2. **Tune hyperparameters**: Adjust learning rate and clipping norm
3. **Use large batches**: Bigger batches = better privacy-utility
4. **Monitor privacy**: Track epsilon during training
5. **Validate results**: Compare with non-private baseline
6. **Use GroupNorm**: Avoid BatchNorm in DP models

## Common Issues

### Model Not Learning
- Increase learning rate (try 2-5x non-private rate)
- Decrease clipping norm if gradients are too large
- Increase epsilon budget
- Use larger batch sizes

### Privacy Budget Exceeded
- Reduce number of epochs
- Increase noise multiplier
- Use larger batch size
- Increase epsilon target

### Out of Memory
- Reduce batch size (but maintain privacy-utility)
- Use gradient accumulation
- Consider per-layer clipping

## Examples

See the `examples/` directory for complete tutorials:
- `keras_api_example.py`: Basic MNIST example
- `dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb`: LLM fine-tuning
- `dp_sgd_flax_linen_mnist.ipynb`: Lower-level Flax alternative

## References

- [Keras Documentation](https://keras.io/)
- [DP-SGD Paper](https://arxiv.org/abs/1607.00133)
- [Private LLM Fine-tuning](https://arxiv.org/abs/2110.06500)

## Related Modules

- `dp_sgd`: Low-level DP-SGD implementation
- `accounting`: Privacy budget tracking
- `noise_addition`: Noise mechanisms
