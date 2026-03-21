# Architecture

## Overview

`jax_privacy` is organized into three layers. At the foundation is the **core
library** (`jax_privacy/`), which provides all privacy primitives: DP accounting,
DP-SGD training utilities, noise addition, matrix-factorization mechanisms,
experimental APIs, and a high-level Keras integration. Built on top of the core
library is the **experiments pipeline** (`experiments/`), a research-grade
training pipeline for image classification that demonstrates how the primitives
compose into end-to-end DP training runs. The outermost layer is **examples and
notebooks** (`examples/`), offering self-contained scripts and Jupyter notebooks
that show how to use the library and the pipeline for common tasks such as MNIST
classification, Keras-based fine-tuning, and Gemma LoRA fine-tuning.

---

## Directory Tree

```
jax_privacy/                            # Repository root
├── jax_privacy/                        # Core library package
│   ├── __init__.py                     # Public API surface
│   ├── accounting/                     # Privacy accounting (ε/δ analysis)
│   │   ├── __init__.py
│   │   ├── accountants.py
│   │   ├── analysis.py
│   │   ├── calibrate.py
│   │   ├── analysis_test.py
│   │   └── calibrate_test.py
│   ├── dp_sgd/                         # DP-SGD training primitives
│   │   ├── __init__.py
│   │   ├── grad_clipping.py
│   │   ├── grad_clipping_utils.py
│   │   ├── gradients.py
│   │   ├── metrics_accumulator.py
│   │   ├── optim.py
│   │   ├── typing.py
│   │   ├── wrapped_privatizer.py
│   │   ├── grad_clipping_test.py
│   │   ├── grad_clipping_sharded_test.py
│   │   ├── gradients_test.py
│   │   └── optim_test.py
│   ├── noise_addition/                 # Gaussian / matrix-factorisation noise
│   │   ├── __init__.py
│   │   ├── additive_privatizers.py
│   │   ├── sharding_utils.py
│   │   ├── additive_privatizers_test.py
│   │   ├── distributed_noise_generation_test.py
│   │   └── sharding_utils_test.py
│   ├── matrix_factorization/           # Matrix-mechanism strategies
│   │   ├── __init__.py
│   │   ├── banded.py
│   │   ├── buffered_toeplitz.py
│   │   ├── checks.py
│   │   ├── dense.py
│   │   ├── optimization.py
│   │   ├── sensitivity.py
│   │   ├── streaming_matrix.py
│   │   ├── test_utils.py
│   │   ├── toeplitz.py
│   │   └── *_test.py (9 test files)
│   ├── experimental/                   # Pre-production / research APIs
│   │   ├── __init__.py
│   │   ├── batch_selection.py
│   │   ├── clipping.py
│   │   ├── compilation_utils.py
│   │   ├── execution_plan.py
│   │   ├── microbatching.py
│   │   └── *_test.py (5 test files)
│   ├── keras/                          # High-level Keras integration
│   │   ├── __init__.py
│   │   └── keras_api.py + keras_api_test.py
│   └── auditing/                       # Privacy auditing
│
├── experiments/                        # Research training pipeline
│   └── image_classification/           # DP image classification experiments
│
├── examples/                           # Standalone usage examples
│   ├── keras_api_example.py
│   ├── jax_api_example.py
│   ├── jax_new_api_example.py
│   ├── distributed_noise_generation.py
│   ├── dp_logistic_regression.py
│   ├── dpmf_strategy_optimization.py
│   ├── dp_sgd_flax_linen_mnist.ipynb
│   ├── dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb
│   └── dp_sgd_keras_gemma3_synthetic_data.ipynb
│
├── docs/                               # Sphinx documentation
│   ├── Makefile / make.bat
│   └── source/
│       ├── conf.py
│       └── requirements.txt
│
├── bibtex/                             # Paper citations
│   ├── de2022unlocking.bib
│   └── berrada2023unlocking.bib
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # CI: Python 3.11, ubuntu-latest, bash test.sh
│
├── setup.py                            # Package installation
├── pyproject.toml                      # Build system
├── requirements.txt                    # Runtime dependencies
├── requirements-dev.txt                # Dev + experiment dependencies
├── conftest.py                         # Global pytest config
├── test.sh                             # Full test runner
├── lint.sh                             # Linting (flake8 + pylint + pytype)
├── .pylintrc                           # Pylint rules
├── .readthedocs.yaml                   # Docs CI config
├── MANIFEST.in                         # Sdist manifest
├── CITATION.cff                        # Citation metadata
├── README.md                           # Project overview
├── CONTRIBUTING.md                     # Contribution guide
├── SECURITY.md                         # Security policy
├── LICENSE                             # Apache 2.0
└── create_pr1_gist.cmd                 # Windows gist helper script
```

---

## Layer Descriptions

### `jax_privacy/` (Core Library)

The installable Python package that provides all privacy primitives. It is the
only layer published to users via `pip install`.

#### `accounting/`

Privacy analysis for differential privacy. Computes (ε, δ) guarantees for a
sequence of randomised mechanisms.

| Module | Purpose |
|---|---|
| `analysis.py` | End-to-end DP analysis, composing mechanism accountants |
| `accountants.py` | Individual mechanism accountants (Gaussian, Laplace, …) |
| `calibrate.py` | Noise-multiplier calibration given a target (ε, δ) budget |

#### `dp_sgd/`

DP-SGD training primitives: per-example gradient clipping, noise injection, and
optimizer wrappers.

| Module | Purpose |
|---|---|
| `grad_clipping.py` | Per-example gradient clipping |
| `grad_clipping_utils.py` | Utility functions for clipping |
| `gradients.py` | Privatized gradient computation |
| `metrics_accumulator.py` | Accumulates training metrics across steps |
| `optim.py` | DP-aware Optax optimizer wrapper |
| `typing.py` | Shared type aliases |
| `wrapped_privatizer.py` | Wraps any privatizer for use in a training loop |

#### `noise_addition/`

Additive noise generation including Gaussian and matrix-factorisation-based
noise for distributed and multi-device settings.

| Module | Purpose |
|---|---|
| `additive_privatizers.py` | Gaussian and factorised-noise privatizers |
| `sharding_utils.py` | JAX sharding helpers for distributed noise generation |

#### `matrix_factorization/`

Matrix mechanism strategies that reduce the noise required for streaming /
continual-release DP.

| Module | Purpose |
|---|---|
| `dense.py` | Dense matrix strategy |
| `banded.py` | Banded matrix strategy |
| `toeplitz.py` | Toeplitz matrix strategy |
| `buffered_toeplitz.py` | Buffered Toeplitz strategy |
| `streaming_matrix.py` | Streaming / online matrix mechanism |
| `sensitivity.py` | Sensitivity computation for matrix strategies |
| `optimization.py` | Strategy parameter optimization |
| `checks.py` | Invariant / shape checks |
| `test_utils.py` | Shared test helpers |

#### `experimental/`

Pre-production APIs under active development. Interfaces may change between
releases without notice.

| Module | Purpose |
|---|---|
| `clipping.py` | Experimental clipping variants |
| `batch_selection.py` | Poisson / custom batch sampling |
| `microbatching.py` | Micro-batch gradient accumulation |
| `execution_plan.py` | XLA execution plan helpers |
| `compilation_utils.py` | JIT / AOT compilation utilities |

#### `keras/`

High-level Keras integration exposing a single `DPKerasModel` wrapper that
makes an existing Keras model differentially private with minimal code changes.

| Module | Purpose |
|---|---|
| `keras_api.py` | `DPKerasModel` public API |
| `keras_api_test.py` | Integration tests |

#### `auditing/`

Privacy auditing utilities for empirically estimating the privacy leakage of a
trained model.

#### `__init__.py`

Re-exports the public API surface so that users can write
`import jax_privacy` and access all top-level symbols.

---

### `experiments/`

A research-grade JAX/Flax training pipeline for DP image classification. It
wires together the core library's DP updater, evaluator, and checkpointing. Not
installed as part of the pip package; intended to be run directly from a local
checkout.

Key sub-directory: `image_classification/` — configuration files, model
definitions, and launch scripts for reproducing published results.

---

### `examples/`

Standalone scripts and Jupyter notebooks demonstrating end-to-end usage of the
library for common tasks.

| File | Description |
|---|---|
| `keras_api_example.py` | Simple Keras API usage with a small model |
| `dp_sgd_flax_linen_mnist.ipynb` | MNIST classification with Flax Linen |
| `dp_sgd_keras_gemma3_lora_finetuning_samsum.ipynb` | LoRA fine-tuning of Gemma 3 with DP |
| `dp_sgd_keras_gemma3_synthetic_data.ipynb` | Gemma 3 fine-tuning on synthetic data |
| `jax_api_example.py` | Raw JAX API usage |
| `jax_new_api_example.py` | New raw JAX API usage |
| `distributed_noise_generation.py` | Multi-device noise generation example |
| `dp_logistic_regression.py` | DP logistic regression from scratch |
| `dpmf_strategy_optimization.py` | Matrix-factorisation strategy optimization |

---

### `docs/`

Sphinx-based documentation published to
[jax-privacy.readthedocs.io](https://jax-privacy.readthedocs.io/).

| Path | Purpose |
|---|---|
| `docs/Makefile` / `docs/make.bat` | Build the HTML docs locally |
| `docs/source/conf.py` | Sphinx configuration |
| `docs/source/requirements.txt` | Docs-only Python dependencies |

---

### `bibtex/`

BibTeX citation files for the papers whose results are reproducible with this
repository.

| File | Paper |
|---|---|
| `de2022unlocking.bib` | "Unlocking High-Accuracy DP Image Classification through Scale" |
| `berrada2023unlocking.bib` | "Unlocking Accuracy and Fairness in DP Image Classification" |

---

## Build & Tooling

| File | Purpose |
|---|---|
| `setup.py` | Pip-installable package definition |
| `pyproject.toml` | Build system declaration (setuptools) |
| `requirements.txt` | Core runtime dependencies |
| `requirements-dev.txt` | Dev/experiment/example dependencies |
| `conftest.py` | Global pytest config (`absl` flags pre-parsed) |
| `test.sh` | Full CI test runner script |
| `lint.sh` | flake8 + pylint + pytype linting script |
| `.pylintrc` | Pylint configuration |
| `.readthedocs.yaml` | Read the Docs build configuration |
| `MANIFEST.in` | Package manifest for sdist |
| `CITATION.cff` | Machine-readable citation metadata |
| `create_pr1_gist.cmd` | Windows helper script for creating consolidation gists |

---

## CI/CD

The repository uses GitHub Actions. The workflow is defined in
`.github/workflows/ci.yml` and has the following properties:

- **Triggers**: push and pull-request events targeting the `main` branch.
- **Matrix**: Python 3.11 on `ubuntu-latest`.
- **Steps**: checks out the repository, installs dependencies, then runs
  `bash test.sh` which discovers and executes every `*_test.py` file via
  pytest.

---

## Test Strategy

- **Co-located tests**: every source module `foo.py` has a sibling
  `foo_test.py` in the same directory. This makes it easy to find the tests for
  any given module.
- **Sharding / distributed tests**: tests that exercise multi-device behaviour
  (e.g. `grad_clipping_sharded_test.py`, `distributed_noise_generation_test.py`)
  are kept in separate files so they can be skipped on single-device CI runners.
- **Property-based tests**: several modules use
  [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing.
- **Orchestration**: `test.sh` drives the full test suite, allowing CI to run
  all tests with a single command while giving developers the flexibility to run
  individual test files locally.

---

## Dependency Layers

```
examples/ ──►  experiments/ ──► jax_privacy/ (core)
                                     │
                              ┌──────┼──────────────┐
                              ▼      ▼               ▼
                         accounting  dp_sgd    noise_addition
                              │      │               │
                              └──────┴───────────────┘
                                      │
                              matrix_factorization
                              experimental / keras
```

---

## Module Import Map

| Import | Exposes |
|---|---|
| `import jax_privacy` | Top-level public API |
| `from jax_privacy import accounting` | DP accounting |
| `from jax_privacy import dp_sgd` | DP-SGD primitives |
| `from jax_privacy import noise_addition` | Noise privatizers |
| `from jax_privacy import matrix_factorization` | Matrix mechanisms |
| `from jax_privacy import experimental` | Experimental APIs |
| `from jax_privacy import keras` | Keras integration |

---

## Contributing Quick-Reference

- **Contribution guide**: see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full
  workflow (branching, code review, CLA).
- **Code style**: the project follows the
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- **Docstrings**: all public functions must include a docstring; follow the
  Google docstring format used throughout the codebase.

---

Copyright 2025 DeepMind Technologies Limited. Licensed under the
[Apache License, Version 2.0](LICENSE).
