# JAX-Privacy Copilot Instructions

## Project Overview

JAX-Privacy is a library for privacy-preserving machine learning in JAX. It provides:

- Production-focused APIs for differentially-private (DP) training of ML models in JAX and Keras
- Core components for implementing differentially private machine learning algorithms in JAX
- A JAX-based machine learning DP pipeline for experimenting with image classification models

The project supports multiple API levels: Keras (high-level), Flax Linen (mid-level), and Raw JAX (low-level).

## Tech Stack

- **Language**: Python 3.11+
- **Core Dependencies**: JAX, Keras, Flax, Optax
- **Testing**: pytest with parallel execution
- **Type Checking**: pytype
- **Linting**: flake8, pylint
- **Documentation**: Sphinx (ReadTheDocs)

## Installation

The repository uses `pip` for package management. For local development:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Building and Testing

### Running Tests

```bash
bash test.sh
```

This script:
- Creates a virtual environment
- Installs dependencies
- Runs pytest with parallel execution (using all CPU cores)
- Isolates tests that use `chex.set_n_cpu_device()`
- Builds documentation with Sphinx

**Important**: Some tests need to be run in isolation (specified in `test.sh`). Don't modify the test isolation logic without understanding the reason.

### Linting

```bash
bash lint.sh
```

This script runs:
- `flake8` with specific error codes (E9, F63, F7, F82, E225, E251)
- `pylint` with the configuration in `.pylintrc`
- `pytype` for type checking

## Code Style Guidelines

Code follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Key Principles

- **Immutability and functional purity**: Core principles throughout the library, aligned with the JAX ecosystem
- **Type annotations**: Required for all public-facing functions and internal functions used across files
- **Docstrings**: Must follow these conventions based on visibility level

#### Public Facing Functions

Functions exposed to users via `__init__.py`:
- **Always** include full docstrings
- **Always** include type annotations
- **Always** include example usages as [doctests](https://docs.python.org/3/library/doctest.html)
- Doctests serve as documentation and validation of API simplicity

#### Internal Functions (used across files)

- Descriptive names
- Type annotations required
- One-line docstring minimum
- Full docstring encouraged for complex logic, non-obvious side effects, or multi-step algorithms

#### File-local Private Functions

- Prefix with underscore (`_`)
- One-line docstrings
- Type annotations optional and context-dependent
- No example usages needed (found in corresponding `_test.py` files)

#### Nested Functions

- Keep simple with minimal boilerplate
- No docstrings or type annotations required
- Use inline comments sparingly

### Test Guidelines

- Test files end with `_test.py`
- Tests for private functions disable `protected-access` warnings (`W0212`)
- Tests disable `missing-module-docstring` convention (`C0114`)
- Some tests require specific isolation (see `test.sh`)

## Common Commands

```bash
# Install for development
pip install -e .

# Run all tests
bash test.sh

# Run linters
bash lint.sh

# Install dev dependencies
pip install -r requirements-dev.txt

# Build documentation
cd docs
pip install -r source/requirements.txt
make html
```

## Important Notes

- **JAX ecosystem alignment**: Keep APIs, names, and design patterns consistent with the broader JAX ecosystem
- **No force push**: The repository does not allow force pushing (no `git rebase` or `git reset --hard`)
- **CI/CD**: GitHub Actions runs tests on pushes and pull requests to `main` branch (see `.github/workflows/ci.yml`)
- **Test isolation**: Some tests use `chex.set_n_cpu_device()` and must run in isolation. These are handled in `test.sh`.
- **Matrix factorization tests**: These are expensive and require `HYPOTHESIS_PROFILE=dpftrl_default`

## File Structure

```
jax_privacy/
├── accounting/          # Privacy accounting
├── auditing/           # Privacy auditing tools
├── dp_sgd/             # DP-SGD implementation
├── experimental/       # Experimental features
├── keras/              # Keras API
├── matrix_factorization/ # Matrix factorization algorithms
└── noise_addition/     # Noise addition utilities
```

## Contributing

We welcome small patches for bug fixes. Major changes are not accepted at this time.

- All contributions require a Contributor License Agreement (CLA)
- All submissions require code review via GitHub pull requests
- Keep changes focused and well-scoped

## When Making Changes

1. **Run linters before committing**: Use `bash lint.sh`
2. **Run tests after changes**: Use `bash test.sh` or targeted pytest commands
3. **Follow the style guide**: Google Python Style Guide with JAX ecosystem alignment
4. **Add docstrings**: Follow the guidelines based on function visibility
5. **Add doctests**: For public-facing functions
6. **Maintain functional purity**: Avoid side effects where possible
7. **Don't break test isolation**: Respect the test groupings in `test.sh`
