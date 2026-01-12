# Copilot Instructions for JAX-Privacy

## Repository Overview

JAX-Privacy is a library for privacy-preserving machine learning in JAX. It provides:
- Production-focused APIs for differentially-private (DP) training of ML models in JAX and Keras
- Core components for implementing differentially private machine learning algorithms
- A JAX-based ML DP pipeline for experimenting with image classification models

## Development Workflow

### Linting

Run the linting script to check code style:
```bash
bash lint.sh
```

This runs:
- `flake8` for syntax errors and code quality
- `pylint` with Google style guide configuration (`.pylintrc`)
- `pytype` for type checking

### Testing

Run tests with:
```bash
bash test.sh
```

Tests use `pytest` with `pytest-xdist` for parallel execution.

### Installation

For development, use local installation:
```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Coding Style

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Key Conventions

1. **Public Facing Functions**: Must have full docstrings, type annotations, and doctests
2. **Internal Functions (used across files)**: Should have descriptive names, type annotations, and a one-line docstring
3. **File-local Private Functions**: Prefix with `_`, include 1-line docstrings; type annotations are optional
4. **Nested Functions**: Keep simple with minimal boilerplate (no docstrings or type annotations required)

### Formatting

- Maximum line length: 80 characters
- Indentation: 2 spaces (as configured in `.pylintrc`, consistent with Google open-source projects like TensorFlow)
- Follow immutability and functional purity principles consistent with the JAX ecosystem

## Dependencies

- Python 3.11+
- Core dependencies are in `requirements.txt`
- Development dependencies are in `requirements-dev.txt`

## Contributing

- Small bug fix patches are welcome
- Major changes are not planned to be accepted at this time
- All submissions require review via GitHub pull requests
- A Contributor License Agreement (CLA) is required

## Project Structure

- `jax_privacy/` - Core library code
- `examples/` - Example notebooks and scripts
- `experiments/` - Research experiment code
- `docs/` - Documentation (built with Sphinx)
