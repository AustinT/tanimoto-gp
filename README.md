# Tanimoto GP

A small library for Tanimoto kernel Gaussian processes over molecules.

For a minimal running example, see `examples/logp_regression.py`.

## Installation

```bash
pip install tanimoto-gp
```

## Development

```bash
# Install all deps (including dev group)
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest
```

## List of possible improvements

- Cache the cholesky factorization of kernel matrix to avoid repeatedly computing this in BO
