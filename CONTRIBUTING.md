# Contributing to Sleight

Contributions are welcome. This document explains how to set up a development environment and add new attacks or defenses.

## Development Setup

```bash
git clone https://github.com/thehhugg/sleight.git
cd sleight
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

Sleight uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Before submitting a PR, run:

```bash
ruff check sleight/ tests/
ruff format sleight/ tests/
```

All public functions must have Google-style docstrings and type hints.

## Adding a New Attack

1. Create a new file in `sleight/attacks/`, e.g. `sleight/attacks/my_attack.py`.
2. Implement a function with this signature pattern:

```python
def my_attack(
    model: tf.keras.Model,
    images: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    epsilon: float,
    **kwargs,
) -> tf.Tensor:
    """One-line description.

    Longer explanation and paper reference.

    Args:
        model: A compiled tf.keras.Model to attack.
        images: Input images, values in [0, 1].
        labels: Ground-truth labels (integer or one-hot).
        epsilon: Maximum perturbation magnitude.

    Returns:
        Adversarial images clipped to [0, 1].
    """
```

3. Convert inputs to tensors at the top of the function using `tf.convert_to_tensor()`.
4. Support both integer and one-hot label formats (see `fgsm_attack.py` for the pattern).
5. Register the function in `sleight/attacks/__init__.py`.
6. Add tests in `tests/test_attacks.py` covering at minimum: output shape, value range [0, 1], and that the attack changes the input.

## Adding a New Defense

1. Create a new file in `sleight/defenses/`, e.g. `sleight/defenses/my_defense.py`.
2. Follow the existing patterns in `adversarial_training.py` for function signatures and docstrings.
3. Register the function in `sleight/defenses/__init__.py`.
4. Add tests in `tests/test_defenses.py`.

## Pull Request Process

1. Branch from `main`.
2. Write your code and tests.
3. Run `pytest tests/ -v` and `ruff check sleight/ tests/` to verify everything passes.
4. Submit a PR against `main` with a clear description of what you changed and why.
