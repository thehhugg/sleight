# Sleight

**A hands-on Python toolkit for exploring adversarial attacks and defenses on neural networks.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Sleight lets you attack neural networks, defend them, and measure the results — all in a few lines of Python. It is designed for students, researchers, and anyone curious about adversarial machine learning.

---

## Quickstart

```bash
git clone https://github.com/thehhugg/sleight.git
cd sleight
pip install -e .
```

```python
from sleight.models import get_mnist_cnn_model
from sleight.data import get_dataset
from sleight.attacks import fgsm_attack

# Load data and train a model
(x_train, y_train), (x_test, y_test) = get_dataset("mnist", one_hot=True)
model = get_mnist_cnn_model()
model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)

# Attack it
adv_images = fgsm_attack(model, x_test[:10], y_test[:10], epsilon=0.1)
```

## Supported Attacks

| Attack | Module | Description | Reference |
|--------|--------|-------------|-----------|
| FGSM | `sleight.attacks.fgsm_attack` | Single-step gradient sign perturbation | [Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572) |
| PGD | `sleight.attacks.pgd_attack` | Iterative projected gradient descent | [Madry et al., 2018](https://arxiv.org/abs/1706.06083) |

## Supported Defenses

| Defense | Module | Description |
|---------|--------|-------------|
| Adversarial Training | `sleight.defenses.adversarial_train` | Train on a mix of clean and adversarial examples |

## Supported Datasets

| Dataset | Loader | Shape |
|---------|--------|-------|
| MNIST | `sleight.data.load_mnist_data` | 28x28x1, 10 classes |
| Fashion-MNIST | `sleight.data.load_fashion_mnist_data` | 28x28x1, 10 classes |
| CIFAR-10 | `sleight.data.load_cifar10_data` | 32x32x3, 10 classes |

## Project Structure

```
sleight/
├── sleight/
│   ├── attacks/          # Adversarial attack implementations
│   ├── defenses/         # Defense mechanism implementations
│   ├── models/           # Pre-built model architectures
│   └── data/             # Dataset loaders and registry
├── tests/                # Unit and integration tests
├── notebooks/            # Jupyter notebook demos
├── docs/                 # Attack and defense explanations
├── pyproject.toml        # Package configuration
└── README.md
```

## Usage Examples

### FGSM Attack

```python
from sleight.attacks import fgsm_attack

# Works with both integer and one-hot labels
adv_images = fgsm_attack(model, images, labels, epsilon=0.1)
```

### PGD Attack

```python
from sleight.attacks import pgd_attack

adv_images = pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=40)
```

### Adversarial Training

```python
from sleight.defenses import adversarial_train
from sleight.attacks import pgd_attack

attack_fn = lambda model, images, labels, epsilon: pgd_attack(
    model, images, labels, epsilon=epsilon, alpha=0.01, num_iter=10
)
robust_model = adversarial_train(model, x_train, y_train, attack_fn, epsilon=0.1, epochs=3)
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Notebooks

Example notebooks are available in the `notebooks/` directory:

- **FGSM Attack Demo** — FGSM on MNIST
- **PGD Adversarial Training** — adversarial training with PGD on MNIST
- **CIFAR-10 FGSM Attack** — FGSM on CIFAR-10
- **CIFAR-10 PGD Adversarial Training** — adversarial training on CIFAR-10

## Documentation

See the [docs/](docs/) folder for explanations of each attack and defense, including the math and original paper references.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new attacks, defenses, and tests.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
