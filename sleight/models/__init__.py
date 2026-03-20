"""Pre-built model architectures for adversarial ML experiments."""

from .mnist_cnn import get_mnist_cnn_model
from .cifar10_cnn import get_cifar10_cnn_model

__all__ = ["get_mnist_cnn_model", "get_cifar10_cnn_model"]
