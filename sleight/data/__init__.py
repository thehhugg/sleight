"""Dataset loaders and registry for common ML benchmarks."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .mnist import load_mnist_data
from .fashion_mnist import load_fashion_mnist_data
from .cifar10 import load_cifar10_data

__all__ = [
    "load_mnist_data",
    "load_fashion_mnist_data",
    "load_cifar10_data",
    "get_dataset",
    "data_registry",
]

DatasetTuple = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

data_registry: dict[str, callable] = {
    "mnist": load_mnist_data,
    "fashion_mnist": load_fashion_mnist_data,
    "cifar10": load_cifar10_data,
}


def get_dataset(name: str, one_hot: bool = False) -> DatasetTuple:
    """Load a dataset by name from the registry.

    Args:
        name: Dataset identifier. One of ``'mnist'``, ``'fashion_mnist'``,
            or ``'cifar10'``.
        one_hot: If ``True``, convert integer labels to one-hot vectors.

    Returns:
        A tuple ``((x_train, y_train), (x_test, y_test))`` of NumPy arrays.
        Images are normalized to [0, 1] and cast to ``float32``.

    Raises:
        ValueError: If ``name`` is not in the registry.
    """
    if name not in data_registry:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(data_registry.keys())}"
        )
    (x_train, y_train), (x_test, y_test) = data_registry[name]()
    if one_hot:
        import tensorflow as tf

        num_classes = len(set(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test) 