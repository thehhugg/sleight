"""Fashion-MNIST dataset loader."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def load_fashion_mnist_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess the Fashion-MNIST dataset.

    Images are normalized to [0, 1], cast to ``float32``, and given a
    channel dimension (28, 28, 1). Labels are returned as integer indices.

    Returns:
        ``((x_train, y_train), (x_test, y_test))``.
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis].astype("float32")
    x_test = x_test[..., np.newaxis].astype("float32")
    return (x_train, y_train), (x_test, y_test) 