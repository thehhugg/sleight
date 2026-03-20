"""CIFAR-10 dataset loader."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def load_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess the CIFAR-10 image dataset.

    Images are normalized to [0, 1] and cast to ``float32``.
    Labels are flattened to 1-D integer arrays.

    Returns:
        ``((x_train, y_train), (x_test, y_test))``.
    """
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test) 