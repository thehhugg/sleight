"""Adversarial training defense implementation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import tensorflow as tf


def adversarial_train(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    attack_fn: Callable[[tf.keras.Model, np.ndarray, np.ndarray, float], np.ndarray],
    epsilon: float = 0.1,
    epochs: int = 3,
    batch_size: int = 64,
) -> tf.keras.Model:
    """Perform adversarial training on a Keras model.

    Trains the model on a mix of clean and adversarial examples generated
    by the provided attack function.

    Args:
        model: A compiled tf.keras.Model to train.
        x_train: Training images as a NumPy array.
        y_train: Training labels as integer class indices.
        attack_fn: A callable with signature
            ``attack_fn(model, images, labels, epsilon) -> adversarial_images``.
        epsilon: Perturbation strength passed to ``attack_fn``.
        epochs: Number of training epochs.
        batch_size: Batch size for training.

    Returns:
        The trained model (same object, modified in place).
    """
    num_batches = int(np.ceil(len(x_train) / batch_size))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=model.output_shape[-1])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        idx = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[idx]
        y_train_shuffled = y_train_cat[idx]
        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(x_train))
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Generate adversarial examples
            x_adv = attack_fn(model, x_batch, y_batch, epsilon)
            x_combined = np.concatenate([x_batch, x_adv], axis=0)
            y_combined = np.concatenate([y_batch, y_batch], axis=0)

            # Train on both clean and adversarial examples
            model.train_on_batch(x_combined, y_combined)
    return model 