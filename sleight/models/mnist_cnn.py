"""MNIST CNN model architecture."""

from __future__ import annotations

import tensorflow as tf


def get_mnist_cnn_model() -> tf.keras.Model:
    """Create and compile a simple CNN for MNIST digit classification.

    Architecture: Conv2D(32) -> MaxPool -> Flatten -> Dense(128) -> Dense(10).
    Compiled with Adam optimizer and categorical crossentropy loss.

    Returns:
        A compiled ``tf.keras.Model`` ready for training on 28x28x1 images.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"]
    )
    return model
