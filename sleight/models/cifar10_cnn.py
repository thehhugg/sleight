"""CIFAR-10 CNN model architecture."""

from __future__ import annotations

import tensorflow as tf


def get_cifar10_cnn_model() -> tf.keras.Model:
    """Create and compile a CNN for CIFAR-10 image classification.

    Architecture: Conv2D(32) -> Conv2D(64) -> MaxPool -> Dropout(0.25) ->
    Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10).
    Compiled with Adam optimizer and categorical crossentropy loss.

    Returns:
        A compiled ``tf.keras.Model`` ready for training on 32x32x3 images.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"]
    )
    return model
