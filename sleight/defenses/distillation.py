"""Defensive distillation defense implementation.

Reference:
    Papernot, N., McDaniel, P., Wu, X., Jha, S., & Swami, A. (2016).
    "Distillation as a Defense to Adversarial Perturbations against
    Deep Neural Networks." IEEE S&P 2016.
    https://arxiv.org/abs/1511.04508
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def defensive_distillation(
    model_fn: callable,
    x_train: np.ndarray,
    y_train: np.ndarray,
    temperature: float = 20.0,
    epochs: int = 10,
    batch_size: int = 64,
) -> tf.keras.Model:
    """Train a model using defensive distillation.

    Defensive distillation trains a teacher model at high temperature,
    then uses the teacher's soft predictions to train a student model
    (same architecture) at the same temperature. The student learns
    smoother decision boundaries that are harder to attack with
    gradient-based methods.

    Args:
        model_fn: A callable that returns a new, uncompiled
            ``tf.keras.Model`` each time it is called. Both teacher
            and student will be created from this function.
        x_train: Training images as a NumPy array.
        y_train: Training labels as integer class indices.
        temperature: Distillation temperature. Higher values produce
            softer probability distributions.
        epochs: Number of training epochs for each model (teacher
            and student).
        batch_size: Training batch size.

    Returns:
        The trained student model.
    """
    # Infer num_classes from the model architecture, not the data
    sample_model = model_fn()
    num_classes = sample_model.output_shape[-1]
    del sample_model
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)

    # --- Train teacher at high temperature ---
    teacher = model_fn()
    # Replace the final softmax with temperature-scaled softmax
    teacher_logits = _replace_final_activation(teacher, temperature)
    teacher_logits.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    teacher_logits.fit(
        x_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=0
    )

    # Generate soft labels from teacher
    soft_labels = teacher_logits.predict(x_train, verbose=0)

    # --- Train student at high temperature on soft labels ---
    student = model_fn()
    student_logits = _replace_final_activation(student, temperature)
    student_logits.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    student_logits.fit(
        x_train, soft_labels, epochs=epochs, batch_size=batch_size, verbose=0
    )

    # Return the student with temperature=1 (normal softmax) for inference
    final_model = model_fn()
    # Copy weights from the temperature-scaled model
    for final_layer, student_layer in zip(
        final_model.layers, student_logits.layers
    ):
        try:
            final_layer.set_weights(student_layer.get_weights())
        except ValueError:
            pass  # Skip layers with incompatible shapes (shouldn't happen)

    final_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return final_model


def _replace_final_activation(
    model: tf.keras.Model, temperature: float
) -> tf.keras.Model:
    """Replace the final softmax layer with temperature-scaled softmax.

    Creates a new model that applies ``softmax(logits / temperature)``
    instead of the standard ``softmax(logits)``.

    Args:
        model: A Keras Sequential model whose last layer uses softmax.
        temperature: The temperature scaling factor.

    Returns:
        A new model with temperature-scaled final activation.
    """
    # Build a new model with the same layers but temperature-scaled output
    layers = model.layers[:-1]
    last_layer = model.layers[-1]

    new_model = tf.keras.Sequential()
    for layer in layers:
        new_model.add(layer)

    # Add the final dense layer without activation
    new_model.add(
        tf.keras.layers.Dense(
            last_layer.units,
            activation=None,
            name="logits",
        )
    )
    # Add temperature-scaled softmax
    new_model.add(
        tf.keras.layers.Lambda(
            lambda x: tf.nn.softmax(x / temperature),
            name="temp_softmax",
        )
    )

    return new_model
