"""Fast Gradient Sign Method (FGSM) attack implementation."""

from __future__ import annotations

import tensorflow as tf
import numpy as np


def fgsm_attack(
    model: tf.keras.Model,
    images: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    epsilon: float,
) -> tf.Tensor:
    """Perform the Fast Gradient Sign Method (FGSM) attack.

    Generates adversarial examples by adding a single-step perturbation
    in the direction of the gradient of the loss with respect to the input.

    Reference:
        Goodfellow et al., "Explaining and Harnessing Adversarial Examples,"
        ICLR 2015. https://arxiv.org/abs/1412.6572

    Args:
        model: A compiled tf.keras.Model to attack.
        images: Input images with shape (batch, ...). Values should be in [0, 1].
        labels: Ground-truth labels. Supports both integer labels (shape
            ``(batch,)`` or ``(batch, 1)``) and one-hot encoded labels
            (shape ``(batch, num_classes)``).
        epsilon: Maximum perturbation magnitude (L-infinity bound).

    Returns:
        Adversarial images as a ``tf.Tensor``, clipped to [0, 1], with the
        same shape as the input.
    """
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

    # Auto-detect label format and choose appropriate loss
    if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[-1] == 1):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        if labels.ndim == 2:
            labels = tf.squeeze(labels, axis=-1)
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = loss_fn(labels, prediction)

    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    return tf.clip_by_value(adversarial_images, 0, 1)
