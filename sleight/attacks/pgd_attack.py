"""Projected Gradient Descent (PGD) attack implementation."""

from __future__ import annotations

import tensorflow as tf
import numpy as np


def pgd_attack(
    model: tf.keras.Model,
    images: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    num_iter: int = 40,
) -> tf.Tensor:
    """Perform the Projected Gradient Descent (PGD) attack.

    An iterative variant of FGSM that applies multiple small perturbation
    steps, projecting back onto the epsilon-ball after each step.

    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial
        Attacks," ICLR 2018. https://arxiv.org/abs/1706.06083

    Args:
        model: A compiled tf.keras.Model to attack.
        images: Input images with shape (batch, ...). Values should be in [0, 1].
        labels: Ground-truth labels. Supports both integer labels (shape
            ``(batch,)`` or ``(batch, 1)``) and one-hot encoded labels
            (shape ``(batch, num_classes)``).
        epsilon: Maximum perturbation magnitude (L-infinity bound).
        alpha: Step size per iteration.
        num_iter: Number of PGD iterations.

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

    adv_images = tf.identity(images)
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            prediction = model(adv_images)
            loss = loss_fn(labels, prediction)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)
    return adv_images
