"""Carlini & Wagner (C&W) L2 attack implementation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def cw_attack(
    model: tf.keras.Model,
    images: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    epsilon: float = 1.0,
    c: float = 1.0,
    learning_rate: float = 0.01,
    num_iter: int = 100,
    targeted: bool = False,
) -> tf.Tensor:
    """Perform the Carlini & Wagner L2 attack.

    Finds adversarial examples by solving an optimization problem that
    minimizes the L2 perturbation while causing misclassification. The
    ``epsilon`` parameter acts as a hard L2 norm cap on the final
    perturbation.

    Reference:
        Carlini, N. & Wagner, D. (2017). "Towards Evaluating the Robustness
        of Neural Networks." IEEE S&P 2017. https://arxiv.org/abs/1608.04644

    Args:
        model: A compiled tf.keras.Model to attack.
        images: Input images with shape (batch, ...). Values should be in [0, 1].
        labels: Ground-truth labels. Supports both integer labels (shape
            ``(batch,)`` or ``(batch, 1)``) and one-hot encoded labels
            (shape ``(batch, num_classes)``).
        epsilon: Maximum L2 perturbation norm. The final perturbation is
            projected to have L2 norm at most ``epsilon``.
        c: Confidence parameter controlling the trade-off between
            perturbation size and misclassification confidence.
        learning_rate: Step size for the Adam optimizer.
        num_iter: Number of optimization iterations.
        targeted: If ``True``, the attack tries to classify as ``labels``
            instead of away from ``labels``.

    Returns:
        Adversarial images as a ``tf.Tensor``, clipped to [0, 1].
    """
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

    # Convert one-hot labels to integer indices
    if labels.ndim >= 2 and labels.shape[-1] != 1:
        labels_int = tf.argmax(labels, axis=-1)
    else:
        labels_int = tf.cast(tf.reshape(labels, [-1]), tf.int64)

    num_classes = model.output_shape[-1]
    batch_size = tf.shape(images)[0]

    # Work in tanh space for unconstrained optimization
    # x = (tanh(w) + 1) / 2, so w = arctanh(2x - 1)
    images_tanh = tf.atanh(tf.clip_by_value(images * 2 - 1, -0.999999, 0.999999))

    # Perturbation variable in tanh space
    w = tf.Variable(tf.zeros_like(images_tanh))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            # Map back to image space
            adv_images = (tf.tanh(images_tanh + w) + 1) / 2

            # L2 distance
            l2_dist = tf.reduce_sum(
                tf.square(adv_images - images), axis=list(range(1, len(images.shape)))
            )

            # Model output (logits-like)
            logits = model(adv_images)

            # f(x) = max(Z(x)_t - max(Z(x)_i, i != t), 0) for untargeted
            one_hot = tf.one_hot(labels_int, num_classes)
            real = tf.reduce_sum(logits * one_hot, axis=1)
            other = tf.reduce_max(logits * (1 - one_hot) - one_hot * 1e4, axis=1)

            if targeted:
                # Minimize real - other (make target class dominant)
                f_val = tf.maximum(other - real, 0.0)
            else:
                # Minimize other - real (make any other class dominant)
                f_val = tf.maximum(real - other, 0.0)

            loss = tf.reduce_mean(l2_dist + c * f_val)

        gradients = tape.gradient(loss, [w])
        optimizer.apply_gradients(zip(gradients, [w]))

    # Final adversarial images
    adv_images = (tf.tanh(images_tanh + w) + 1) / 2

    # Project perturbation to L2 epsilon ball
    perturbation = adv_images - images
    flat_shape = [tf.shape(perturbation)[0], -1]
    perturbation_flat = tf.reshape(perturbation, flat_shape)
    l2_norms = tf.norm(perturbation_flat, axis=1, keepdims=True)
    scale = tf.minimum(1.0, epsilon / (l2_norms + 1e-12))
    perturbation_flat = perturbation_flat * scale
    perturbation = tf.reshape(perturbation_flat, tf.shape(images))

    adv_images = tf.clip_by_value(images + perturbation, 0, 1)
    return adv_images
