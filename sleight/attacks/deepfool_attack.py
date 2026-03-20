"""DeepFool attack implementation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def deepfool_attack(
    model: tf.keras.Model,
    images: tf.Tensor | np.ndarray,
    labels: tf.Tensor | np.ndarray,
    epsilon: float = 1.0,
    num_classes: int | None = None,
    max_iter: int = 50,
    overshoot: float = 0.02,
) -> tf.Tensor:
    """Perform the DeepFool attack.

    Computes minimal perturbations that cross the nearest decision boundary.
    Each image is perturbed independently by iteratively finding the closest
    class boundary and stepping toward it.

    The ``epsilon`` parameter acts as a hard L2 norm cap on the final
    perturbation. The ``labels`` parameter is accepted for API consistency
    but is not used (DeepFool is an untargeted attack that finds the
    nearest boundary regardless of the true label).

    Reference:
        Moosavi-Dezfooli, S.-M., Fawzi, A., & Frossard, P. (2016).
        "DeepFool: a simple and accurate method to fool deep neural networks."
        CVPR 2016. https://arxiv.org/abs/1511.04599

    Args:
        model: A compiled tf.keras.Model to attack.
        images: Input images with shape (batch, ...). Values should be in [0, 1].
        labels: Ground-truth labels (accepted for API consistency, not used).
        epsilon: Maximum L2 perturbation norm.
        num_classes: Number of classes to consider. If ``None``, inferred
            from the model output shape.
        max_iter: Maximum number of iterations per image.
        overshoot: Small factor multiplied to the perturbation to ensure
            the boundary is crossed.

    Returns:
        Adversarial images as a ``tf.Tensor``, clipped to [0, 1].
    """
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)

    if num_classes is None:
        num_classes = model.output_shape[-1]

    adv_images = []
    for i in range(images.shape[0]):
        adv = _deepfool_single(
            model, images[i], num_classes, max_iter, overshoot, epsilon
        )
        adv_images.append(adv)

    return tf.stack(adv_images)


def _deepfool_single(
    model: tf.keras.Model,
    image: tf.Tensor,
    num_classes: int,
    max_iter: int,
    overshoot: float,
    epsilon: float,
) -> tf.Tensor:
    """Run DeepFool on a single image."""
    image = tf.identity(image)
    pert_image = tf.identity(image)
    total_pert = tf.zeros_like(image)

    # Get initial prediction
    with tf.GradientTape() as tape:
        tape.watch(pert_image)
        logits = model(tf.expand_dims(pert_image, 0))
    orig_label = tf.argmax(logits[0]).numpy()

    for _ in range(max_iter):
        # Compute gradients for all classes
        pert_image_var = tf.Variable(pert_image)
        with tf.GradientTape(persistent=True) as tape:
            logits = model(tf.expand_dims(pert_image_var, 0))[0]

        current_label = tf.argmax(logits).numpy()
        if current_label != orig_label:
            break

        # Find the closest boundary
        grad_orig = tape.gradient(logits[orig_label], pert_image_var)
        min_dist = float("inf")
        best_pert = tf.zeros_like(image)

        for k in range(num_classes):
            if k == orig_label:
                continue
            grad_k = tape.gradient(logits[k], pert_image_var)
            if grad_k is None:
                continue

            w_k = grad_k - grad_orig
            f_k = float(logits[k] - logits[orig_label])

            w_k_norm = tf.norm(tf.reshape(w_k, [-1]))
            if w_k_norm < 1e-12:
                continue

            dist = abs(f_k) / float(w_k_norm)
            if dist < min_dist:
                min_dist = dist
                best_pert = (abs(f_k) / (float(w_k_norm) ** 2 + 1e-12)) * w_k

        del tape

        total_pert = total_pert + (1 + overshoot) * best_pert
        pert_image = image + total_pert

    # Project to L2 epsilon ball
    pert_flat = tf.reshape(total_pert, [-1])
    l2_norm = tf.norm(pert_flat)
    if l2_norm > epsilon:
        total_pert = total_pert * (epsilon / (l2_norm + 1e-12))

    return tf.clip_by_value(image + total_pert, 0, 1)
