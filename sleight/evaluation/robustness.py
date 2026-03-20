"""Robustness evaluation utilities for adversarial attacks."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import tensorflow as tf


def evaluate_robustness(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    attack_fn: Callable[..., tf.Tensor],
    epsilon: float,
    **attack_kwargs,
) -> dict[str, float]:
    """Evaluate model robustness against a single attack at a fixed epsilon.

    Computes clean accuracy, adversarial accuracy, and attack success rate.

    Args:
        model: A compiled tf.keras.Model.
        images: Test images as a NumPy array, values in [0, 1].
        labels: Ground-truth labels as integer class indices (1-D).
        attack_fn: Attack callable with signature
            ``attack_fn(model, images, labels, epsilon, **kwargs) -> adv_images``.
        epsilon: Perturbation budget passed to the attack.
        **attack_kwargs: Additional keyword arguments forwarded to ``attack_fn``.

    Returns:
        A dict with keys ``clean_accuracy``, ``adversarial_accuracy``,
        ``attack_success_rate``, and ``mean_perturbation``.
    """
    # Clean accuracy
    clean_preds = np.argmax(model.predict(images, verbose=0), axis=1)
    clean_correct = clean_preds == labels
    clean_accuracy = float(np.mean(clean_correct))

    # Generate adversarial examples
    adv_images = attack_fn(model, images, labels, epsilon, **attack_kwargs)
    adv_images_np = np.array(adv_images)

    # Adversarial accuracy
    adv_preds = np.argmax(model.predict(adv_images_np, verbose=0), axis=1)
    adv_correct = adv_preds == labels
    adversarial_accuracy = float(np.mean(adv_correct))

    # Attack success rate (fraction of correctly-classified samples that flip)
    if np.sum(clean_correct) > 0:
        attack_success_rate = float(
            np.sum(clean_correct & ~adv_correct) / np.sum(clean_correct)
        )
    else:
        attack_success_rate = 0.0

    # Mean L-infinity perturbation
    mean_perturbation = float(np.mean(np.max(np.abs(adv_images_np - images), axis=(1, 2, 3))))

    return {
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_perturbation": mean_perturbation,
    }


def epsilon_sweep(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    attack_fn: Callable[..., tf.Tensor],
    epsilons: Sequence[float],
    **attack_kwargs,
) -> list[dict[str, float]]:
    """Evaluate robustness across a range of epsilon values.

    Runs ``evaluate_robustness`` for each epsilon and returns a list of
    result dicts, useful for plotting accuracy-vs-epsilon curves.

    Args:
        model: A compiled tf.keras.Model.
        images: Test images as a NumPy array, values in [0, 1].
        labels: Ground-truth labels as integer class indices (1-D).
        attack_fn: Attack callable (see ``evaluate_robustness``).
        epsilons: Sequence of epsilon values to evaluate.
        **attack_kwargs: Additional keyword arguments forwarded to ``attack_fn``.

    Returns:
        A list of dicts, one per epsilon. Each dict has the same keys as
        ``evaluate_robustness`` output, plus an ``epsilon`` key.
    """
    results = []
    for eps in epsilons:
        result = evaluate_robustness(
            model, images, labels, attack_fn, eps, **attack_kwargs
        )
        result["epsilon"] = eps
        results.append(result)
    return results
