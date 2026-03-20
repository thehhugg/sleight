"""Tests for the evaluation module."""

import numpy as np
import tensorflow as tf
import pytest

from sleight.evaluation import (
    evaluate_robustness,
    epsilon_sweep,
    plot_accuracy_vs_epsilon,
    plot_attack_comparison,
)


def _make_simple_model():
    """Create a tiny model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def _dummy_attack(model, images, labels, epsilon):
    """A trivial attack that adds uniform noise."""
    noise = np.random.uniform(-epsilon, epsilon, size=images.shape).astype("float32")
    return np.clip(images + noise, 0, 1)


def test_evaluate_robustness():
    model = _make_simple_model()
    images = np.random.rand(20, 4, 4, 1).astype("float32")
    labels = np.random.randint(0, 10, size=20)

    result = evaluate_robustness(model, images, labels, _dummy_attack, epsilon=0.1)

    assert "clean_accuracy" in result
    assert "adversarial_accuracy" in result
    assert "attack_success_rate" in result
    assert "mean_perturbation" in result
    assert 0.0 <= result["clean_accuracy"] <= 1.0
    assert 0.0 <= result["adversarial_accuracy"] <= 1.0
    assert 0.0 <= result["attack_success_rate"] <= 1.0
    assert result["mean_perturbation"] >= 0.0


def test_epsilon_sweep():
    model = _make_simple_model()
    images = np.random.rand(20, 4, 4, 1).astype("float32")
    labels = np.random.randint(0, 10, size=20)

    epsilons = [0.0, 0.1, 0.3]
    results = epsilon_sweep(model, images, labels, _dummy_attack, epsilons)

    assert len(results) == 3
    for r, eps in zip(results, epsilons):
        assert r["epsilon"] == eps
        assert "clean_accuracy" in r


def test_plot_accuracy_vs_epsilon(tmp_path):
    results = [
        {"epsilon": 0.0, "clean_accuracy": 0.95, "adversarial_accuracy": 0.95},
        {"epsilon": 0.1, "clean_accuracy": 0.95, "adversarial_accuracy": 0.70},
        {"epsilon": 0.3, "clean_accuracy": 0.95, "adversarial_accuracy": 0.30},
    ]
    save_path = str(tmp_path / "plot.png")
    fig = plot_accuracy_vs_epsilon(results, save_path=save_path)
    assert fig is not None
    import os

    assert os.path.exists(save_path)


def test_plot_attack_comparison(tmp_path):
    results_dict = {
        "FGSM": [
            {"epsilon": 0.0, "adversarial_accuracy": 0.95},
            {"epsilon": 0.1, "adversarial_accuracy": 0.70},
        ],
        "PGD": [
            {"epsilon": 0.0, "adversarial_accuracy": 0.95},
            {"epsilon": 0.1, "adversarial_accuracy": 0.50},
        ],
    }
    save_path = str(tmp_path / "comparison.png")
    fig = plot_attack_comparison(results_dict, save_path=save_path)
    assert fig is not None
    import os

    assert os.path.exists(save_path)
