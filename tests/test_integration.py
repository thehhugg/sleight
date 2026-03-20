"""Integration tests and edge cases for sleight.

These tests verify that the components work together end-to-end
and handle boundary conditions correctly.
"""

import numpy as np
import pytest
import tensorflow as tf

from sleight.attacks import fgsm_attack, pgd_attack, cw_attack, deepfool_attack
from sleight.defenses import (
    adversarial_train,
    jpeg_compression,
    spatial_smoothing,
    bit_depth_reduction,
)
from sleight.evaluation import evaluate_robustness, epsilon_sweep


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_model(input_shape=(8, 8, 1), num_classes=5):
    """Build a tiny model for fast integration tests."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _make_data(n=30, input_shape=(8, 8, 1), num_classes=5):
    """Generate random data for testing."""
    x = np.random.rand(n, *input_shape).astype(np.float32)
    y_int = np.random.randint(0, num_classes, size=n)
    y_onehot = tf.keras.utils.to_categorical(y_int, num_classes)
    return x, y_int, y_onehot


# ---------------------------------------------------------------------------
# Integration: attack -> evaluate pipeline
# ---------------------------------------------------------------------------

class TestAttackEvaluationPipeline:
    """Test that each attack integrates with the evaluation module."""

    def setup_method(self):
        self.model = _make_model()
        self.x, self.y_int, self.y_onehot = _make_data()
        self.model.fit(self.x, self.y_onehot, epochs=2, verbose=0)

    def test_fgsm_evaluation(self):
        result = evaluate_robustness(self.model, self.x, self.y_int, fgsm_attack, 0.1)
        assert 0 <= result["adversarial_accuracy"] <= 1
        assert 0 <= result["attack_success_rate"] <= 1

    def test_pgd_evaluation(self):
        result = evaluate_robustness(
            self.model, self.x, self.y_int, pgd_attack, 0.1, alpha=0.01, num_iter=3
        )
        assert 0 <= result["adversarial_accuracy"] <= 1

    def test_cw_evaluation(self):
        result = evaluate_robustness(
            self.model, self.x, self.y_int, cw_attack, 1.0, c=1.0, num_iter=5
        )
        assert 0 <= result["adversarial_accuracy"] <= 1

    def test_deepfool_evaluation(self):
        result = evaluate_robustness(
            self.model, self.x[:5], self.y_int[:5], deepfool_attack, 3.0, max_iter=5
        )
        assert 0 <= result["adversarial_accuracy"] <= 1

    def test_epsilon_sweep_fgsm(self):
        results = epsilon_sweep(
            self.model, self.x, self.y_int, fgsm_attack, [0.0, 0.1, 0.3]
        )
        assert len(results) == 3
        # At epsilon=0, adversarial accuracy should equal clean accuracy
        assert results[0]["adversarial_accuracy"] == results[0]["clean_accuracy"]


# ---------------------------------------------------------------------------
# Integration: defense -> attack pipeline
# ---------------------------------------------------------------------------

class TestDefenseAttackPipeline:
    """Test that defenses can be applied to adversarial images."""

    def setup_method(self):
        self.model = _make_model()
        self.x, self.y_int, self.y_onehot = _make_data()
        self.model.fit(self.x, self.y_onehot, epochs=2, verbose=0)
        self.adv = np.array(fgsm_attack(self.model, self.x, self.y_onehot, 0.2))

    def test_jpeg_then_predict(self):
        defended = jpeg_compression(self.adv, quality=50)
        preds = self.model.predict(defended, verbose=0)
        assert preds.shape == (len(self.x), 5)

    def test_smoothing_then_predict(self):
        defended = spatial_smoothing(self.adv, kernel_size=3)
        preds = self.model.predict(defended, verbose=0)
        assert preds.shape == (len(self.x), 5)

    def test_bit_depth_then_predict(self):
        defended = bit_depth_reduction(self.adv, bits=3)
        preds = self.model.predict(defended, verbose=0)
        assert preds.shape == (len(self.x), 5)


# ---------------------------------------------------------------------------
# Edge cases: attacks
# ---------------------------------------------------------------------------

class TestAttackEdgeCases:
    """Test attacks with edge-case inputs."""

    def setup_method(self):
        self.model = _make_model()
        self.x, self.y_int, self.y_onehot = _make_data(n=4)

    def test_fgsm_epsilon_zero(self):
        """Epsilon=0 should return the original images."""
        adv = np.array(fgsm_attack(self.model, self.x, self.y_onehot, epsilon=0.0))
        np.testing.assert_allclose(adv, self.x, atol=1e-6)

    def test_pgd_epsilon_zero(self):
        """PGD with epsilon=0 should return originals."""
        adv = np.array(pgd_attack(self.model, self.x, self.y_onehot, epsilon=0.0, alpha=0.01, num_iter=3))
        np.testing.assert_allclose(adv, self.x, atol=1e-6)

    def test_fgsm_large_epsilon(self):
        """Large epsilon should still clip to [0, 1]."""
        adv = np.array(fgsm_attack(self.model, self.x, self.y_onehot, epsilon=10.0))
        assert np.all(adv >= 0) and np.all(adv <= 1)

    def test_pgd_large_epsilon(self):
        adv = np.array(pgd_attack(self.model, self.x, self.y_onehot, epsilon=10.0, alpha=1.0, num_iter=3))
        assert np.all(adv >= 0) and np.all(adv <= 1)

    def test_cw_single_image(self):
        """C&W should work with a single image."""
        adv = cw_attack(self.model, self.x[:1], self.y_onehot[:1], epsilon=1.0, num_iter=5)
        assert adv.shape == self.x[:1].shape

    def test_deepfool_single_image(self):
        """DeepFool should work with a single image."""
        adv = deepfool_attack(self.model, self.x[:1], self.y_int[:1], epsilon=3.0, max_iter=5)
        assert adv.shape == self.x[:1].shape

    def test_fgsm_integer_labels(self):
        """FGSM should accept integer labels."""
        adv = np.array(fgsm_attack(self.model, self.x, self.y_int, epsilon=0.1))
        assert adv.shape == self.x.shape

    def test_pgd_integer_labels(self):
        """PGD should accept integer labels."""
        adv = np.array(pgd_attack(self.model, self.x, self.y_int, epsilon=0.1, alpha=0.01, num_iter=3))
        assert adv.shape == self.x.shape


# ---------------------------------------------------------------------------
# Edge cases: defenses
# ---------------------------------------------------------------------------

class TestDefenseEdgeCases:
    """Test defenses with edge-case inputs."""

    def test_jpeg_all_zeros(self):
        images = np.zeros((2, 16, 16, 1), dtype=np.float32)
        result = jpeg_compression(images, quality=50)
        assert result.shape == images.shape
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_jpeg_all_ones(self):
        images = np.ones((2, 16, 16, 1), dtype=np.float32)
        result = jpeg_compression(images, quality=50)
        assert result.shape == images.shape

    def test_smoothing_kernel_1(self):
        """Kernel size 1 should be identity."""
        images = np.random.rand(2, 8, 8, 1).astype(np.float32)
        result = spatial_smoothing(images, kernel_size=1)
        np.testing.assert_allclose(result, images, atol=1e-6)

    def test_smoothing_invalid_kernel(self):
        """Even kernel size should raise ValueError."""
        images = np.random.rand(2, 8, 8, 1).astype(np.float32)
        with pytest.raises(ValueError):
            spatial_smoothing(images, kernel_size=4)

    def test_bit_depth_8_bits(self):
        """8-bit reduction should be near-identity."""
        images = np.random.rand(2, 8, 8, 1).astype(np.float32)
        result = bit_depth_reduction(images, bits=8)
        np.testing.assert_allclose(result, images, atol=1.0 / 255 + 1e-6)

    def test_bit_depth_1_bit(self):
        """1-bit should produce only 0 and 1."""
        images = np.random.rand(2, 8, 8, 1).astype(np.float32)
        result = bit_depth_reduction(images, bits=1)
        unique = np.unique(result)
        assert all(v in [0.0, 1.0] for v in unique)

    def test_bit_depth_invalid(self):
        images = np.random.rand(2, 8, 8, 1).astype(np.float32)
        with pytest.raises(ValueError):
            bit_depth_reduction(images, bits=0)
        with pytest.raises(ValueError):
            bit_depth_reduction(images, bits=9)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    """Smoke test the CLI module."""

    def test_list_command(self, capsys):
        from sleight.cli import main
        main(["list"])
        captured = capsys.readouterr()
        assert "fgsm" in captured.out
        assert "pgd" in captured.out
        assert "adversarial_training" in captured.out

    def test_help(self):
        from sleight.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
