import numpy as np
import tensorflow as tf
from sleight.defenses.adversarial_training import adversarial_train
from sleight.defenses.input_transforms import (
    jpeg_compression,
    spatial_smoothing,
    bit_depth_reduction,
)
from sleight.defenses.distillation import defensive_distillation


def get_dummy_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def dummy_attack_fn(model, images, labels, epsilon):
    # Just return the images for testing
    return images


def test_adversarial_train():
    model = get_dummy_model((28, 28, 1), 10)
    x = np.random.rand(8, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, size=(8,))
    trained_model = adversarial_train(
        model, x, y, dummy_attack_fn, epsilon=0.1, epochs=1, batch_size=4
    )
    assert isinstance(trained_model, tf.keras.Model)
    preds = trained_model.predict(x)
    assert preds.shape == (8, 10)


def test_jpeg_compression():
    images = np.random.rand(3, 28, 28, 1).astype(np.float32)
    result = jpeg_compression(images, quality=50)
    assert result.shape == images.shape
    assert np.all(result >= 0) and np.all(result <= 1)
    # JPEG should change the image (lossy compression)
    assert not np.allclose(result, images, atol=0.01)


def test_jpeg_compression_rgb():
    images = np.random.rand(2, 32, 32, 3).astype(np.float32)
    result = jpeg_compression(images, quality=75)
    assert result.shape == images.shape
    assert np.all(result >= 0) and np.all(result <= 1)


def test_spatial_smoothing():
    images = np.random.rand(3, 28, 28, 1).astype(np.float32)
    result = spatial_smoothing(images, kernel_size=3)
    assert result.shape == images.shape
    assert np.all(result >= 0) and np.all(result <= 1)


def test_bit_depth_reduction():
    images = np.random.rand(3, 28, 28, 1).astype(np.float32)
    result = bit_depth_reduction(images, bits=4)
    assert result.shape == images.shape
    assert np.all(result >= 0) and np.all(result <= 1)
    # Check quantization: values should be multiples of 1/15
    unique_vals = np.unique(np.round(result * 15) / 15)
    assert len(unique_vals) <= 16


def test_defensive_distillation():
    def model_fn():
        m = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        return m

    x = np.random.rand(16, 4, 4, 1).astype(np.float32)
    y = np.random.randint(0, 10, size=16)
    student = defensive_distillation(
        model_fn, x, y, temperature=10.0, epochs=2, batch_size=8
    )
    assert isinstance(student, tf.keras.Model)
    preds = student.predict(x)
    assert preds.shape == (16, 10)
