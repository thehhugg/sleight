import numpy as np
import tensorflow as tf
from sleight.attacks.fgsm_attack import fgsm_attack
from sleight.attacks.pgd_attack import pgd_attack


def get_dummy_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


def test_fgsm_attack():
    model = get_dummy_model((28, 28, 1), 10)
    x = np.random.rand(2, 28, 28, 1).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.array([1, 2]), 10)
    adv = fgsm_attack(model, x, y, epsilon=0.1)
    assert adv.shape == x.shape
    assert np.all(adv >= 0) and np.all(adv <= 1)


def test_pgd_attack():
    model = get_dummy_model((28, 28, 1), 10)
    x = np.random.rand(2, 28, 28, 1).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.array([1, 2]), 10)
    adv = pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, num_iter=5)
    assert adv.shape == x.shape
    assert np.all(adv >= 0) and np.all(adv <= 1)
