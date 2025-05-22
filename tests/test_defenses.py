import numpy as np
import tensorflow as tf
from sleight.defenses.adversarial_training import adversarial_train

def get_dummy_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def dummy_attack_fn(model, images, labels, epsilon):
    # Just return the images for testing
    return images

def test_adversarial_train():
    model = get_dummy_model((28, 28, 1), 10)
    x = np.random.rand(8, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, size=(8,))
    trained_model = adversarial_train(model, x, y, dummy_attack_fn, epsilon=0.1, epochs=1, batch_size=4)
    assert isinstance(trained_model, tf.keras.Model)
    preds = trained_model.predict(x)
    assert preds.shape == (8, 10) 