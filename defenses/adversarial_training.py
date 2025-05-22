import tensorflow as tf
import numpy as np


def adversarial_train(model, x_train, y_train, attack_fn, epsilon=0.1, epochs=3, batch_size=64):
    """
    Perform adversarial training on a Keras model.
    Args:
        model: tf.keras.Model, the model to train
        x_train: np.ndarray, training images
        y_train: np.ndarray, training labels
        attack_fn: function(model, images, labels, epsilon) -> adversarial images
        epsilon: float, perturbation strength
        epochs: int, number of epochs
        batch_size: int, batch size
    Returns:
        Trained model
    """
    num_batches = int(np.ceil(len(x_train) / batch_size))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=model.output_shape[-1])

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        idx = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[idx]
        y_train_shuffled = y_train_cat[idx]
        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(x_train))
            x_batch = x_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Generate adversarial examples
            x_adv = attack_fn(model, x_batch, y_batch, epsilon)
            x_combined = np.concatenate([x_batch, x_adv], axis=0)
            y_combined = np.concatenate([y_batch, y_batch], axis=0)

            # Train on both clean and adversarial examples
            model.train_on_batch(x_combined, y_combined)
    return model 