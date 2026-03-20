import tensorflow as tf


def fgsm_attack(model, images, labels, epsilon):
    """Perform the Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: tf.keras.Model, the model to attack.
        images: Input images (tf.Tensor or np.ndarray).
        labels: One-hot encoded labels (tf.Tensor or np.ndarray).
        epsilon: Float, maximum perturbation magnitude.

    Returns:
        Adversarial images as a tf.Tensor, clipped to [0, 1].
    """
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = loss_object(labels, prediction)

    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    return tf.clip_by_value(adversarial_images, 0, 1)
