import tensorflow as tf

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=40):
    """
    Perform the PGD (Projected Gradient Descent) attack.
    Args:
        model: tf.keras.Model, the model to attack
        images: tf.Tensor or np.ndarray, input images
        labels: tf.Tensor or np.ndarray, one-hot or categorical labels
        epsilon: float, max perturbation
        alpha: float, step size
        num_iter: int, number of iterations
    Returns:
        Adversarial images (tf.Tensor)
    """
    images = tf.cast(tf.convert_to_tensor(images), tf.float32)
    labels = tf.cast(tf.convert_to_tensor(labels), tf.float32)

    adv_images = tf.identity(images)
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            prediction = model(adv_images)
            loss = tf.keras.losses.CategoricalCrossentropy()(labels, prediction)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0, 1)
    return adv_images 