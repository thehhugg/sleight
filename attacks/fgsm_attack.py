import tensorflow as tf

def fgsm_attack(model, images, labels, epsilon):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = loss_object(labels, prediction)

    gradient = tape.gradient(loss, images)

    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    return tf.clip_by_value(adversarial_images, 0, 1)