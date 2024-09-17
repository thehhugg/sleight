import tensorflow as tf

def fgsm_attack(model, images, labels, epsilon):
    # Get gradients of loss, re: input image
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = loss_object(labels, prediction)

    gradient = tape.gradient(loss, images)

    #Get sign of gradients to create perturbation
    signed_grad = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_grad
    return tf.clip_by_value(adversarial_images, 0, 1)