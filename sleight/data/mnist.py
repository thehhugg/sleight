import tensorflow as tf
import numpy as np

def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis].astype("float32")
    x_test = x_test[..., np.newaxis].astype("float32")
    return (x_train, y_train), (x_test, y_test) 