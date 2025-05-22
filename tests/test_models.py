import tensorflow as tf
from sleight.models.mnist_cnn import get_mnist_cnn_model
from sleight.models.cifar10_cnn import get_cifar10_cnn_model

def test_mnist_cnn_model():
    model = get_mnist_cnn_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape[1:] == (28, 28, 1)
    assert model.output_shape[-1] == 10
    assert model.loss

def test_cifar10_cnn_model():
    model = get_cifar10_cnn_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape[1:] == (32, 32, 3)
    assert model.output_shape[-1] == 10
    assert model.loss 