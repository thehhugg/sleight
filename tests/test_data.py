import pytest
import numpy as np
from sleight.data import load_mnist_data, load_fashion_mnist_data, load_cifar10_data, get_dataset

def test_mnist_loader():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    assert x_train.shape[1:] == (28, 28, 1)
    assert x_test.shape[1:] == (28, 28, 1)
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert y_train.ndim == 1
    assert y_test.ndim == 1

def test_fashion_mnist_loader():
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist_data()
    assert x_train.shape[1:] == (28, 28, 1)
    assert x_test.shape[1:] == (28, 28, 1)
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert y_train.ndim == 1
    assert y_test.ndim == 1

def test_cifar10_loader():
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    assert x_train.shape[1:] == (32, 32, 3)
    assert x_test.shape[1:] == (32, 32, 3)
    assert x_train.dtype == np.float32
    assert x_test.dtype == np.float32
    assert y_train.ndim == 1
    assert y_test.ndim == 1

def test_get_dataset_one_hot():
    (x_train, y_train), (x_test, y_test) = get_dataset('mnist', one_hot=True)
    assert y_train.shape[1] == 10
    assert y_test.shape[1] == 10
    assert np.allclose(x_train.max(), 1.0, atol=1e-5)
    assert np.allclose(x_train.min(), 0.0, atol=1e-5)

def test_get_dataset_invalid():
    with pytest.raises(ValueError):
        get_dataset('not_a_real_dataset') 