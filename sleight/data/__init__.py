from .mnist import load_mnist_data
from .fashion_mnist import load_fashion_mnist_data
from .cifar10 import load_cifar10_data

data_registry = {
    "mnist": load_mnist_data,
    "fashion_mnist": load_fashion_mnist_data,
    "cifar10": load_cifar10_data,
}

def get_dataset(name, one_hot=False):
    """
    Load a dataset by name.
    Args:
        name (str): Dataset name ('mnist', 'fashion_mnist', 'cifar10')
        one_hot (bool): Whether to return one-hot labels
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    if name not in data_registry:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(data_registry.keys())}")
    (x_train, y_train), (x_test, y_test) = data_registry[name]()
    if one_hot:
        import tensorflow as tf
        num_classes = len(set(y_train))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test) 