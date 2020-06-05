import os

import six
import numpy as np

from tfsnippet.utils import CacheDir, validate_enum_arg

if six.PY2:
    import cPickle as pickle
else:
    import pickle

__all__ = ['load_cifar10', 'load_cifar100']

CIFAR_10_URI = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_10_CONTENT_DIR = 'cifar-10-batches-py'
CIFAR_100_URI = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR_100_CONTENT_DIR = 'cifar-100-python'


def _load_batch(path, channels_last, x_shape, x_dtype, y_dtype, normalize_x,
                expected_batch_label, labels_key='labels'):
    # load from file
    with open(path, 'rb') as f:
        if six.PY2:
            d = pickle.load(f)
        else:
            d = {
                k.decode('utf-8'): v
                for k, v in pickle.load(f, encoding='bytes').items()
            }
            d['batch_label'] = d['batch_label'].decode('utf-8')
    assert(d['batch_label'] == expected_batch_label)

    data = np.asarray(d['data'], dtype=x_dtype)
    labels = np.asarray(d[labels_key], dtype=y_dtype)

    # change shape
    data = data.reshape((data.shape[0], 3, 32, 32))
    if channels_last:
        data = np.transpose(data, (0, 2, 3, 1))
    if x_shape:
        data = data.reshape([data.shape[0]] + list(x_shape))

    # normalize x
    if normalize_x:
        data /= np.asarray(255., dtype=data.dtype)

    return data, labels


def _validate_x_shape(x_shape, channels_last):
    if x_shape is None:
        if channels_last:
            x_shape = (32, 32, 3)
        else:
            x_shape = (3, 32, 32)
    x_shape = tuple([int(v) for v in x_shape])
    if np.prod(x_shape) != 3 * 32 * 32:
        raise ValueError('`x_shape` does not product to 3072: {!r}'.
                         format(x_shape))
    return x_shape


def load_cifar10(channels_last=True, x_shape=None, x_dtype=np.float32,
                 y_dtype=np.int32, normalize_x=False):
    """
    Load the CIFAR-10 dataset as NumPy arrays.

    Args:
        channels_last (bool): Whether or not to place the channels axis
            at the last?
        x_shape: Reshape each digit into this shape.  Default to
            ``(32, 32, 3)`` if `channels_last` is :obj:`True`, otherwise
            default to ``(3, 32, 32)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
    """
    # check the arguments
    x_shape = _validate_x_shape(x_shape, channels_last)

    # fetch data
    path = CacheDir('cifar').download_and_extract(CIFAR_10_URI)
    data_dir = os.path.join(path, CIFAR_10_CONTENT_DIR)

    # load the data
    train_num = 50000
    train_x = np.zeros((train_num,) + x_shape, dtype=x_dtype)
    train_y = np.zeros((train_num,), dtype=y_dtype)

    for i in range(1, 6):
        path = os.path.join(data_dir, 'data_batch_{}'.format(i))
        x, y = _load_batch(
            path, channels_last=channels_last, x_shape=x_shape,
            x_dtype=x_dtype, y_dtype=y_dtype, normalize_x=normalize_x,
            expected_batch_label='training batch {} of 5'.format(i)
        )
        (train_x[(i - 1) * 10000: i * 10000, ...],
         train_y[(i - 1) * 10000: i * 10000]) = x, y

    path = os.path.join(data_dir, 'test_batch')
    test_x, test_y = _load_batch(
        path, channels_last=channels_last, x_shape=x_shape,
        x_dtype=x_dtype, y_dtype=y_dtype, normalize_x=normalize_x,
        expected_batch_label='testing batch 1 of 1'
    )
    assert(len(test_x) == len(test_y) == 10000)

    return (train_x, train_y), (test_x, test_y)


def load_cifar100(label_mode='fine', channels_last=True, x_shape=None,
                  x_dtype=np.float32, y_dtype=np.int32, normalize_x=False):
    """
    Load the CIFAR-100 dataset as NumPy arrays.

    Args:
        label_mode: One of {"fine", "coarse"}.
        channels_last (bool): Whether or not to place the channels axis
            at the last?  Default :obj:`False`.
        x_shape: Reshape each digit into this shape.  Default to
            ``(32, 32, 3)`` if `channels_last` is :obj:`True`, otherwise
            default to ``(3, 32, 32)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
    """
    # check the arguments
    label_mode = validate_enum_arg('label_mode', label_mode, ('fine', 'coarse'))
    x_shape = _validate_x_shape(x_shape, channels_last)

    # fetch data
    path = CacheDir('cifar').download_and_extract(CIFAR_100_URI)
    data_dir = os.path.join(path, CIFAR_100_CONTENT_DIR)

    # load the data
    path = os.path.join(data_dir, 'train')
    train_x, train_y = _load_batch(
        path, channels_last=channels_last, x_shape=x_shape,
        x_dtype=x_dtype, y_dtype=y_dtype, normalize_x=normalize_x,
        expected_batch_label='training batch 1 of 1',
        labels_key='{}_labels'.format(label_mode)
    )
    assert(len(train_x) == len(train_y) == 50000)

    path = os.path.join(data_dir, 'test')
    test_x, test_y = _load_batch(
        path, channels_last=channels_last, x_shape=x_shape,
        x_dtype=x_dtype, y_dtype=y_dtype, normalize_x=normalize_x,
        expected_batch_label='testing batch 1 of 1',
        labels_key='{}_labels'.format(label_mode)
    )
    assert(len(test_x) == len(test_y) == 10000)

    return (train_x, train_y), (test_x, test_y)
