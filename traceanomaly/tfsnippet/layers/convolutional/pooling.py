import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (validate_enum_arg, flatten_to_ndims, unflatten_from_ndims,
                             add_name_arg_doc)

from .utils import validate_conv2d_strides_tuple, validate_conv2d_input

__all__ = ['avg_pool2d', 'max_pool2d', 'global_avg_pool2d']


def _pool2d(pool_fn, input, pool_size, strides=(1, 1), channels_last=True,
            padding='same', name=None, default_name=None):
    input, _, data_format = validate_conv2d_input(input, channels_last)

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    strides = validate_conv2d_strides_tuple('strides', strides, channels_last)
    ksize = validate_conv2d_strides_tuple('pool_size', pool_size, channels_last)

    # call pooling
    with tf.name_scope(name, default_name=default_name):
        output, s1, s2 = flatten_to_ndims(input, 4)
        output = pool_fn(
            value=output, ksize=ksize, strides=strides, padding=padding,
            data_format=data_format
        )
        output = unflatten_from_ndims(output, s1, s2)
    return output


@add_arg_scope
@add_name_arg_doc
def avg_pool2d(input, pool_size, strides=(1, 1), channels_last=True,
               padding='same', name=None):
    """
    2D average pooling over spatial dimensions.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        pool_size (int or (int, int)): Pooling size over spatial dimensions.
        strides (int or (int, int)): Strides over spatial dimensions.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        padding: One of {"valid", "same"}, case in-sensitive.

    Returns:
        tf.Tensor: The output tensor.
    """
    return _pool2d(
        tf.nn.avg_pool,
        input=input,
        pool_size=pool_size,
        strides=strides,
        channels_last=channels_last,
        padding=padding,
        name=name,
        default_name='avg_pool2d'
    )


@add_arg_scope
@add_name_arg_doc
def max_pool2d(input, pool_size, strides=(1, 1), channels_last=True,
               padding='same', name=None):
    """
    2D max pooling over spatial dimensions.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        pool_size (int or (int, int)): Pooling size over spatial dimensions.
        strides (int or (int, int)): Strides over spatial dimensions.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        padding: One of {"valid", "same"}, case in-sensitive.

    Returns:
        tf.Tensor: The output tensor.
    """
    return _pool2d(
        tf.nn.max_pool,
        input=input,
        pool_size=pool_size,
        strides=strides,
        channels_last=channels_last,
        padding=padding,
        name=name,
        default_name='max_pool2d'
    )


@add_arg_scope
@add_name_arg_doc
def global_avg_pool2d(input, channels_last=True, keepdims=False, name=None):
    """
    2D global average pooling over spatial dimensions.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        keepdims (bool): Whether or not to keep the reduced spatial dimensions?
            Default is :obj:`False`.

    Returns:
        tf.Tensor: The output tensor.
    """
    input, _, data_format = validate_conv2d_input(input, channels_last)
    if channels_last:
        reduce_axis = [-3, -2]
    else:
        reduce_axis = [-2, -1]

    with tf.name_scope(name, default_name='global_avg_pool2d'):
        return tf.reduce_mean(input, axis=reduce_axis, keepdims=keepdims)
