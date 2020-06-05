import tensorflow as tf

from tfsnippet.utils import (add_name_arg_doc, flatten_to_ndims,
                             unflatten_from_ndims)

__all__ = ['space_to_depth', 'depth_to_space']


@add_name_arg_doc
def space_to_depth(input, block_size, channels_last=True, name=None):
    """
    Wraps :func:`tf.space_to_depth`, to support tensors higher than 4-d.

    Args:
        input: The input tensor, at least 4-d.
        block_size (int): An int >= 2, the size of the spatial block.
        channels_last (bool): Whether or not the channels axis
            is the last axis in the input tensor?

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`tf.space_to_depth`
    """
    block_size = int(block_size)
    data_format = 'NHWC' if channels_last else 'NCHW'
    input = tf.convert_to_tensor(input)
    with tf.name_scope(name or 'space_to_depth', values=[input]):
        output, s1, s2 = flatten_to_ndims(input, ndims=4)
        output = tf.space_to_depth(output, block_size, data_format=data_format)
        output = unflatten_from_ndims(output, s1, s2)
        return output


@add_name_arg_doc
def depth_to_space(input, block_size, channels_last=True, name=None):
    """
    Wraps :func:`tf.depth_to_space`, to support tensors higher than 4-d.

    Args:
        input: The input tensor, at least 4-d.
        block_size (int): An int >= 2, the size of the spatial block.
        channels_last (bool): Whether or not the channels axis
            is the last axis in the input tensor?

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`tf.depth_to_space`
    """
    block_size = int(block_size)
    data_format = 'NHWC' if channels_last else 'NCHW'
    input = tf.convert_to_tensor(input)
    with tf.name_scope(name or 'space_to_depth', values=[input]):
        output, s1, s2 = flatten_to_ndims(input, ndims=4)
        output = tf.depth_to_space(output, block_size, data_format=data_format)
        output = unflatten_from_ndims(output, s1, s2)
        return output
