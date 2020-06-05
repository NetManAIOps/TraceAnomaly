from functools import partial

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.utils import (validate_int_tuple_arg, is_integer,
                             add_name_and_scope_arg_doc, InputSpec,
                             get_static_shape)
from .conv2d_ import conv2d, deconv2d

__all__ = [
    'resnet_general_block',
    'resnet_conv2d_block',
    'resnet_deconv2d_block',
]


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_general_block(conv_fn,
                         input,
                         in_channels,
                         out_channels,
                         kernel_size,
                         strides=1,
                         shortcut_kernel_size=1,
                         resize_at_exit=False,
                         activation_fn=None,
                         normalizer_fn=None,
                         dropout_fn=None,
                         name=None,
                         scope=None):
    """
    A general implementation of ResNet block.

    The architecture of this ResNet implementation follows the work
    "Wide residual networks" (Zagoruyko & Komodakis, 2016).  It basically does
    the following things:

    .. code-block:: python

        shortcut = input
        if strides != 1 or in_channels != out_channels:
            shortcut = conv_fn(
                input=shortcut,
                out_channels=out_channels,
                kernel_size=shortcut_kernel_size,
                strides=strides,
                name='shortcut'
            )

        residual = input
        residual = conv_fn(
            input=activation_fn(normalizer_fn(residual)),
            out_channels=in_channels if resize_at_exit else out_channels,
            kernel_size=kernel_size,
            strides=strides,
            name='conv'
        )
        residual = dropout_fn(residual)
        residual = conv_fn(
            input=activation_fn(normalizer_fn(residual)),
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            name='conv_1'
        )

        output = shortcut + residual

    Args:
        conv_fn: The convolution function for "shortcut", "conv" and "conv_1"
            convolutional layers.  It must accept, and only expect, 5 named
            arguments ``(input, out_channels, kernel_size, strides, name)``.
        input (Tensor): The input tensor.
        in_channels (int): The channel numbers of the tensor.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv" and "conv_1" convolutional layers.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for all three convolutional layers.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" convolutional layer.
        resize_at_exit (bool): If :obj:`True`, resize the spatial dimensions
            at the "conv_1" convolutional layer.  If :obj:`False`, resize at
            the "conv" convolutional layer. (see above)
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        dropout_fn: The dropout function.

    Returns:
        tf.Tensor: The output tensor.
    """
    def validate_size_tuple(n, s):
        if is_integer(s):
            # Do not change a single integer into a tuple!
            # This is because we do not know the dimensionality of the
            # convolution operation here, so we cannot build the size
            # tuple with correct number of elements from the integer notation.
            return int(s)
        return validate_int_tuple_arg(n, s)

    def has_non_unit_item(x):
        if is_integer(x):
            return x != 1
        else:
            return any(i != 1 for i in x)

    # check the parameters
    input = tf.convert_to_tensor(input)
    in_channels = int(in_channels)
    out_channels = int(out_channels)
    kernel_size = validate_size_tuple('kernel_size', kernel_size)
    strides = validate_size_tuple('strides', strides)
    shortcut_kernel_size = validate_size_tuple(
        'shortcut_kernel_size', shortcut_kernel_size)

    # define two types of convolution operations: resizing conv, and
    # size keeping conv
    def resize_conv(input, kernel_size, name):
        return conv_fn(input=input, out_channels=out_channels,
                       kernel_size=kernel_size, strides=strides,
                       name=name)

    def keep_conv(input, kernel_size, n_channels, name):
        return conv_fn(input=input, out_channels=n_channels,
                       kernel_size=kernel_size, strides=1,
                       name=name)

    # define a helper to apply fn on input `x`
    def apply_fn(fn, x):
        if fn is not None:
            x = fn(x)
        return x

    with tf.variable_scope(scope, default_name=name or 'resnet_general_block'):
        # build the shortcut path
        if has_non_unit_item(strides) or in_channels != out_channels:
            shortcut = resize_conv(input, shortcut_kernel_size, name='shortcut')
        else:
            shortcut = input

        # build the residual path
        if resize_at_exit:
            conv0 = partial(keep_conv, kernel_size=kernel_size, name='conv',
                            n_channels=in_channels)
            conv1 = partial(resize_conv, kernel_size=kernel_size, name='conv_1')
        else:
            conv0 = partial(resize_conv, kernel_size=kernel_size, name='conv')
            conv1 = partial(keep_conv, kernel_size=kernel_size, name='conv_1',
                            n_channels=out_channels)

        with tf.variable_scope('residual'):
            residual = input
            residual = apply_fn(normalizer_fn, residual)
            residual = apply_fn(activation_fn, residual)
            residual = conv0(residual)
            residual = apply_fn(dropout_fn, residual)
            residual = apply_fn(normalizer_fn, residual)
            residual = apply_fn(activation_fn, residual)
            residual = conv1(residual)

        # merge the shortcut path and the residual path
        output = shortcut + residual

    return output


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_conv2d_block(input,
                        out_channels,
                        kernel_size,
                        strides=(1, 1),
                        shortcut_kernel_size=(1, 1),
                        channels_last=True,
                        resize_at_exit=True,
                        activation_fn=None,
                        normalizer_fn=None,
                        weight_norm=False,
                        dropout_fn=None,
                        kernel_initializer=None,
                        kernel_regularizer=None,
                        kernel_constraint=None,
                        use_bias=None,
                        bias_initializer=tf.zeros_initializer(),
                        bias_regularizer=None,
                        bias_constraint=None,
                        trainable=True,
                        name=None,
                        scope=None):
    """
    2D convolutional ResNet block.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv" and "conv_1" convolutional layers.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for all three convolutional layers.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" convolutional layer.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        resize_at_exit (bool): See :func:`resnet_general_block`.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm: Passed to :func:`conv2d`.
        dropout_fn: The dropout function.
        kernel_initializer: Passed to :func:`conv2d`.
        kernel_regularizer: Passed to :func:`conv2d`.
        kernel_constraint: Passed to :func:`conv2d`.
        use_bias: Whether or not to use `bias` in :func:`conv2d`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias_initializer: Passed to :func:`conv2d`.
        bias_regularizer: Passed to :func:`conv2d`.
        bias_constraint: Passed to :func:`conv2d`.
        trainable: Passed to :func:`conv2d`.

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`resnet_general_block`
    """
    # check the input and infer the input shape
    if channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
        c_axis = -1
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
        c_axis = -3
    input = input_spec.validate('input', input)
    in_channels = get_static_shape(input)[c_axis]

    # check the functional arguments
    if use_bias is None:
        use_bias = normalizer_fn is None

    # derive the convolution function
    conv_fn = partial(
        conv2d,
        channels_last=channels_last,
        weight_norm=weight_norm,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        bias_constraint=bias_constraint,
        trainable=trainable,
    )

    # build the resnet block
    return resnet_general_block(
        conv_fn,
        input=input,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        shortcut_kernel_size=shortcut_kernel_size,
        resize_at_exit=resize_at_exit,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
        name=name or 'resnet_conv2d_block',
        scope=scope
    )


@add_arg_scope
@add_name_and_scope_arg_doc
def resnet_deconv2d_block(input,
                          out_channels,
                          kernel_size,
                          strides=(1, 1),
                          shortcut_kernel_size=(1, 1),
                          channels_last=True,
                          resize_at_exit=False,
                          activation_fn=None,
                          normalizer_fn=None,
                          weight_norm=False,
                          dropout_fn=None,
                          kernel_initializer=None,
                          kernel_regularizer=None,
                          kernel_constraint=None,
                          use_bias=None,
                          bias_initializer=tf.zeros_initializer(),
                          bias_regularizer=None,
                          bias_constraint=None,
                          trainable=True,
                          name=None,
                          scope=None):
    """
    2D deconvolutional ResNet block.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or tuple[int]): Kernel size over spatial dimensions,
            for "conv" and "conv_1" deconvolutional layers.
        strides (int or tuple[int]): Strides over spatial dimensions,
            for all three deconvolutional layers.
        shortcut_kernel_size (int or tuple[int]): Kernel size over spatial
            dimensions, for the "shortcut" deconvolutional layer.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        resize_at_exit (bool): See :func:`resnet_general_block`.
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm: Passed to :func:`deconv2d`.
        dropout_fn: The dropout function.
        kernel_initializer: Passed to :func:`deconv2d`.
        kernel_regularizer: Passed to :func:`deconv2d`.
        kernel_constraint: Passed to :func:`deconv2d`.
        use_bias: Whether or not to use `bias` in :func:`deconv2d`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias_initializer: Passed to :func:`deconv2d`.
        bias_regularizer: Passed to :func:`deconv2d`.
        bias_constraint: Passed to :func:`deconv2d`.
        trainable: Passed to :func:`convdeconv2d2d`.

    Returns:
        tf.Tensor: The output tensor.

    See Also:
        :func:`resnet_general_block`
    """
    # check the input and infer the input shape
    if channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
        c_axis = -1
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
        c_axis = -3
    input = input_spec.validate('input', input)
    in_channels = get_static_shape(input)[c_axis]

    # check the functional arguments
    if use_bias is None:
        use_bias = normalizer_fn is None

    # derive the convolution function
    conv_fn = partial(
        deconv2d,
        channels_last=channels_last,
        weight_norm=weight_norm,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint,
        use_bias=use_bias,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        bias_constraint=bias_constraint,
        trainable=trainable,
    )

    # build the resnet block
    return resnet_general_block(
        conv_fn,
        input=input,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        shortcut_kernel_size=shortcut_kernel_size,
        resize_at_exit=resize_at_exit,
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        dropout_fn=dropout_fn,
        name=name or 'resnet_deconv2d_block',
        scope=scope
    )
