import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.ops import assert_rank, assert_scalar_equal
from tfsnippet.utils import (validate_positive_int_arg, ParamSpec,
                             unflatten_from_ndims,
                             flatten_to_ndims, is_tensor_object, assert_deps,
                             get_shape,
                             add_name_and_scope_arg_doc, model_variable)
from .utils import *
from ..initialization import default_kernel_initializer
from ..utils import validate_weight_norm_arg

__all__ = [
    'conv2d', 'deconv2d',
]


@add_arg_scope
@add_name_and_scope_arg_doc
def conv2d(input,
           out_channels,
           kernel_size,
           strides=(1, 1),
           dilations=1,
           padding='same',
           channels_last=True,
           activation_fn=None,
           normalizer_fn=None,
           weight_norm=False,
           kernel=None,
           kernel_initializer=None,
           kernel_regularizer=None,
           kernel_constraint=None,
           use_bias=None,
           bias=None,
           bias_initializer=tf.zeros_initializer(),
           bias_regularizer=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           scope=None):
    """
    2D convolutional layer.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        out_channels (int): The channel numbers of the output.
        kernel_size (int or (int, int)): Kernel size over spatial dimensions.
        strides (int or (int, int)): Strides over spatial dimensions.
        dilations (int): The dilation factor over spatial dimensions.
        padding: One of {"valid", "same"}, case in-sensitive.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm (bool or (tf.Tensor) -> tf.Tensor)):
            If :obj:`True`, apply :func:`~tfsnippet.layers.weight_norm` on
            `kernel`.  `use_scale` will be :obj:`True` if `normalizer_fn`
            is not specified, and :obj:`False` otherwise.  The axis reduction
            will be determined by the layer.

            If it is a callable function, then it will be used to normalize
            the `kernel` instead of :func:`~tfsnippet.layers.weight_norm`.
            The user must ensure the axis reduction is correct by themselves.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable (bool): Whether or not the parameters are trainable?

    Returns:
        tf.Tensor: The output tensor.
    """
    input, in_channels, data_format = \
        validate_conv2d_input(input, channels_last)
    out_channels = validate_positive_int_arg('out_channels', out_channels)
    dtype = input.dtype.base_dtype

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    strides = validate_conv2d_strides_tuple('strides', strides, channels_last)
    dilations = validate_positive_int_arg('dilations', dilations)

    if dilations > 1 and not channels_last:
        raise ValueError('`channels_last` == False is incompatible with '
                         '`dilations` > 1.')

    if any(i > 1 for i in strides) and dilations > 1:
        raise ValueError('`strides` > 1 is incompatible with `dilations` > 1.')

    weight_norm_fn = validate_weight_norm_arg(
        weight_norm, axis=-1, use_scale=normalizer_fn is None)
    if use_bias is None:
        use_bias = normalizer_fn is None

    # get the specification of outputs and parameters
    kernel_size = validate_conv2d_size_tuple('kernel_size', kernel_size)
    kernel_shape = kernel_size + (in_channels, out_channels)
    bias_shape = (out_channels,)

    # validate the parameters
    if kernel is not None:
        kernel_spec = ParamSpec(shape=kernel_shape, dtype=dtype)
        kernel = kernel_spec.validate('kernel', kernel)
    if kernel_initializer is None:
        kernel_initializer = default_kernel_initializer(weight_norm)
    if bias is not None:
        bias_spec = ParamSpec(shape=bias_shape, dtype=dtype)
        bias = bias_spec.validate('bias', bias)

    # the main part of the conv2d layer
    with tf.variable_scope(scope, default_name=name or 'conv2d'):
        # create the variables
        if kernel is None:
            kernel = model_variable(
                'kernel',
                shape=kernel_shape,
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable
            )

        if weight_norm_fn is not None:
            kernel = weight_norm_fn(kernel)

        if use_bias and bias is None:
            bias = model_variable(
                'bias',
                shape=bias_shape,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )

        # flatten to 4d
        output, s1, s2 = flatten_to_ndims(input, 4)

        # do convolution
        if dilations > 1:
            output = tf.nn.atrous_conv2d(
                value=output,
                filters=kernel,
                rate=dilations,
                padding=padding
            )
        else:
            output = tf.nn.conv2d(
                input=output,
                filter=kernel,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilations=[1] * 4
            )

        # add bias
        if use_bias:
            output = tf.nn.bias_add(output, bias, data_format=data_format)

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

        # unflatten back to original shape
        output = unflatten_from_ndims(output, s1, s2)

    return output


@add_arg_scope
@add_name_and_scope_arg_doc
def deconv2d(input,
             out_channels,
             kernel_size,
             strides=(1, 1),
             padding='same',
             channels_last=True,
             output_shape=None,
             activation_fn=None,
             normalizer_fn=None,
             weight_norm=False,
             kernel=None,
             kernel_initializer=None,
             kernel_regularizer=None,
             kernel_constraint=None,
             use_bias=None,
             bias=None,
             bias_initializer=tf.zeros_initializer(),
             bias_regularizer=None,
             bias_constraint=None,
             trainable=True,
             name=None,
             scope=None):
    """
    2D deconvolutional layer.

    Args:
        input (Tensor): The input tensor, at least 4-d.
        out_channels (int): The channel numbers of the deconvolution output.
        kernel_size (int or (int, int)): Kernel size over spatial dimensions.
        strides (int or (int, int)): Strides over spatial dimensions.
        padding: One of {"valid", "same"}, case in-sensitive.
        channels_last (bool): Whether or not the channel axis is the last
            axis in `input`? (i.e., the data format is "NHWC")
        output_shape: If specified, use this as the shape of the
            deconvolution output; otherwise compute the size of each dimension
            by::

                output_size = input_size * strides
                if padding == 'valid':
                    output_size += max(kernel_size - strides, 0)

        activation_fn: The activation function.
        normalizer_fn: The normalizer function.
        weight_norm (bool or (tf.Tensor) -> tf.Tensor)):
            If :obj:`True`, apply :func:`~tfsnippet.layers.weight_norm` on
            `kernel`.  `use_scale` will be :obj:`True` if `normalizer_fn`
            is not specified, and :obj:`False` otherwise.  The axis reduction
            will be determined by the layer.

            If it is a callable function, then it will be used to normalize
            the `kernel` instead of :func:`~tfsnippet.layers.weight_norm`.
            The user must ensure the axis reduction is correct by themselves.
        kernel (Tensor): Instead of creating a new variable, use this tensor.
        kernel_initializer: The initializer for `kernel`.
            Would be ``default_kernel_initializer(...)`` if not specified.
        kernel_regularizer: The regularizer for `kernel`.
        kernel_constraint: The constraint for `kernel`.
        use_bias (bool or None): Whether or not to use `bias`?
            If :obj:`True`, will always use bias.
            If :obj:`None`, will use bias only if `normalizer_fn` is not given.
            If :obj:`False`, will never use bias.
            Default is :obj:`None`.
        bias (Tensor): Instead of creating a new variable, use this tensor.
        bias_initializer: The initializer for `bias`.
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        trainable (bool): Whether or not the parameters are trainable?

    Returns:
        tf.Tensor: The output tensor.
    """
    input, in_channels, data_format = \
        validate_conv2d_input(input, channels_last)
    out_channels = validate_positive_int_arg('out_channels', out_channels)
    dtype = input.dtype.base_dtype

    # check functional arguments
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['VALID', 'SAME'])
    strides = validate_conv2d_strides_tuple('strides', strides, channels_last)

    weight_norm_fn = validate_weight_norm_arg(
        weight_norm, axis=-1, use_scale=normalizer_fn is None)
    if use_bias is None:
        use_bias = normalizer_fn is None

    # get the specification of outputs and parameters
    kernel_size = validate_conv2d_size_tuple('kernel_size', kernel_size)
    kernel_shape = kernel_size + (out_channels, in_channels)
    bias_shape = (out_channels,)

    given_h, given_w = None, None
    given_output_shape = output_shape

    if is_tensor_object(given_output_shape):
        given_output_shape = tf.convert_to_tensor(given_output_shape)
    elif given_output_shape is not None:
        given_h, given_w = given_output_shape

    # validate the parameters
    if kernel is not None:
        kernel_spec = ParamSpec(shape=kernel_shape, dtype=dtype)
        kernel = kernel_spec.validate('kernel', kernel)
    if kernel_initializer is None:
        kernel_initializer = default_kernel_initializer(weight_norm)
    if bias is not None:
        bias_spec = ParamSpec(shape=bias_shape, dtype=dtype)
        bias = bias_spec.validate('bias', bias)

    # the main part of the conv2d layer
    with tf.variable_scope(scope, default_name=name or 'deconv2d'):
        with tf.name_scope('output_shape'):
            # detect the input shape and axis arrangements
            input_shape = get_static_shape(input)
            if channels_last:
                c_axis, h_axis, w_axis = -1, -3, -2
            else:
                c_axis, h_axis, w_axis = -3, -2, -1

            output_shape = [None, None, None, None]
            output_shape[c_axis] = out_channels
            if given_output_shape is None:
                if input_shape[h_axis] is not None:
                    output_shape[h_axis] = get_deconv_output_length(
                        input_shape[h_axis], kernel_shape[0], strides[h_axis],
                        padding
                    )
                if input_shape[w_axis] is not None:
                    output_shape[w_axis] = get_deconv_output_length(
                        input_shape[w_axis], kernel_shape[1], strides[w_axis],
                        padding
                    )
            else:
                if not is_tensor_object(given_output_shape):
                    output_shape[h_axis] = given_h
                    output_shape[w_axis] = given_w

            # infer the batch shape in 4-d
            batch_shape = input_shape[:-3]
            if None not in batch_shape:
                output_shape[0] = int(np.prod(batch_shape))

            # now the static output shape is ready
            output_static_shape = tf.TensorShape(output_shape)

            # prepare for the dynamic batch shape
            if output_shape[0] is None:
                output_shape[0] = tf.reduce_prod(get_shape(input)[:-3])

            # prepare for the dynamic spatial dimensions
            if output_shape[h_axis] is None or output_shape[w_axis] is None:
                if given_output_shape is None:
                    input_shape = get_shape(input)
                    if output_shape[h_axis] is None:
                        output_shape[h_axis] = get_deconv_output_length(
                            input_shape[h_axis], kernel_shape[0],
                            strides[h_axis], padding
                        )
                    if output_shape[w_axis] is None:
                        output_shape[w_axis] = get_deconv_output_length(
                            input_shape[w_axis], kernel_shape[1],
                            strides[w_axis], padding
                        )
                else:
                    assert(is_tensor_object(given_output_shape))
                    with assert_deps([
                                assert_rank(given_output_shape, 1),
                                assert_scalar_equal(
                                    tf.size(given_output_shape), 2)
                            ]):
                        output_shape[h_axis] = given_output_shape[0]
                        output_shape[w_axis] = given_output_shape[1]

            # compose the final dynamic shape
            if any(is_tensor_object(s) for s in output_shape):
                output_shape = tf.stack(output_shape)
            else:
                output_shape = tuple(output_shape)

        # create the variables
        if kernel is None:
            kernel = model_variable(
                'kernel',
                shape=kernel_shape,
                dtype=dtype,
                initializer=kernel_initializer,
                regularizer=kernel_regularizer,
                constraint=kernel_constraint,
                trainable=trainable
            )

        if weight_norm_fn is not None:
            kernel = weight_norm_fn(kernel)

        if use_bias and bias is None:
            bias = model_variable(
                'bias',
                shape=bias_shape,
                initializer=bias_initializer,
                regularizer=bias_regularizer,
                constraint=bias_constraint,
                trainable=trainable
            )

        # flatten to 4d
        output, s1, s2 = flatten_to_ndims(input, 4)

        # do convolution or deconvolution
        output = tf.nn.conv2d_transpose(
            value=output,
            filter=kernel,
            output_shape=output_shape,
            strides=strides,
            padding=padding,
            data_format=data_format
        )
        if output_static_shape is not None:
            output.set_shape(output_static_shape)

        # add bias
        if use_bias:
            output = tf.nn.bias_add(output, bias, data_format=data_format)

        # apply the normalization function if specified
        if normalizer_fn is not None:
            output = normalizer_fn(output)

        # apply the activation function if specified
        if activation_fn is not None:
            output = activation_fn(output)

        # unflatten back to original shape
        output = unflatten_from_ndims(output, s1, s2)

    return output
