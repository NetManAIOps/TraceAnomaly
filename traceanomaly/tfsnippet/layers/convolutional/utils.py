from tfsnippet.utils import (validate_int_tuple_arg, InputSpec,
                             get_static_shape, validate_enum_arg)


def validate_conv2d_input(input, channels_last):
    """
    Validate the input for 2-d convolution.

    Args:
        input: The input tensor, must be at least 4-d.
        channels_last (bool): Whether or not the last dimension is the
            channels dimension? (i.e., `data_format` is "NHWC")

    Returns:
        (tf.Tensor, int, str): The validated input tensor, the number of input
            channels, and the data format.
    """
    if channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
        channel_axis = -1
        data_format = 'NHWC'
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
        channel_axis = -3
        data_format = 'NCHW'
    input = input_spec.validate('input', input)
    input_shape = get_static_shape(input)
    in_channels = input_shape[channel_axis]

    return input, in_channels, data_format


def validate_conv2d_size_tuple(arg_name, arg_value):
    """
    Validate the `arg_value`, ensure it is one or two positive integers,
    such that it can be used as the kernel size.

    Args:
        arg_name: Name of the argument.
        arg_value: An integer, or a tuple of two integers.

    Returns:
        (int, int): The validated two integers.
    """
    arg_value = validate_int_tuple_arg(arg_name, arg_value)
    if len(arg_value) not in (1, 2) or any(a < 1 for a in arg_value):
        raise ValueError('Invalid value for argument `{}`: expected to be '
                         'one or two positive integers, but got {!r}.'.
                         format(arg_name, arg_value))
    if len(arg_value) == 1:
        arg_value = arg_value * 2
    return arg_value


def validate_conv2d_strides_tuple(arg_name, arg_value, channels_last):
    """
    Validate the `arg_value`, ensure it is one or two positive integers,
    such that is can be used as the strides.

    Args:
        arg_name: Name of the argument.
        arg_value: An integer, or a tuple of two integers.
        channels_last: Whether or not the last axis is the channel dimension?

    Returns:
        (int, int, int, int): The validated two integers, plus two `1` as
            the strides for batch and channels dimensions.
    """
    value = validate_conv2d_size_tuple(arg_name, arg_value)
    if channels_last:
        value = (1,) + value + (1,)
    else:
        value = (1, 1) + value
    return value


def get_deconv_output_length(input_length, kernel_size, strides, padding):
    """
    Get the output length of deconvolution at a specific dimension.

    Args:
        input_length: Input tensor length.
        kernel_size: The size of the kernel.
        strides: The stride of convolution.
        padding: One of {"same", "valid"}, case in-sensitive

    Returns:
        int: The output length of deconvolution.
    """
    padding = validate_enum_arg(
        'padding', str(padding).upper(), ['SAME', 'VALID'])
    output_length = input_length * strides
    if padding == 'VALID':
        output_length += max(kernel_size - strides, 0)
    return output_length
