import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc, validate_int_tuple_arg

__all__ = ['add_n_broadcast', 'log_mean_exp', 'log_sum_exp']


@add_name_arg_doc
def add_n_broadcast(tensors, name=None):
    """
    Add zero or many tensors with broadcasting.

    Args:
        tensors (Iterable[Tensor]): A list of tensors to be summed.

    Returns:
        tf.Tensor: The summed tensor.
    """
    tensors = [tf.convert_to_tensor(t) for t in tensors]
    if not tensors:
        raise ValueError('`tensors` must not be empty.')
    with tf.name_scope(name, default_name='add_n_broadcast', values=tensors):
        ret = tensors[0]
        for t in tensors[1:]:
            ret += t
        return ret


@add_name_arg_doc
def log_sum_exp(x, axis=None, keepdims=False, name=None):
    """
    Compute :math:`\\log \\sum_{k=1}^K \\exp(x_k)`.

    .. math::

        \\begin{align*}
            \\log \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max})
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}

    Args:
        x (Tensor): The input `x`.
        axis (int or tuple[int]): The dimension to take summation.
            Default :obj:`None`, all dimensions.
        keepdims (bool): Whether or not to keep the summed dimensions?
            (default :obj:`False`)

    Returns:
        tf.Tensor: The computed value.
    """
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = tf.convert_to_tensor(x)
    with tf.name_scope(name, default_name='log_sum_exp', values=[x]):
        x_max_keepdims = tf.reduce_max(x, axis=axis, keepdims=True)
        if not keepdims:
            x_max = tf.squeeze(x_max_keepdims, axis=axis)
        else:
            x_max = x_max_keepdims
        sum_exp = tf.reduce_sum(tf.exp(x - x_max_keepdims), axis=axis,
                                keepdims=keepdims)
        return x_max + tf.log(sum_exp)


@add_name_arg_doc
def log_mean_exp(x, axis=None, keepdims=False, name=None):
    """
    Compute :math:`\\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)`.

    .. math::

        \\begin{align*}
            \\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max}) \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}

    Args:
        x (Tensor): The input `x`.
        axis (int or tuple[int]): The dimension to take average.
            Default :obj:`None`, all dimensions.
        keepdims (bool): Whether or not to keep the summed dimensions?
            (default :obj:`False`)

    Returns:
        tf.Tensor: The computed value.
    """
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = tf.convert_to_tensor(x)
    with tf.name_scope(name, default_name='log_mean_exp', values=[x]):
        x = tf.convert_to_tensor(x)
        x_max_keepdims = tf.reduce_max(x, axis=axis, keepdims=True)
        if not keepdims:
            x_max = tf.squeeze(x_max_keepdims, axis=axis)
        else:
            x_max = x_max_keepdims
        mean_exp = tf.reduce_mean(tf.exp(x - x_max_keepdims), axis=axis,
                                  keepdims=keepdims)
        return x_max + tf.log(mean_exp)
