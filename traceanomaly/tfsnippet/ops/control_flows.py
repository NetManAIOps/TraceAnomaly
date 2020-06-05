import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc, is_tensor_object

__all__ = ['smart_cond']


@add_name_arg_doc
def smart_cond(cond, true_fn, false_fn, name=None):
    """
    Execute `true_fn` or `false_fn` according to `cond`.

    Args:
        cond (bool or tf.Tensor): A bool constant or a tensor.
        true_fn (() -> tf.Tensor): The function of the true branch.
        false_fn (() -> tf.Tensor): The function of the false branch.

    Returns:
        tf.Tensor: The output tensor.
    """
    if is_tensor_object(cond):
        return tf.cond(cond, true_fn, false_fn, name=name)
    else:
        if cond:
            return true_fn()
        else:
            return false_fn()
