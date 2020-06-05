import tensorflow as tf

from tfsnippet.ops import log_mean_exp
from .utils import _require_multi_samples

__all__ = [
    'sgvb_estimator', 'iwae_estimator',
]


def sgvb_estimator(values, axis=None, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big]`,
    by SGVB (Kingma, D.P. and Welling, M., 2013) algorithm.

    .. math::

        \\nabla \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big] = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big] = \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[\\nabla f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big]

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): TensorFlow name scope of the graph nodes.
            (default "sgvb_estimator")

    Returns:
        tf.Tensor: The surrogate for optimizing the target function
            with SGVB gradient estimator.
    """
    values = tf.convert_to_tensor(values)
    with tf.name_scope(name, default_name='sgvb_estimator', values=[values]):
        estimator = values
        if axis is not None:
            estimator = tf.reduce_mean(estimator, axis=axis, keepdims=keepdims)
        return estimator


def iwae_estimator(log_values, axis, keepdims=False, name=None):
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]`,
    by IWAE (Burda, Y., Grosse, R. and Salakhutdinov, R., 2015) algorithm.

    .. math::

        \\begin{aligned}
            &\\nabla\\,\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]
                = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\nabla \\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg] = \\\\
                & \\quad \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\nabla \\frac{1}{K} \\sum_{k=1}^K w_k}{\\frac{1}{K} \\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\sum_{k=1}^K w_k \\nabla \\log w_k}{\\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\sum_{k=1}^K \\widetilde{w}_k \\nabla \\log w_k\\Bigg]
        \\end{aligned}

    Args:
        log_values: Log values of the target function given `z` and `x`, i.e.,
            :math:`\\log f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): TensorFlow name scope of the graph nodes.
            (default "iwae_estimator")

    Returns:
        tf.Tensor: The surrogate for optimizing the target function
            with IWAE gradient estimator.
    """
    _require_multi_samples(axis, 'iwae estimator')
    log_values = tf.convert_to_tensor(log_values)
    with tf.name_scope(name, default_name='iwae_estimator',
                       values=[log_values]):
        estimator = log_mean_exp(log_values, axis=axis, keepdims=keepdims)
        return estimator
