import tensorflow as tf

from tfsnippet.ops import log_mean_exp
from .utils import _require_multi_samples

__all__ = ['elbo_objective', 'monte_carlo_objective']


def elbo_objective(log_joint, latent_log_prob, axis=None, keepdims=False,
                   name=None):
    """
    Derive the ELBO objective.

    .. math::

        \\mathbb{E}_{\\mathbf{z} \\sim q_{\\phi}(\\mathbf{z}|\\mathbf{x})}\\big[
             \\log p_{\\theta}(\\mathbf{x},\\mathbf{z}) - \\log q_{\\phi}(\\mathbf{z}|\\mathbf{x})
        \\big]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_prob: :math:`q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): TensorFlow name scope of the graph nodes.
            (default "elbo_objective")

    Returns:
        tf.Tensor: The ELBO objective.  Not applicable for training.
    """
    log_joint = tf.convert_to_tensor(log_joint)
    latent_log_prob = tf.convert_to_tensor(latent_log_prob)
    with tf.name_scope(name,
                       default_name='elbo_objective',
                       values=[log_joint, latent_log_prob]):
        objective = log_joint - latent_log_prob
        if axis is not None:
            objective = tf.reduce_mean(objective, axis=axis, keepdims=keepdims)
        return objective


def monte_carlo_objective(log_joint, latent_log_prob, axis=None,
                          keepdims=False, name=None):
    """
    Derive the Monte-Carlo objective.

    .. math::

        \\mathcal{L}_{K}(\\mathbf{x};\\theta,\\phi) =
            \\mathbb{E}_{\\mathbf{z}^{(1:K)} \\sim q_{\\phi}(\\mathbf{z}|\\mathbf{x})}\\Bigg[
                \\log \\frac{1}{K} \\sum_{k=1}^K {
                    \\frac{p_{\\theta}(\\mathbf{x},\\mathbf{z}^{(k)})}
                         {q_{\\phi}(\\mathbf{z}^{(k)}|\\mathbf{x})}
                }
            \\Bigg]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_prob: :math:`q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the averaged dimensions?  (default :obj:`False`)
        name (str): TensorFlow name scope of the graph nodes.
            (default "monte_carlo_objective")

    Returns:
        tf.Tensor: The Monte Carlo objective.  Not applicable for training.
    """
    _require_multi_samples(axis, 'monte carlo objective')
    log_joint = tf.convert_to_tensor(log_joint)
    latent_log_prob = tf.convert_to_tensor(latent_log_prob)
    with tf.name_scope(name,
                       default_name='monte_carlo_objective',
                       values=[log_joint, latent_log_prob]):
        likelihood = log_joint - latent_log_prob
        objective = log_mean_exp(likelihood, axis=axis, keepdims=keepdims)
        return objective
