import tensorflow as tf

from tfsnippet.bayes import BayesianNet
from tfsnippet.ops import add_n_broadcast
from .inference import VariationalInference

__all__ = ['VariationalChain']


class VariationalChain(object):
    """
    Chain of the variational and model nets for variational inference.

    In the context of variational inference, it is a common usage for chaining
    the variational net and the model net, by feeding the samples of latent
    variables from the variational net as the observations of the model net.
    :class:`VariationalChain` holds the :class:`BayesianNet` instances of
    the variational and the model nets, and the :class:`VariationalInference`
    object for this chain.

    See Also:
        :meth:`tfsnippet.bayes.BayesianNet.variational_chain`
    """

    def __init__(self, variational, model, log_joint=None, latent_names=None,
                 latent_axis=None):
        """
        Construct the :class:`VariationalChain`.

        Args:
            variational (BayesianNet): The variational net.
            model (BayesianNet): The model net.
            log_joint (tf.Tensor): The log-joint of the model net. If
                :obj:`None`, the log-densities of all variables
                within `model` net will be summed up as the log-joint.
                (default :obj:`None`)
            latent_names (Iterable[str]): Names of the latent variables in
                variational inference. If :obj:`None`, all of the variables
                within `variational` net will be collected. (default
                :obj:`None`)
            latent_axis: The axis or axes to be considered as the sampling
                dimensions of latent variables.  The specified axes will be
                summed up in the variational lower-bounds or training
                objectives. (default :obj:`None`)
        """
        if latent_names is None:
            latent_names = tuple(variational)
        else:
            latent_names = tuple(latent_names)

        with tf.name_scope('VariationalChain'):
            if log_joint is None:
                with tf.name_scope('model_log_joint'):
                    log_joint = add_n_broadcast(
                        model.local_log_probs(iter(model)))
            with tf.name_scope('latent_log_probs'):
                latent_log_probs = variational.local_log_probs(latent_names)

        self._variational = variational
        self._model = model
        self._log_joint = log_joint
        self._latent_names = latent_names
        self._latent_axis = latent_axis
        self._vi = VariationalInference(
            log_joint=self.log_joint,
            latent_log_probs=latent_log_probs,
            axis=latent_axis
        )

    @property
    def variational(self):
        """
        Get the variational net.

        Returns:
            BayesianNet: The variational net.
        """
        return self._variational

    @property
    def model(self):
        """
        Get the model net.

        Returns:
            BayesianNet: The model net.
        """
        return self._model

    @property
    def log_joint(self):
        """
        Get the log-joint of the model.

        Returns:
            tf.Tensor: The log-joint of the model.
        """
        return self._log_joint

    @property
    def latent_names(self):
        """
        Get the names of the latent variables for variational inference.

        Returns:
            tuple[str]: The names of the latent variables.
        """
        return self._latent_names

    @property
    def latent_axis(self):
        """Get the axes of sampling dimensions of latent variables."""
        return self._latent_axis

    @property
    def vi(self):
        """
        Get the variational inference object.

        Returns:
            VariationalInference: The variational inference object.
        """
        return self._vi
