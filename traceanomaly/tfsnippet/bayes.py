import warnings
from collections import OrderedDict

import six
import tensorflow as tf
from frozendict import frozendict

from tfsnippet.distributions import (Distribution, FlowDistribution,
                                     as_distribution)
from tfsnippet.layers import BaseFlow
from tfsnippet.stochastic import StochasticTensor
from tfsnippet.utils import get_default_scope_name

__all__ = ['BayesianNet']


class BayesianNet(object):
    """
    Bayesian networks.

    :class:`BayesianNet` is a class which helps to construct Bayesian
    networks and to derive the variational lower-bounds.
    It is inspired by :class:`zhusuan.BayesianNet`.

    Due to the expressive limitations of TensorFlow, it is hard to build
    :class:`BayesianNet` with the concept of `random variables`.
    Instead, we only collect :class:`StochasticTensor` objects, i.e.,
    tensors sampled from the distributions of these random variables.
    Thus :class:`BayesianNet` is actually a collection of (multiple)
    ancestral samples from the random variables.
    Fortunately, we can approximate most interested statistics of the desired
    random variables with these samples, by using Monte Carlo methods.
    For example, obtaining the expectation of a random variable by averaging
    over multiple samples from it.
    The :class:`StochasticTensor` objects are called `stochastic nodes`
    within the context of :class:`BayesianNet`.

    To build a Bayesian network, first obtain a :class:`BayesianNet`:

    .. code-block:: python

        net = tfsnippet.bayes.BayesianNet()

    Then add stochastic nodes into the network:

    A Bayesian Linear Regression example, as of :class:`zhusuan.BayesianNet`:

    .. math::

        w \\sim N(0, \\alpha^2 I)

        y \\sim N(w^Tx, \\beta^2)

    .. code-block:: python

        from tfsnippet.bayes import BayesianNet()
        from tfsnippet.distributions import Normal

        def bayesian_linear_regression(x, alpha, beta, observed=None):
            net = BayesianNet(observed)
            w = net.add('w', Normal(mean=0., logstd=tf.log(alpha)))
            y_mean = tf.reduce_sum(tf.expand_dims(w, 0) * x, 1)
            y = net.add('y', Normal(mean=y_mean, logstd=tf.log(beta)))
            return net

    To observe any stochastic nodes in the network, pass a dictionary mapping
    of ``(name, Tensor)`` as `observed` when constructing :class:`BayesianNet`.
    For example:

    .. code-block:: python

        model = bayesian_linear_regression(..., observed={'w': w_obs})

    After construction, :class:`BayesianNet` supports queries on the network.

    .. code-block:: python

        # get samples of random variable y following generative process
        # in the network
        model.output('y')

        # because w is observed in this case, its observed value will be
        # returned
        model.output('w')

        # also multiple outputs can be fetched together
        model.outputs(['y', 'w'])

        # get local log probability values of w and y, which returns
        # log p(w) and log p(y|w, x)
        model.local_log_probs(['w', 'y'])

        # query many quantities at the same time
        model.query(['w', 'y'])

    See Also:
        :class:`zhusuan.BayesianNet`
    """

    def __init__(self, observed=None):
        """
        Construct the :class:`BayesianNet`.

        Args:
            observed: Dict of ``(str, tf.Tensor)``, the names of stochastic
                nodes and their observations.
        """
        super(BayesianNet, self).__init__()
        self._observed = frozendict([
            (name, tf.convert_to_tensor(tensor))
            for name, tensor in (six.iteritems(observed) if observed else ())
        ])
        self._stochastic_tensors = OrderedDict()

    @property
    def observed(self):
        """
        Get the read-only dict of observations.

        Returns:
            collections.Mapping[str, tf.Tensor]: The read-only dict of
                observations.
        """
        return self._observed

    def _check_names_exist(self, names):
        names = tuple(names)
        for name in names:
            if not isinstance(name, six.string_types):
                raise TypeError('`names` is not a list of str')
            if name not in self._stochastic_tensors:
                raise KeyError('StochasticTensor with name {!r} does not exist'.
                               format(name))
        return names

    def add(self, name, distribution, n_samples=None, group_ndims=0,
            is_reparameterized=None, flow=None):
        """
        Add a stochastic node to the network.

        A :class:`StochasticTensor` will be created for this node.
        If `name` exists in `observed` dict, its value will be used as the
        observation of this node.  Otherwise samples will be taken from
        `distribution`.

        Args:
            name (str): Name of the stochastic node.
            distribution (Distribution or zhusuan.distributions.Distribution):
                Distribution where the samples should be taken from.
            n_samples (int or tf.Tensor): Number of samples to take.
                If specified, `n_samples` will be taken, with a dedicated
                sampling dimension ``[n_samples]`` at the front.
                If not specified, just one sample will be taken, without the
                dedicated dimension.
            group_ndims (int or tf.Tensor): Number of dimensions at the end of
                ``[n_samples] + batch_shape`` to be considered as events group.
                (default 0)
            is_reparameterized: Whether or not the re-parameterization trick
                should be applied? (default :obj:`None`, following the setting
                of `distribution`)
            flow (BaseFlow): If specified, transform `distribution` by `flow`.

        Returns:
            StochasticTensor: The sampled stochastic tensor.

        Raises:
            TypeError: If `name` is not a str, or `distribution` is a
                :class:`TransformedDistribution`.
            KeyError: If :class:`StochasticTensor` with `name` already exists.
            ValueError: If `transform` cannot be applied.

        See Also:
            :meth:`tfsnippet.distributions.Distribution.sample`
        """
        if not isinstance(name, six.string_types):
            raise TypeError('`name` must be a str')
        if name in self._stochastic_tensors:
            raise KeyError('StochasticTensor with name {!r} already exists in '
                           'the BayesianNet.  Names must be unique.'.
                           format(name))
        if flow is not None and name in self._observed and \
                not flow.explicitly_invertible:
            raise TypeError('The observed variable {!r} expects `flow` to be '
                            'explicitly invertible, but it is not: {!r}.'.
                            format(name, flow))

        distribution = as_distribution(distribution)
        if flow is not None:
            distribution = FlowDistribution(distribution, flow)

        if name in self._observed:
            t = StochasticTensor(
                distribution=distribution,
                tensor=self._observed[name],
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
        else:
            t = distribution.sample(
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
            assert(isinstance(t, StochasticTensor))

        self._stochastic_tensors[name] = t
        return t

    def get(self, name):
        """
        Get :class:`StochasticTensor` of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            StochasticTensor: :class:`StochasticTensor` of the queried node,
                or :obj:`None` if no node exists with `name`.
        """
        return self._stochastic_tensors.get(name)

    def __getitem__(self, name):
        """
        Get :class:`StochasticTensor` of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            StochasticTensor: :class:`StochasticTensor` of the queried node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        self._check_names_exist((name,))
        return self._stochastic_tensors[name]

    def __contains__(self, name):
        """Test whether or not a stochastic node with `name` exists."""
        return name in self._stochastic_tensors

    def __iter__(self):
        """Get an iterator of the stochastic node names."""
        return iter(self._stochastic_tensors)

    def outputs(self, names):
        """
        Get the outputs of stochastic nodes.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[tf.Tensor]: Outputs of the queried stochastic nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        return [self._stochastic_tensors[n].tensor for n in names]

    def output(self, name):
        """
        Get the output of a stochastic node.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            tf.Tensor: Output of the queried stochastic node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        return self.outputs((name,))[0]

    def local_log_probs(self, names):
        """
        Get the log-densities of stochastic nodes.

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[tf.Tensor]: Log-densities of the queried stochastic nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        ret = []
        for name in names:
            ns = '{}.log_prob'.format(get_default_scope_name(name))
            ret.append(self._stochastic_tensors[name].log_prob(name=ns))
        return ret

    def local_log_prob(self, name):
        """
        Get the log-density of a stochastic node.

        Args:
            name (str): Name of the queried stochastic node.

        Returns:
            tf.Tensor: Log-density of the queried stochastic node.

        Raises:
            KeyError: If non-exist name is queried.
        """
        return self.local_log_probs((name,))[0]

    def query(self, names):
        """
        Get the outputs and log-densities of stochastic node(s).

        Args:
            names (Iterable[str]): Names of the queried stochastic nodes.

        Returns:
            list[(tf.Tensor, tf.Tensor)]: Tuples of `(output, log-prob)` of the
                queried stochastic nodes.

        Raises:
            KeyError: If non-exist name is queried.
        """
        names = self._check_names_exist(names)
        return list(zip(self.outputs(names), self.local_log_probs(names)))

    def variational_chain(self, model_builder, latent_names=None,
                          latent_axis=None, observed=None, **kwargs):
        """
        Treat this :class:`BayesianNet` as variational, and build the model
        net chained after this variational net.

        Args:
            model_builder: Function which receives the `observed` dict, and
                produce the model :class:`BayesianNet` or a tuple of the model
                :class:`BayesianNet` and the log-joint of the model.
            latent_names (Iterable[str]): Names of the nodes to be considered
                as latent variables in this :class:`BayesianNet`.  All these
                variables will be fed into `model_builder` as observed
                variables, overriding the observations in `observed`.
                (default all the variables in this :class:`BayesianNet`)
            latent_axis: The axis or axes to be considered as the sampling
                dimensions of latent variables.  The specified axes will be
                summed up in the variational lower-bounds or training
                objectives. (default :obj:`None`)
            observed: Dict of ``(name, observation)`` fed into `model_builder`.
                (default :obj:`None`)
            \\**kwargs: Additional named arguments passed to `model_builder`.

        Returns:
            tfsnippet.variational.VariationalChain: The object that holds this
                :class:`BayesianNet` as the `variational` net, the constructed
                `model` net, and the
                :class:`~tfsnippet.variational.VariationalInference` object
                for obtaining the variational lower-bounds and training
                objectives.

        See Also:
            :class:`tfsnippet.variational.VariationalChain`
        """
        from tfsnippet.variational.chain import VariationalChain

        # build the observed dict: observed + latent samples
        merged_obs = {}
        # add the user-provided observed dict
        if observed:
            merged_obs.update(observed)
        # add the latent samples
        if latent_names is None:
            latent_names = tuple(self)
        else:
            latent_names = tuple(latent_names)
        merged_obs.update({
            n: t
            for n, t in zip(latent_names, self.outputs(latent_names))
        })

        for n in self:
            if n not in latent_names:  # pragma: no cover
                warnings.warn('The stochastic tensor `{}` in {!r} is not fed '
                              'into `model_builder` as observed variable when '
                              'building the variational chain.'.
                              format(n, self))

        # build the model and its log-joint
        model_and_log_joint = model_builder(merged_obs, **kwargs)
        if isinstance(model_and_log_joint, tuple):
            model, log_joint = model_and_log_joint
        else:
            model, log_joint = model_and_log_joint, None

        # build the VariationalModelChain
        return VariationalChain(
            variational=self,
            model=model,
            log_joint=log_joint,
            latent_names=latent_names,
            latent_axis=latent_axis,
        )

    chain = variational_chain
    """Alias for :meth:`variational_chain`."""
