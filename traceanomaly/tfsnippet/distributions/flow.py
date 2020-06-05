import tensorflow as tf

from tfsnippet.stochastic import StochasticTensor
from tfsnippet.layers import BaseFlow
from tfsnippet.utils import validate_group_ndims_arg
from .base import Distribution
from .wrapper import as_distribution

__all__ = ['FlowDistribution']


class FlowDistribution(Distribution):
    """
    Transform a :class:`Distribution` by a :class:`BaseFlow`, as a new
    distribution.
    """

    def __init__(self, distribution, flow):
        """
        Construct a new :class:`FlowDistribution` from the given `distribution`.

        Args:
            distribution (Distribution): The distribution to transform from.
                It must be continuous,
            flow (BaseFlow): A normalizing flow to transform the `distribution`.
        """
        if not isinstance(flow, BaseFlow):
            raise TypeError('`flow` is not an instance of `BaseFlow`: {!r}'.
                            format(flow))
        distribution = as_distribution(distribution)
        if not distribution.is_continuous:
            raise ValueError('{!r} cannot be transformed by a flow, because '
                             'it is not continuous.'.format(distribution))
        if not distribution.dtype.is_floating:
            raise ValueError('{!r} cannot be transformed by a flow, because '
                             'its data type is not float.'.format(distribution))

        self._flow = flow
        self._distribution = distribution

    @property
    def flow(self):
        """
        Get the transformation flow.

        Returns:
            BaseFlow: The transformation flow.
        """
        return self._flow

    @property
    def distribution(self):
        """
        Get the base distribution.

        Returns:
            Distribution: The base distribution to transform from.
        """
        return self._distribution

    @property
    def dtype(self):
        return self._distribution.dtype

    @property
    def is_continuous(self):
        return self._distribution.is_continuous

    @property
    def is_reparameterized(self):
        return self._distribution.is_reparameterized

    @property
    def value_shape(self):
        return self._distribution.value_shape

    def get_value_shape(self):
        return self._distribution.get_value_shape()

    @property
    def batch_shape(self):
        return self._distribution.batch_shape

    def get_batch_shape(self):
        return self._distribution.get_batch_shape()

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        group_ndims = validate_group_ndims_arg(group_ndims)
        if not compute_density and compute_density is not None:
            raise RuntimeError('`FlowDistribution` requires `compute_prob` '
                               'not to be False.')

        with tf.name_scope(
                name, default_name='FlowDistribution.sample'):
            # sample from the base distribution
            x = self._distribution.sample(
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
                compute_density=True
            )

            # now do the transformation
            is_reparameterized = x.is_reparameterized
            y, log_det = self._flow.transform(x)  # y, log |dy/dx|
            if not is_reparameterized:
                y = tf.stop_gradient(y)  # important!

            # compose the transformed tensor
            return StochasticTensor(
                distribution=self,
                tensor=y,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
                # log p(y) = log p(x) - log |dy/dx|
                log_prob=x.log_prob() - log_det
            )

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(
                name,
                default_name='FlowDistribution.log_prob',
                values=[given]):
            x, log_det = self._flow.inverse_transform(given)  # x, log |dx/dy|
            log_px = self._distribution.log_prob(x, group_ndims=group_ndims)
            return log_px + log_det  # log p(y) = log p(x) + log |dx/dy|
