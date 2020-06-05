import contextlib

import tensorflow as tf
import zhusuan

from tfsnippet.utils import get_default_scope_name
from .base import Distribution
from .utils import reduce_group_ndims

__all__ = ['as_distribution']


def as_distribution(distribution):
    """
    Convert a supported type of `distribution` into :class:`Distribution` type.

    Args:
        distribution: A supported distribution instance. Supported types are:
            1. :class:`Distribution`,
            2. :class:`zhusuan.distributions.Distribution`.

    Returns:
        Distribution: The wrapped distribution of :class:`Distribution` type.

    Raises:
        TypeError: If the specified `distribution` cannot be converted.
    """
    if isinstance(distribution, Distribution):
        return distribution
    if isinstance(distribution, zhusuan.distributions.Distribution):
        return ZhuSuanDistribution(distribution)
    raise TypeError('Type `{}` cannot be casted into `tfsnippet.distributions.'
                    'Distribution`'.format(distribution.__class__.__name__))


class ZhuSuanDistribution(Distribution):
    """
    Wrapping a :class:`zhusuan.distributions.Distribution` into
    :class:`~tfsnippet.distributions.Distribution`.

    .. _`ZhuSuan`: https://github.com/thu-ml/zhusuan
    """

    def __init__(self, distribution):
        """
        Construct the :class:`ZhuSuanDistribution`.

        Args:
            distribution (zhusuan.distributions.Distribution): The distribution
                from ZhuSuan. `group_ndims` attribute of `distribution` would
                be totally ignored.  Thread-safety is not guaranteed for also
                using `distribution` outside of :class:`ZhuSuanDistribution`,
                since :class:`ZhuSuanDistribution` may temporarily modify
                internal states of `distribution`.
        """
        if not isinstance(distribution, zhusuan.distributions.Distribution):
            raise TypeError('`distribution` is not an instance of `zhusuan.'
                            'distributions.Distribution`')
        super(ZhuSuanDistribution, self).__init__()
        self._distribution = distribution

    def __repr__(self):
        return 'Distribution({!r})'.format(self._distribution)

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

    def sample(self, n_samples=None, is_reparameterized=None, group_ndims=0,
               compute_density=None, name=None):
        from tfsnippet.stochastic import StochasticTensor

        if is_reparameterized and not self.is_reparameterized:
            raise RuntimeError('Distribution is not re-parameterized')
        elif is_reparameterized is False and self.is_reparameterized:
            @contextlib.contextmanager
            def set_is_reparameterized():
                try:
                    self._distribution._is_reparameterized = False
                    yield False
                finally:
                    self._distribution._is_reparameterized = True
        else:
            @contextlib.contextmanager
            def set_is_reparameterized():
                yield self.is_reparameterized

        with tf.name_scope(name=name, default_name='sample'):
            with set_is_reparameterized() as is_reparameterized:
                samples = self._distribution.sample(n_samples=n_samples)
                t = StochasticTensor(
                    distribution=self,
                    tensor=samples,
                    n_samples=n_samples,
                    group_ndims=group_ndims,
                    is_reparameterized=is_reparameterized,
                )
                if compute_density:
                    with tf.name_scope('compute_prob_and_log_prob'):
                        log_p = t.log_prob()
                        t._self_prob = tf.exp(log_p)
                return t

    def log_prob(self, given, group_ndims=0, name=None):
        with tf.name_scope(name=name,
                           default_name=get_default_scope_name('log_prob', self)):
            given = self._distribution._check_input_shape(given)
            log_prob = self._distribution._log_prob(given)
            return reduce_group_ndims(tf.reduce_sum, log_prob, group_ndims)

    def prob(self, given, group_ndims=0, name=None):
        default_name = '{}.prob'.format(
            self._distribution.__class__.__name__)
        with tf.name_scope(name, default_name=default_name):
            return tf.exp(self.log_prob(given, group_ndims=group_ndims))
