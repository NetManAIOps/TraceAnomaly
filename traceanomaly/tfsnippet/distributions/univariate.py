import tensorflow as tf
import zhusuan.distributions as zd

from tfsnippet.utils import settings
from .wrapper import ZhuSuanDistribution

__all__ = ['Normal', 'Bernoulli', 'Categorical', 'Discrete', 'Uniform']


class Normal(ZhuSuanDistribution):
    """
    Univariate Normal distribution.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Normal`
    """

    def __init__(self, mean, std=None, logstd=None, is_reparameterized=True,
                 check_numerics=None):
        """
        Construct the :class:`Normal`.

        Args:
            mean: A `float` tensor, the mean of the Normal distribution.
                Should be broadcastable against `std` / `logstd`.
            std: A `float` tensor, the standard deviation of the Normal
                distribution.  Should be positive, and broadcastable against
                `mean`.  One and only one of `std` or `logstd` should be
                specified.
            logstd: A `float` tensor, the log standard deviation of the Normal
                distribution.  Should be broadcastable against `mean`.
            is_reparameterized (bool): Whether or not the gradients can
                be propagated through parameters? (default :obj:`True`)
            check_numerics (bool): Whether or not to check numerical issues.
                Default to ``tfsnippet.settings.check_numerics``.
        """
        if check_numerics is None:
            check_numerics = settings.check_numerics
        super(Normal, self).__init__(zd.Normal(
            mean=mean,
            std=std,
            logstd=logstd,
            is_reparameterized=is_reparameterized,
            check_numerics=check_numerics,
        ))

    @property
    def mean(self):
        """Get the mean of the Normal distribution."""
        return self._distribution.mean

    @property
    def logstd(self):
        """Get the log standard deviation of the Normal distribution."""
        return self._distribution.logstd

    @property
    def std(self):
        """Get the standard deviation of the Normal distribution."""
        return self._distribution.std


class Bernoulli(ZhuSuanDistribution):
    """
    Univariate Bernoulli distribution.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Bernoulli`
    """

    def __init__(self, logits, dtype=tf.int32):
        """
        Construct the :class:`Bernoulli`.

        Args:
            logits: A `float` tensor, log-odds of probabilities of being 1.
                :math:`\\mathrm{logits} = \\log \\frac{p}{1 - p}`
            dtype: The value type of samples from the distribution.
                (default ``tf.int32``)
        """
        super(Bernoulli, self).__init__(
            zd.Bernoulli(logits=logits, dtype=dtype))

    @property
    def logits(self):
        """The log-odds of probabilities of being 1."""
        return self._distribution.logits


class Categorical(ZhuSuanDistribution):
    """
    Univariate Categorical distribution.

    A batch of samples is an (N-1)-D Tensor with `dtype` values in range
    ``[0, n_categories)``.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Categorical`
    """

    def __init__(self, logits, dtype=None):
        """
        Construct the :class:`Categorical`.

        Args:
            logits: An N-D (N >= 1) `float` Tensor of shape
                ``(..., n_categories)``.  Each slice `[i, j,..., k, :]`
                represents the un-normalized log probabilities for all
                categories.  :math:`\\mathrm{logits} \\propto \\log p`
            dtype: The value type of samples from the distribution.
                (default ``tf.int32``)
        """
        if dtype is None:
            dtype = tf.int32
        super(Categorical, self).__init__(
            zd.Categorical(logits=logits, dtype=dtype))

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._distribution.logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._distribution.n_categories


Discrete = Categorical


class Uniform(ZhuSuanDistribution):
    """
    Univariate Uniform distribution.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Uniform`
    """

    def __init__(self, minval=0., maxval=1., is_reparameterized=True,
                 check_numerics=None):
        """
        Construct the :class:`Uniform`.

        Args:
            minval: A `float` Tensor. The lower bound on the range of the
                uniform distribution. Should be broadcastable to match `maxval`.
            maxval: A `float` Tensor. The upper bound on the range of the
                uniform distribution. Should be element-wise bigger than
                `minval`.
            is_reparameterized (bool): Whether or not the gradients can
                be propagated through parameters? (default :obj:`True`)
            check_numerics (bool): Whether or not to check numerical issues.
                Default to ``tfsnippet.settings.check_numerics``.
        """
        if check_numerics is None:
            check_numerics = settings.check_numerics
        super(Uniform, self).__init__(
            zd.Uniform(
                minval=minval,
                maxval=maxval,
                is_reparameterized=is_reparameterized,
                check_numerics=check_numerics,
            ))

    @property
    def minval(self):
        """The lower bound on the range of the uniform distribution."""
        return self._distribution.minval

    @property
    def maxval(self):
        """The upper bound on the range of the uniform distribution."""
        return self._distribution.maxval
