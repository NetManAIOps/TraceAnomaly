import tensorflow as tf
import zhusuan.distributions as zd

from tfsnippet.utils import settings
from .wrapper import ZhuSuanDistribution

__all__ = ['OnehotCategorical', 'Concrete', 'ExpConcrete']


class OnehotCategorical(ZhuSuanDistribution):
    """
    One-hot multivariate Categorical distribution.

    A batch of samples is an N-D Tensor with `dtype` values in range
    ``[0, n_categories)``.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.OnehotCategorical`
    """

    def __init__(self, logits, dtype=None):
        """
        Construct the :class:`OnehotCategorical`.

        Args:
            logits: An N-D (N >= 1) `float` Tensor of shape
                ``(..., n_categories)``.  Each slice `[i, j,..., k, :]`
                represents the un-normalized log-probabilities for all
                categories.  :math:`\\mathrm{logits} \\propto \\log p`
            dtype: The value type of samples from the distribution.
                (default ``tf.int32``)
        """
        if dtype is None:
            dtype = tf.int32
        super(OnehotCategorical, self).__init__(
            zd.OnehotCategorical(logits=logits, dtype=dtype))

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._distribution.logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._distribution.n_categories


class Concrete(ZhuSuanDistribution):
    """
    The class of Concrete (or Gumbel-Softmax) distribution from
    (Maddison, 2016; Jang, 2016), served as the
    continuous relaxation of the :class:`~OnehotCategorical`.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.Concrete`
    """

    def __init__(self, temperature, logits, is_reparameterized=True,
                 check_numerics=None):
        """
        Construct the :class:`ExpConcrete`.

        Args:
            temperature: A 0-D `float` Tensor. The temperature of the relaxed
                distribution. The temperature should be positive.
            logits: An N-D (N >= 1) `float` Tensor of shape
                ``(..., n_categories)``.  Each slice `[i, j,..., k, :]`
                represents the un-normalized log probabilities for all
                categories.  :math:`\\mathrm{logits} \\propto \\log p`
            is_reparameterized (bool): Whether or not the gradients can
                be propagated through parameters? (default :obj:`True`)
            check_numerics (bool): Whether or not to check numerical issues.
                Default to ``tfsnippet.settings.check_numerics``.
        """
        if check_numerics is None:
            check_numerics = settings.check_numerics
        super(Concrete, self).__init__(
            zd.Concrete(temperature=temperature,
                        logits=logits,
                        is_reparameterized=is_reparameterized,
                        check_numerics=check_numerics))

    @property
    def temperature(self):
        """The temperature of this concrete distribution."""
        return self._distribution.temperature

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._distribution.logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._distribution.n_categories


class ExpConcrete(ZhuSuanDistribution):
    """
    The class of ExpConcrete distribution from (Maddison, 2016), transformed
    from :class:`~Concrete` by taking logarithm.

    See Also:
        :class:`tfsnippet.distributions.Distribution`,
        :class:`zhusuan.distributions.Distribution`,
        :class:`zhusuan.distributions.ExpConcrete`
    """

    def __init__(self, temperature, logits, is_reparameterized=True,
                 check_numerics=None):
        """
        Construct the :class:`ExpConcrete`.

        Args:
            temperature: A 0-D `float` Tensor. The temperature of the relaxed
                distribution. The temperature should be positive.
            logits: An N-D (N >= 1) `float` Tensor of shape
                ``(..., n_categories)``.  Each slice `[i, j,..., k, :]`
                represents the un-normalized log probabilities for all
                categories.  :math:`\\mathrm{logits} \\propto \\log p`
            is_reparameterized (bool): Whether or not the gradients can
                be propagated through parameters? (default :obj:`True`)
            check_numerics (bool): Whether or not to check numerical issues.
                Default to ``tfsnippet.settings.check_numerics``.
        """
        if check_numerics is None:
            check_numerics = settings.check_numerics
        super(ExpConcrete, self).__init__(
            zd.ExpConcrete(temperature=temperature,
                           logits=logits,
                           is_reparameterized=is_reparameterized,
                           check_numerics=check_numerics))

    @property
    def temperature(self):
        """The temperature of this concrete distribution."""
        return self._distribution.temperature

    @property
    def logits(self):
        """The un-normalized log probabilities."""
        return self._distribution.logits

    @property
    def n_categories(self):
        """The number of categories in the distribution."""
        return self._distribution.n_categories
