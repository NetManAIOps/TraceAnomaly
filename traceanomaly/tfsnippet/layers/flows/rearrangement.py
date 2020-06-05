import numpy as np
import tensorflow as tf

from tfsnippet.utils import (add_name_and_scope_arg_doc, VarScopeRandomState,
                             get_static_shape, get_shape, model_variable)
from .base import FeatureMappingFlow
from .utils import ZeroLogDet

__all__ = ['FeatureShufflingFlow']


class FeatureShufflingFlow(FeatureMappingFlow):
    """
    An invertible flow which shuffles the order of input features.

    This type of flow is proposed in (Kingma & Dhariwal, 2018), as a possible
    replacement to the alternating pattern of coupling layers proposed in
    (Dinh et al., 2016).  Although the experiments have shown that this flow
    is inferior to learnt feature mappings (e.g., :class:`InvertibleDense`
    and :class:`InvertibleConv2d`), it is faster than learnt mappings, and
    is still superior to the vanilla alternating pattern.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, axis=-1, value_ndims=1, random_state=None,
                 name=None, scope=None):
        """
        Construct a new :class:`FeatureShufflingFlow`.

        Args:
            axis (int): The feature axis, to apply the transformation.
            value_ndims (int): Number of dimensions to be considered as the
                value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
            random_state (np.random.RandomState): Use this random state,
                instead of constructing a :class:`VarScopeRandomState`.
        """
        super(FeatureShufflingFlow, self).__init__(
            axis=int(axis), value_ndims=value_ndims, name=name, scope=scope)

        self._value_ndims = int(value_ndims)
        if random_state is None:
            random_state = VarScopeRandomState(self.variable_scope)
        self._random_state = random_state

        # the permutation variables
        self._permutation = None
        self._inv_permutation = None

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        n_features = self._n_features = get_static_shape(input)[self.axis]
        permutation = np.arange(n_features, dtype=np.int32)
        self._random_state.shuffle(permutation)

        self._permutation = model_variable(
            'permutation', dtype=tf.int32, initializer=permutation,
            trainable=False
        )
        self._inv_permutation = tf.invert_permutation(self._permutation)

    def _transform_or_inverse_transform(self, x, compute_y, compute_log_det,
                                        permutation):
        assert (0 > self.axis >= -self.value_ndims >= -len(get_static_shape(x)))
        assert (get_static_shape(x)[self.axis] == self._n_features)

        # compute y
        y = None
        if compute_y:
            y = tf.gather(x, permutation, axis=self.axis)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(get_shape(x)[:-self.value_ndims],
                                 x.dtype.base_dtype)

        return y, log_det

    def _transform(self, x, compute_y, compute_log_det):
        return self._transform_or_inverse_transform(
            x, compute_y, compute_log_det, self._permutation)

    def _inverse_transform(self, y, compute_x, compute_log_det):
        return self._transform_or_inverse_transform(
            y, compute_x, compute_log_det, self._inv_permutation)
