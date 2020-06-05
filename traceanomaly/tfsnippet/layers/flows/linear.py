import numpy as np
import tensorflow as tf

from tfsnippet.utils import (InvertibleMatrix, get_static_shape,
                             get_dimensions_size, is_tensor_object)
from ..convolutional import conv2d
from ..core import dense
from .base import FeatureMappingFlow
from .utils import broadcast_log_det_against_input

__all__ = ['InvertibleDense', 'InvertibleConv2d']


def apply_log_det_factor(log_det, input, axis, value_ndims):
    shape = get_static_shape(input)
    assert(shape is not None)
    assert(len(shape) >= value_ndims)
    assert(value_ndims > 0)
    if axis < 0:
        axis = axis + len(shape)
        assert(axis >= 0)
    reduced_axis = [a for a in range(-value_ndims, 0) if a + len(shape) != axis]

    if reduced_axis:
        shape = get_dimensions_size(input, reduced_axis)
        if is_tensor_object(shape):
            log_det *= tf.cast(tf.reduce_prod(shape), log_det.dtype)
        else:
            log_det *= np.prod(shape)

    return log_det


class InvertibleDense(FeatureMappingFlow):
    """
    Invertible dense layer, modified from the invertible 1x1 2d convolution
    proposed in (Kingma & Dhariwal, 2018).
    """

    def __init__(self,
                 strict_invertible=False,
                 random_state=None,
                 trainable=True,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`InvertibleDense`.

        Args:
            strict_invertible (bool): If :obj:`True`, will derive the kernel
                matrix using a variant of PLU decomposition, to enforce
                invertibility (see :class:`InvertibleMatrix`).
                If :obj:`False`, the matrix will only be initialized to be an
                orthogonal invertible matrix, without further constraint.
                (default :obj:`False`)
            random_state (np.random.RandomState): Use this random state,
                instead of constructing a :class:`VarScopeRandomState`.
            trainable (bool): Whether or not the variables are trainable?
        """
        self._strict_invertible = bool(strict_invertible)
        self._random_state = random_state
        self._trainable = bool(trainable)

        super(InvertibleDense, self).__init__(
            axis=-1,
            value_ndims=1,
            require_batch_dims=True,
            name=name,
            scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        dtype = input.dtype.base_dtype
        n_features = get_static_shape(input)[self.axis]

        self._kernel_matrix = InvertibleMatrix(
            size=n_features, strict=self._strict_invertible, dtype=dtype,
            trainable=self._trainable, random_state=self._random_state,
            scope='kernel'
        )

    def _transform(self, x, compute_y, compute_log_det):
        # compute y
        y = None
        if compute_y:
            n_features = get_static_shape(x)[self.axis]
            y = dense(x, n_features, kernel=self._kernel_matrix.matrix,
                      use_bias=False)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = apply_log_det_factor(
                self._kernel_matrix.log_det, x, self.axis, self.value_ndims)
            log_det = broadcast_log_det_against_input(
                log_det, x, value_ndims=self.value_ndims)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # compute x
        x = None
        if compute_x:
            n_features = get_static_shape(y)[self.axis]
            x = dense(y, n_features, kernel=self._kernel_matrix.inv_matrix,
                      use_bias=False)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = apply_log_det_factor(
                -self._kernel_matrix.log_det, y, self.axis, self.value_ndims)
            log_det = broadcast_log_det_against_input(
                log_det, y, value_ndims=self.value_ndims)

        return x, log_det


class InvertibleConv2d(FeatureMappingFlow):
    """
    Invertible 1x1 2D convolution proposed in (Kingma & Dhariwal, 2018).
    """

    def __init__(self,
                 channels_last=True,
                 strict_invertible=False,
                 random_state=None,
                 trainable=True,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`InvertibleConv2d`.

        Args:
            channels_last (bool): Whether or not the channels axis
                is the last axis in the `input` tensor?
            strict_invertible (bool): If :obj:`True`, will derive the kernel
                matrix using a variant of PLU decomposition, to enforce
                invertibility (see :class:`InvertibleMatrix`).
                If :obj:`False`, the matrix will only be initialized to be an
                orthogonal invertible matrix, without further constraint.
                (default :obj:`False`)
            random_state (np.random.RandomState): Use this random state,
                instead of constructing a :class:`VarScopeRandomState`.
            trainable (bool): Whether or not the variables are trainable?
        """
        self._channels_last = bool(channels_last)
        self._strict_invertible = bool(strict_invertible)
        self._random_state = random_state
        self._trainable = bool(trainable)

        super(InvertibleConv2d, self).__init__(
            axis=-1 if channels_last else -3,
            value_ndims=3,
            require_batch_dims=True,
            name=name,
            scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        dtype = input.dtype.base_dtype
        n_features = get_static_shape(input)[self.axis]

        self._kernel_matrix = InvertibleMatrix(
            size=n_features, strict=self._strict_invertible, dtype=dtype,
            trainable=self._trainable, random_state=self._random_state,
            scope='kernel'
        )

    def _transform(self, x, compute_y, compute_log_det):
        # compute y
        y = None
        if compute_y:
            n_features = get_static_shape(x)[self.axis]
            kernel = tf.reshape(
                self._kernel_matrix.matrix,
                [1, 1] + list(self._kernel_matrix.shape)
            )
            y = conv2d(
                x, n_features, (1, 1), channels_last=self._channels_last,
                kernel=kernel, use_bias=False
            )

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = apply_log_det_factor(
                self._kernel_matrix.log_det, x, self.axis, self.value_ndims)
            log_det = broadcast_log_det_against_input(
                log_det, x, value_ndims=self.value_ndims)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # compute x
        x = None
        if compute_x:
            n_features = get_static_shape(y)[self.axis]
            kernel = tf.reshape(
                self._kernel_matrix.inv_matrix,
                [1, 1] + list(self._kernel_matrix.shape)
            )
            x = conv2d(
                y, n_features, (1, 1), channels_last=self._channels_last,
                kernel=kernel, use_bias=False
            )

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = apply_log_det_factor(
                -self._kernel_matrix.log_det, y, self.axis, self.value_ndims)
            log_det = broadcast_log_det_against_input(
                log_det, y, value_ndims=self.value_ndims)

        return x, log_det
