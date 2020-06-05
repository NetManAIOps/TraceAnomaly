import tensorflow as tf

from tfsnippet.ops import assert_rank_at_least
from tfsnippet.utils import (add_name_arg_doc, get_static_shape, get_shape,
                             assert_deps, broadcast_to_shape_strict,
                             maybe_check_numerics, DocInherit, is_tensor_object,
                             TensorWrapper, broadcast_to_shape,
                             register_tensor_wrapper_class)

__all__ = [
    'broadcast_log_det_against_input',
]


@add_name_arg_doc
def is_log_det_shape_matches_input(log_det, input, value_ndims, name=None):
    """
    Check whether or not the shape of `log_det` matches the shape of `input`.

    Basically, the shapes of `log_det` and `input` should satisfy::

        if value_ndims > 0:
            assert(log_det.shape == input.shape[:-value_ndims])
        else:
            assert(log_det.shape == input.shape)

    Args:
        log_det: Tensor, the log-determinant.
        input: Tensor, the input.
        value_ndims (int): The number of dimensions of each values sample.

    Returns:
        bool or tf.Tensor: A boolean or a tensor, indicating whether or not
            the shape of `log_det` matches the shape of `input`.
    """
    if not is_tensor_object(log_det):
        log_det = tf.convert_to_tensor(log_det)
    if not is_tensor_object(input):
        input = tf.convert_to_tensor(input)
    value_ndims = int(value_ndims)

    with tf.name_scope(name or 'is_log_det_shape_matches_input'):
        log_det_shape = get_static_shape(log_det)
        input_shape = get_static_shape(input)

        # if both shapes have deterministic ndims, we can compare each axis
        # separately.
        if log_det_shape is not None and input_shape is not None:
            if len(log_det_shape) + value_ndims != len(input_shape):
                return False
            dynamic_axis = []

            for i, (a, b) in enumerate(zip(log_det_shape, input_shape)):
                if a is None or b is None:
                    dynamic_axis.append(i)
                elif a != b:
                    return False

            if not dynamic_axis:
                return True

            log_det_shape = get_shape(log_det)
            input_shape = get_shape(input)
            return tf.reduce_all([
                tf.equal(log_det_shape[i], input_shape[i])
                for i in dynamic_axis
            ])

        # otherwise we need to do a fully dynamic check, including check
        # ``log_det.ndims + value_ndims == input_shape.ndims``
        is_ndims_matches = tf.equal(
            tf.rank(log_det) + value_ndims, tf.rank(input))
        log_det_shape = get_shape(log_det)
        input_shape = get_shape(input)
        if value_ndims > 0:
            input_shape = input_shape[:-value_ndims]

        return tf.cond(
            is_ndims_matches,
            lambda: tf.reduce_all(tf.equal(
                # The following trick ensures we're comparing two tensors
                # with the same shape, such as to avoid some potential issues
                # about the cond operation.
                tf.concat([log_det_shape, input_shape], 0),
                tf.concat([input_shape, log_det_shape], 0),
            )),
            lambda: tf.constant(False, dtype=tf.bool)
        )


@add_name_arg_doc
def assert_log_det_shape_matches_input(log_det, input, value_ndims, name=None):
    """
    Assert the shape of `log_det` matches the shape of `input`.

    Args:
        log_det: Tensor, the log-determinant.
        input: Tensor, the input.
        value_ndims (int): The number of dimensions of each values sample.

    Returns:
        tf.Operation or None: The assertion operation, or None if the
            assertion can be made statically.
    """
    if not is_tensor_object(log_det):
        log_det = tf.convert_to_tensor(log_det)
    if not is_tensor_object(input):
        input = tf.convert_to_tensor(input)
    value_ndims = int(value_ndims)

    with tf.name_scope(name or 'assert_log_det_shape_matches_input'):
        cmp_result = is_log_det_shape_matches_input(log_det, input, value_ndims)
        error_message = (
            'The shape of `log_det` does not match the shape of '
            '`input`: log_det {!r} vs input {!r}, value_ndims is {!r}'.
            format(log_det, input, value_ndims)
        )

        if cmp_result is False:
            raise AssertionError(error_message)

        elif cmp_result is True:
            return None

        else:
            return tf.assert_equal(cmp_result, True, message=error_message)


@add_name_arg_doc
def broadcast_log_det_against_input(log_det, input, value_ndims, name=None):
    """
    Broadcast the shape of `log_det` to match the shape of `input`.

    Args:
        log_det: Tensor, the log-determinant.
        input: Tensor, the input.
        value_ndims (int): The number of dimensions of each values sample.

    Returns:
        tf.Tensor: The broadcasted log-determinant.
    """
    log_det = tf.convert_to_tensor(log_det)
    input = tf.convert_to_tensor(input)
    value_ndims = int(value_ndims)

    with tf.name_scope(name or 'broadcast_log_det_to_input_shape',
                       values=[log_det, input]):
        shape = get_shape(input)
        if value_ndims > 0:
            err_msg = (
                'Cannot broadcast `log_det` against `input`: log_det is {}, '
                'input is {}, value_ndims is {}.'.
                format(log_det, input, value_ndims)
            )
            with assert_deps([assert_rank_at_least(
                    input, value_ndims, message=err_msg)]):
                shape = shape[:-value_ndims]

        return broadcast_to_shape_strict(log_det, shape)


@DocInherit
class Scale(object):
    """
    Base class to help compute `x * scale`, `x / scale`, `log(scale)` and
    `log(1. / scale)`, given `scale = f(pre_scale)`.
    """

    def __init__(self, pre_scale, epsilon):
        """
        Construct a new :class:`Scale`.

        Args:
            pre_scale: Used to compute the scale via `scale = f(pre_scale)`.
            epsilon: Small float number to avoid dividing by zero or taking
                logarithm of zero.
        """
        self._pre_scale = tf.convert_to_tensor(pre_scale)
        self._epsilon = epsilon
        self._cached_scale = None
        self._cached_inv_scale = None
        self._cached_log_scale = None
        self._cached_neg_log_scale = None

    def _scale(self):
        raise NotImplementedError()

    def _inv_scale(self):
        raise NotImplementedError()

    def _log_scale(self):
        raise NotImplementedError()

    def _neg_log_scale(self):
        raise NotImplementedError()

    def scale(self):
        """Compute `f(pre_scale)`."""
        if self._cached_scale is None:
            with tf.name_scope('scale', values=[self._pre_scale]):
                self._cached_scale = maybe_check_numerics(
                    self._scale(),
                    message=('numeric issues in {}.scale'.
                             format(self.__class__.__name__))
                )
        return self._cached_scale

    def inv_scale(self):
        """Compute `1. / f(pre_scale)`."""
        if self._cached_inv_scale is None:
            with tf.name_scope('inv_scale', values=[self._pre_scale]):
                self._cached_inv_scale = maybe_check_numerics(
                    self._inv_scale(),
                    message=('numeric issues in {}.inv_scale'.
                             format(self.__class__.__name__))
                )
        return self._cached_inv_scale

    def log_scale(self):
        """Compute `log(f(pre_scale))`."""
        if self._cached_log_scale is None:
            with tf.name_scope('log_scale', values=[self._pre_scale]):
                self._cached_log_scale = maybe_check_numerics(
                    self._log_scale(),
                    message=('numeric issues in {}.log_scale'.
                             format(self.__class__.__name__))
                )
        return self._cached_log_scale

    def neg_log_scale(self):
        """Compute `-log(f(pre_scale))`."""
        if self._cached_neg_log_scale is None:
            with tf.name_scope('neg_log_scale', values=[self._pre_scale]):
                self._cached_neg_log_scale = maybe_check_numerics(
                    self._neg_log_scale(),
                    message=('numeric issues in {}.neg_log_scale'.
                             format(self.__class__.__name__))
                )
        return self._cached_neg_log_scale

    def _mult(self, x):
        """Compute `x * f(pre_scale)`."""
        return x * self.scale()

    def _div(self, x):
        """Compute `x / f(pre_scale)`."""
        return x * self.inv_scale()

    def __rdiv__(self, other):
        return self._div(tf.convert_to_tensor(other))

    def __rtruediv__(self, other):
        return self._div(tf.convert_to_tensor(other))

    def __rmul__(self, other):
        return self._mult(tf.convert_to_tensor(other))


class SigmoidScale(Scale):
    """A variant of :class:`Scale`, where `scale = sigmoid(pre_scale)`."""

    def _scale(self):
        return tf.nn.sigmoid(self._pre_scale)

    def _inv_scale(self):
        return tf.exp(-self._pre_scale) + 1.

    def _log_scale(self):
        return -tf.nn.softplus(-self._pre_scale)

    def _neg_log_scale(self):
        return tf.nn.softplus(-self._pre_scale)


class ExpScale(Scale):
    """A variant of :class:`Scale`, where `scale = exp(pre_scale)`."""

    def _scale(self):
        return tf.exp(self._pre_scale)

    def _inv_scale(self):
        return tf.exp(-self._pre_scale)

    def _log_scale(self):
        return self._pre_scale

    def _neg_log_scale(self):
        return -self._pre_scale


class LinearScale(Scale):
    """A variant of :class:`Scale`, where `scale = pre_scale`."""

    def _scale(self):
        return self._pre_scale

    def _inv_scale(self):
        return 1. / self._pre_scale

    def _log_scale(self):
        return tf.log(tf.maximum(tf.abs(self._pre_scale), self._epsilon))

    def _neg_log_scale(self):
        return -tf.log(tf.maximum(tf.abs(self._pre_scale), self._epsilon))

    def _div(self, x):
        # TODO: use epsilon to prevent dividing by zero
        return maybe_check_numerics(
            x / self.scale(), message='numeric issues in LinearScale._div')


class ZeroLogDet(TensorWrapper):
    """
    A special object to represent a zero log-determinant.

    Using this class instead of constructing a `tf.Tensor` via `tf.zeros`
    may help to reduce the introduced operations and the execution time cost.
    """

    def __init__(self, shape, dtype):
        """
        Construct a new :class:`ZeroLogDet`.

        Args:
            shape (tuple[int] or Tensor): The shape of the log-det.
            dtype (tf.DType): The data type.
        """
        if not is_tensor_object(shape):
            shape = tuple(int(v) for v in shape)
        self._self_shape = shape
        self._self_dtype = tf.as_dtype(dtype)
        self._self_tensor = None

    def __repr__(self):
        return 'ZeroLogDet({},{})'.format(self._self_shape, self.dtype.name)

    @property
    def dtype(self):
        """Get the data type of the log-det."""
        return self._self_dtype

    @property
    def log_det_shape(self):
        """Get the shape of the log-det."""
        return self._self_shape

    @property
    def tensor(self):
        if self._self_tensor is None:
            self._self_tensor = tf.zeros(self.log_det_shape, dtype=self.dtype)
        return self._self_tensor

    def __neg__(self):
        return self

    def __add__(self, other):
        return broadcast_to_shape(other, self.log_det_shape)

    def __sub__(self, other):
        return -broadcast_to_shape(other, self.log_det_shape)


register_tensor_wrapper_class(ZeroLogDet)
