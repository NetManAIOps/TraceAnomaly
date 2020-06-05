import six
import numpy as np
import tensorflow as tf

from .debugging import assert_deps
from .tensor_wrapper import TensorWrapper

try:
    import zhusuan as zs
    _DYNAMIC_TENSOR_TYPES = (tf.Tensor, tf.Variable, TensorWrapper,
                             zs.StochasticTensor)
except ImportError:  # pragma: no cover
    _DYNAMIC_TENSOR_TYPES = (tf.Tensor, tf.Variable, TensorWrapper)

__all__ = ['is_integer', 'is_float', 'is_tensor_object', 'TensorArgValidator']

__INTEGER_TYPES = (
    six.integer_types +
    (np.integer, np.int, np.uint,
     np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64)
)
__FLOATING_TYPES = (
    float,
    np.float,
    np.float16, np.float32, np.float64,
)

for _extra_float_type in ('float8', 'float128', 'float256'):
    if hasattr(np, _extra_float_type):
        __FLOATING_TYPES = __FLOATING_TYPES + (getattr(np, _extra_float_type),)


def is_integer(x):
    """
    Test whether or not `x` is a Python or NumPy integer.

    Args:
        x: The object to be tested.

    Returns:
        bool: A boolean indicating whether `x` is a Python or NumPy integer.
    """
    if isinstance(x, bool):
        return False
    return isinstance(x, __INTEGER_TYPES)


def is_float(x):
    """
    Test whether or not `x` is a Python or NumPy float.

    Args:
        x: The object to be tested.

    Returns:
        bool: A boolean indicating whether `x` is a Python or NumPy float.
    """
    return isinstance(x, __FLOATING_TYPES)


def is_tensor_object(x):
    """
    Test whether or not `x` is a tensor object.

    :class:`tf.Tensor`, :class:`tf.Variable`, :class:`TensorWrapper` and
    :class:`zhusuan.StochasticTensor` are considered to be tensor objects.

    Args:
        x: The object to be tested.

    Returns:
        bool: A boolean indicating whether `x` is a tensor object.
    """
    return isinstance(x, _DYNAMIC_TENSOR_TYPES)


class TensorArgValidator(object):
    """Class to validate argument values of tensors."""

    def __init__(self, name):
        """
        Construct the :class:`TensorArgValidator`.

        Args:
            name (str): Name of the argument to be validated, used in error
                messages.
        """
        self.name = name

    def _value_test(self, value, tf_assertion, value_test, err_msg):
        if is_tensor_object(value):
            assert_op = tf_assertion(value, err_msg.format(self.name))
            with assert_deps([assert_op]) as asserted:
                if asserted:  # pragma: no cover
                    value = tf.identity(value)
            return value
        else:
            if not value_test(value):
                raise ValueError(err_msg.format(self.name))
            return value

    def require_int32(self, value):
        """
        Require `value` to be an 32-bit integer.

        Args:
            value: Value to be validated.
                If ``is_tensor_object(value) == True``, it will be casted
                into a :class:`tf.Tensor` with `dtype` as :obj:`tf.int32`.
                If otherwise ``is_integer(value) == True``, the type will
                not be casted, but its value will be checked to ensure it
                falls between  `-2**31 ~ 2**31-1`.
        Returns:
            The validated value.

        Raises:
            TypeError: If specified `value` cannot be casted into int32, or the
                value is out of range.
        """
        if is_tensor_object(value):
            compatible = tf.int32.is_compatible_with(value.dtype)
            if compatible:
                value = tf.convert_to_tensor(value, dtype=tf.int32)
        else:
            compatible = (
                is_integer(value) and
                np.iinfo(np.int32).max >= value >= np.iinfo(np.int32).min
            )
        if not compatible:
            raise TypeError('{} cannot be converted to int32'.
                            format(self.name))
        return value

    def require_non_negative(self, value):
        """
        Require `value` to be non-negative, i.e., ``value >= 0``.

        Args:
            value: Value to be validated.
                If ``is_tensor_object(value) == True``, additional
                assertion will be added to `value`.  Otherwise it will
                be validated against ``value >= 0`` immediately.

        Returns:
            The validated value.

        Raises:
            ValueError: If specified `value` is not non-negative.
        """
        return self._value_test(
            value,
            lambda v, m: tf.assert_greater_equal(
                v, tf.constant(0, value.dtype), message=m),
            lambda v: v >= 0,
            '{} must be non-negative'
        )

    def require_positive(self, value):
        """
        Require `value` to be positive, i.e., ``value > 0``.

        Args:
            value: Value to be validated.
                If ``is_tensor_object(value) == True``, additional assertion
                will be added to `value`.  Otherwise it will be validated
                against ``value > 0`` immediately.

        Returns:
            The validated value.

        Raises:
            ValueError: If specified `value` is not non-negative.
        """
        return self._value_test(
            value,
            lambda v, m: tf.assert_greater(
                v, tf.constant(0, dtype=value.dtype), message=m),
            lambda v: v > 0,
            '{} must be positive'
        )
