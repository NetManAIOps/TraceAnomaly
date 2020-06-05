import six
import tensorflow as tf

from .doc_utils import DocInherit
from .shape_utils import get_static_shape
from .type_utils import is_integer

__all__ = ['InputSpec', 'ParamSpec']


def _try_parse_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


@DocInherit
class _TensorSpec(object):
    """
    Base class to describe and validate the specification of a tensor.
    """

    def __init__(self, shape=None, dtype=None):
        """
        Construct a new :class:`TensorSpec`.

        Args:
            shape (Iterable[int or str or None]): A tuple to describe the shape
                of the tensor.  Each item can be one of the following values:

                *  A positive integer: indicates a dimension with known size.

                *  -1, :obj:`None`, or '?': indicates a dimension with any size.

                * str(a positive integer) + '?': indicate a dimension with
                    known size equal to the number, or unknown size.

                * '*': indicates a dimension with any DETERMINED size.

                * '...': should be the first item, indicates the tensor can
                  have zero or many dimensions at the left of the remaining
                  dimensions.

                (default :obj:`None`, no specification for the shape)
            dtype: The data type of the tensor.
                (default :obj:`dtype`, no specification for the data type)
        """
        # check the shape argument
        if shape is not None:
            shape = tuple(shape)
            if shape == ('...',):  # this should be equivalent to shape == None
                shape = None

        if shape is not None:
            allow_more_dims = False
            value_shape = []
            for i, s in enumerate(shape):
                if s == '...':
                    if i != 0:
                        raise ValueError('`...` should only be the first item '
                                         'of `shape`.')
                    allow_more_dims = True
                elif s == '?' or s is None or s == -1:
                    value_shape.append('?')
                elif isinstance(s, six.string_types) and \
                        s.endswith('?') and \
                        _try_parse_int(s[:-1]) is not None and \
                        _try_parse_int(s[:-1]) > 0:
                    value_shape.append(s)
                elif s == '*':
                    value_shape.append('*')
                elif is_integer(s) and s > 0:
                    value_shape.append(int(s))
                else:
                    raise ValueError('Invalid value in `shape` {}: {}'.
                                     format(shape, s))
            value_shape = tuple(value_shape)
        else:
            allow_more_dims = None
            value_shape = None

        self._allow_more_dims = allow_more_dims
        self._value_shape = value_shape

        # check the dtype argument
        if dtype is not None:
            dtype = tf.as_dtype(dtype)
        self._dtype = dtype

    def __eq__(self, other):
        return (
                isinstance(other, _TensorSpec) and
                self._value_shape == other._value_shape and
                self._allow_more_dims == other._allow_more_dims and
                self._dtype == other._dtype
        )

    def __hash__(self):
        return hash((
            self._value_shape,
            self._allow_more_dims,
            self._dtype
        ))

    def __repr__(self):
        spec = []
        if self._value_shape is not None:
            spec.append('shape=' + self._format_shape())
        if self._dtype is not None:
            spec.append('dtype=' + self._dtype.name)
        return '{}({})'.format(self.__class__.__name__, ','.join(spec))

    @property
    def shape(self):
        """
        Get the shape specification.

        Returns:
            tuple[int or str or None] or None: The value shape, or None.
        """
        if self._allow_more_dims:
            return ('...',) + self._value_shape
        else:
            return self._value_shape

    @property
    def value_shape(self):
        """
        Get the value shape (the shape excluding leading "...").

        Returns:
            tuple[int or str or None] or None: The value shape, or None.
        """
        return self._value_shape

    @property
    def value_ndims(self):
        """
        Get the value shape ndims.

        Returns:
            int or None: The value shape ndims, or None.
        """
        if self._value_shape is not None:
            return len(self._value_shape)

    @property
    def dtype(self):
        """
        Get the data type of the tensor.

        Returns:
            tf.DType or None: The data type, or None.
        """
        return self._dtype

    def _format_shape(self):
        shape = self.shape
        if len(shape) == 1:
            return '({},)'.format(shape[0])
        else:
            return '({})'.format(','.join(str(s) for s in shape))

    def _validate_shape(self, name, x):
        if self._value_shape is None:
            return

        x_shape = get_static_shape(x)

        def raise_error():
            raise ValueError('The shape of `{}` is invalid: expected {}, but '
                             'got {}.'.
                             format(name, self._format_shape(), x_shape))

        if x_shape is None:
            raise_error()

        if not self._allow_more_dims and len(x_shape) != len(self._value_shape):
            raise_error()

        if self._allow_more_dims and len(x_shape) < len(self._value_shape):
            raise_error()

        if self._value_shape:  # in case self._shape == ()
            right_shape = x_shape[-len(self._value_shape):]
            for a, b in zip(right_shape, self._value_shape):
                if b == '*':
                    if a is None:
                        raise_error()
                elif b == '?':
                    pass
                elif isinstance(b, six.string_types) and b.endswith('?'):
                    if a is not None and a != int(b[:-1]):
                        raise_error()
                else:  # b is an integer
                    assert(is_integer(b))
                    if a != b:
                        raise_error()

    def _validate_dtype(self, name, x):
        if self._dtype is not None:
            if x.dtype.base_dtype != self._dtype:
                raise TypeError('The dtype of `{}` is invalid: expected {}, '
                                'but got {}.'.
                                format(name, self._dtype, x.dtype.base_dtype))

    def validate(self, name, x):
        """
        Validate the input tensor `x`.

        Args:
            name (str): The name of the tensor, used in error messages.
            x: The input tensor.

        Returns:
            The validated tensor.

        Raises:
            ValueError, TypeError: If `x` cannot pass validation.
        """
        x = tf.convert_to_tensor(x)
        self._validate_shape(name, x)
        self._validate_dtype(name, x)
        return x


class InputSpec(_TensorSpec):
    """
    Class to describe the specification for an input tensor.

    Mostly identical with :class:`TensorSpec`.
    """


class ParamSpec(_TensorSpec):
    """
    Class to describe the specification for a parameter.

    Unlike :class:`TensorSpec`, the shape of the parameter must be fully
    determined, i.e., without any unknown dimension, and the ndims must
    be identical to the specification.
    """

    def __init__(self, *args, **kwargs):
        super(ParamSpec, self).__init__(*args, **kwargs)
        if self.shape is None or any(not is_integer(s) for s in self.shape):
            shape_format = None if self.shape is None else self._format_shape()
            raise ValueError('The shape of a `ParamSpec` must be fully '
                             'determined: got {}.'.format(shape_format))
