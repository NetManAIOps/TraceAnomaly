# -*- coding: utf-8 -*-
import six
import tensorflow as tf
from tensorflow.python.client.session import \
    register_session_run_conversion_functions

__all__ = ['TensorWrapper', 'register_tensor_wrapper_class']


class TensorWrapper(object):
    """
    Tensor-like object that wraps a `tf.Tensor` instance.

    This class is typically used to implement `super-tensor` classes,
    adding auxiliary methods to a :class:`tf.Tensor`.
    The derived classes should call `register_rensor_wrapper` to register
    themselves into TensorFlow type system.

    Access to any undefined attributes, properties and methods will be
    transparently proxied to the wrapped tensor.
    Also, :class:`TensorWrapper` can be directly used in mathematical
    expressions and most TensorFlow arithmetic functions.
    For example, ``TensorWrapper(...) + tf.exp(TensorWrapper(...))``.

    On the other hand, :class:`TensorWrapper` are neither :class:`tf.Tensor`
    nor sub-classes of :class:`tf.Tensor`, i.e.,
    ``isinstance(TensorWrapper(...), tf.Tensor) == False``.
    This is essential for sub-classes of :class:`TensorWrapper` being
    converted correctly to :class:`tf.Tensor` by :func:`tf.convert_to_tensor`,
    using the official type conversion system of TensorFlow.

    All the attributes defined in sub-classes of :class:`TensorWrapper`
    must have names starting with ``_self_``.  The properties and methods
    are not restricted by this rule.

    An example of inheriting :class:`TensorWrapper` is shown as follows:

    .. code-block:: python

        class MyTensorWrapper(TensorWrapper):

            def __init__(self, wrapped, flag):
                super(MyTensorWrapper, self).__init__()
                self._self_wrapped = wrapped
                self._self_flag = flag

            @property
            def tensor(self):
                return self._self_wrapped

            @property
            def flag(self):
                return self._self_flag

        register_tensor_wrapper_class(MyTensorWrapper)

        # tests
        t = MyTensorWrapper(tf.constant(0., dtype=tf.float32), flag=123)
        assert(t.dtype == tf.float32)
        assert(t.flag == 123)
    """

    # def __init__(self, wrapped):
    #     if isinstance(wrapped, TensorWrapper):
    #         wrapped = wrapped.__wrapped__
    #     if not isinstance(wrapped, tf.Tensor):
    #         raise TypeError('{!r} is not an instance of `tf.Tensor`'.
    #                         format(wrapped))

    @property
    def tensor(self):
        """
        Get the wrapped :class:`tf.Tensor`.
        Derived classes must override this to return the actual wrapped tensor.

        Returns:
            tf.Tensor: The wrapped tensor.
        """
        raise NotImplementedError()

    # mimic `tf.Tensor` interface
    def __dir__(self):
        if six.PY3:
            ret = object.__dir__(self)
        else:
            # code is based on
            # http://www.quora.com/How-dir-is-implemented-Is-there-any-PEP-related-to-that
            def get_attrs(obj):
                import types
                if not hasattr(obj, '__dict__'):
                    return []  # slots only
                if not isinstance(obj.__dict__, (dict, types.DictProxyType)):
                    raise TypeError('{!r}.__dict__ is not a dictionary'.
                                    format(obj.__name__))
                return obj.__dict__.keys()

            def dir2(obj):
                attrs = set()
                if not hasattr(obj, '__bases__'):
                    # obj is an instance
                    if not hasattr(obj, '__class__'):
                        # slots
                        return sorted(get_attrs(obj))
                    klass = obj.__class__
                    attrs.update(get_attrs(klass))
                else:
                    # obj is a class
                    klass = obj

                for cls in klass.__bases__:
                    attrs.update(get_attrs(cls))
                    attrs.update(dir2(cls))
                attrs.update(get_attrs(obj))
                return list(attrs)

            ret = dir2(self)

        ret = list(set(dir(self.tensor) + ret))
        return ret

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __setattr__(self, name, value):
        if name.startswith('_self_'):
            object.__setattr__(self, name, value)
        elif hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.tensor, name, value)

    def __delattr__(self, name):
        if name.startswith('_self_'):
            object.__delattr__(self, name)
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.tensor, name)

    def __iter__(self):
        raise TypeError('`{}` is not iterable'.format(self.__class__.__name__))

    def __bool__(self):
        raise TypeError(
            'Using a `{}` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.'.format(self.__class__.__name__)
        )

    def __nonzero__(self):
        raise TypeError(
            'Using a `{}` as a Python `bool` is not allowed. '
            'Use `if t is not None:` instead of `if t:` to test if a '
            'tensor is defined, and use TensorFlow ops such as '
            'tf.cond to execute subgraphs conditioned on the value of '
            'a tensor.'.format(self.__class__.__name__)
        )

    # overloading arithmetic operations
    def __abs__(self):
        return tf.abs(self)

    def __neg__(self):
        return tf.negative(self)

    def __add__(self, other):
        return tf.add(self, other)

    def __radd__(self, other):
        return tf.add(other, self)

    def __sub__(self, other):
        return tf.subtract(self, other)

    def __rsub__(self, other):
        return tf.subtract(other, self)

    def __mul__(self, other):
        return tf.multiply(self, other)

    def __rmul__(self, other):
        return tf.multiply(other, self)

    def __div__(self, other):
        return tf.div(self, other)

    def __rdiv__(self, other):
        return tf.div(other, self)

    def __truediv__(self, other):
        return tf.truediv(self, other)

    def __rtruediv__(self, other):
        return tf.truediv(other, self)

    def __floordiv__(self, other):
        return tf.floordiv(self, other)

    def __rfloordiv__(self, other):
        return tf.floordiv(other, self)

    def __mod__(self, other):
        return tf.mod(self, other)

    def __rmod__(self, other):
        return tf.mod(other, self)

    def __pow__(self, other):
        return tf.pow(self, other)

    def __rpow__(self, other):
        return tf.pow(other, self)

    # logical operations
    def __invert__(self):
        return tf.logical_not(self)

    def __and__(self, other):
        return tf.logical_and(self, other)

    def __rand__(self, other):
        return tf.logical_and(other, self)

    def __or__(self, other):
        return tf.logical_or(self, other)

    def __ror__(self, other):
        return tf.logical_or(other, self)

    def __xor__(self, other):
        return tf.logical_xor(self, other)

    def __rxor__(self, other):
        return tf.logical_xor(other, self)

    # boolean operations
    def __lt__(self, other):
        return tf.less(self, other)

    def __le__(self, other):
        return tf.less_equal(self, other)

    def __gt__(self, other):
        return tf.greater(self, other)

    def __ge__(self, other):
        return tf.greater_equal(self, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (tf.convert_to_tensor(self))[item]


def register_tensor_wrapper_class(cls):
    """
    Register a sub-class of :class:`TensorWrapper` into TensorFlow type system.

    Args:
        cls: The subclass of :class:`TensorWrapper` to be registered.
    """
    if not isinstance(cls, six.class_types) or \
            not issubclass(cls, TensorWrapper):
        raise TypeError('`{}` is not a type, or not a subclass of '
                        '`TensorWrapper`'.format(cls))

    def to_tensor(value, dtype=None, name=None, as_ref=False):
        if dtype and not dtype.is_compatible_with(value.dtype):
            raise ValueError('Incompatible type conversion requested to type '
                             '{} for tensor of type {}'.
                             format(dtype.name, value.dtype.name))
        if as_ref:  # pragma: no cover
            raise ValueError('{!r}: Ref type not supported'.format(value))
        return value.tensor

    tf.register_tensor_conversion_function(cls, to_tensor)

    # bring support for session.run(StochasticTensor), and for using as keys
    # in feed_dict.
    register_session_run_conversion_functions(
        cls,
        fetch_function=lambda t: ([t.tensor], lambda val: val[0]),
        feed_function=lambda t, v: [(t.tensor, v)],
        feed_function_for_partial_run=lambda t: [t.tensor]
    )
