import six
import tensorflow as tf

__all__ = [
    'add_name_scope',
    'add_variable_scope',
]


def add_name_scope(method):
    """
    Automatically open a new name scope when calling the method.

    Usage::

        @add_name_scope
        def dense(inputs, name=None):
            return tf.layers.dense(inputs)

    Args:
        method: The method to decorate.  It must accept an optional named
            argument `name`, to receive the inbound name argument.
            If the `name` argument is not specified as named argument during
            calling, the name of the method will be used as `name`.

    Returns:
        The decorated method.
    """
    method_name = method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        if kwargs.get('name') is None:
            kwargs['name'] = method_name
        with tf.name_scope(kwargs['name']):
            return method(*args, **kwargs)
    return wrapper


def add_variable_scope(method):
    """
    Automatically open a new variable scope when calling the method.

    Usage::

        @add_variable_scope
        def dense(inputs):
            return tf.layers.dense(inputs)

    Args:
        method: The method to decorate.
            If the `name` argument is not specified as named argument during
            calling, the name of the method will be used as `name`.

    Returns:
        The decorated method.
    """
    method_name = method.__name__

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        name = kwargs.pop('name', None)
        with tf.variable_scope(name, default_name=method_name):
            return method(*args, **kwargs)
    return wrapper
