from contextlib import contextmanager

import six
import tensorflow as tf
from tensorflow.python.ops import variable_scope as variable_scope_ops

__all__ = [
    'get_default_scope_name',
    'reopen_variable_scope',
    'root_variable_scope',
]


def get_default_scope_name(name, cls_or_instance=None):
    """
    Generate a valid default scope name.

    Args:
        name (str): The base name.
        cls_or_instance: The class or the instance object, optional.
            If it has attribute ``variable_scope``, then ``variable_scope.name``
            will be used as a hint for the name prefix.  Otherwise, its class
            name will be used as the name prefix.

    Returns:
        str: The generated scope name.
    """
    # compose the candidate name
    prefix = ''
    if cls_or_instance is not None:
        if hasattr(cls_or_instance, 'variable_scope') and \
                isinstance(cls_or_instance.variable_scope, tf.VariableScope):
            vs_name = cls_or_instance.variable_scope.name
            vs_name = vs_name.rsplit('/', 1)[-1]
            prefix = '{}.'.format(vs_name)
        else:
            if not isinstance(cls_or_instance, six.class_types):
                cls_or_instance = cls_or_instance.__class__
            prefix = '{}.'.format(cls_or_instance.__name__).lstrip('_')
    name = prefix + name

    # validate the name
    name = name.lstrip('_')
    return name


@contextmanager
def reopen_variable_scope(var_scope, **kwargs):
    """
    Reopen the specified `var_scope` and its original name scope.

    Args:
        var_scope (tf.VariableScope): The variable scope instance.
        **kwargs: Named arguments for opening the variable scope.
    """
    if not isinstance(var_scope, tf.VariableScope):
        raise TypeError('`var_scope` must be an instance of `tf.VariableScope`')

    with tf.variable_scope(var_scope,
                           auxiliary_name_scope=False,
                           **kwargs) as vs:
        with tf.name_scope(var_scope.original_name_scope):
            yield vs


@contextmanager
def root_variable_scope(**kwargs):
    """
    Open the root variable scope and its name scope.

    Args:
        **kwargs: Named arguments for opening the root variable scope.
    """
    # `tf.variable_scope` does not support opening the root variable scope
    # from empty name.  It always prepend the name of current variable scope
    # to the front of opened variable scope.  So we get the current scope,
    # and pretend it to be the root scope.
    scope = tf.get_variable_scope()
    old_name = scope.name
    try:
        scope._name = ''
        with variable_scope_ops._pure_variable_scope('', **kwargs) as vs:
            scope._name = old_name
            with tf.name_scope(None):
                yield vs
    finally:
        scope._name = old_name
