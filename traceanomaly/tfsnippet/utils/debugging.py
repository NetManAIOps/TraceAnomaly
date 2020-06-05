from contextlib import contextmanager

import tensorflow as tf

from .deprecation import deprecated
from .doc_utils import add_name_arg_doc

__all__ = [
    'is_assertion_enabled',
    'set_assertion_enabled',
    'should_check_numerics',
    'set_check_numerics',
    'maybe_check_numerics',
    'assert_deps',
]


@deprecated('check `tfsnippet.settings.enable_assertions` instead')
def is_assertion_enabled():
    """Whether or not to enable assertions?"""
    from .config_ import settings
    return settings.enable_assertions


@deprecated('set `tfsnippet.settings.enable_assertions = True/False` instead')
def set_assertion_enabled(enabled):
    """
    Set whether or not to enable assertions?

    If the assertions are disabled, then :func:`assert_deps` will not execute
    any given operations.
    """
    from .config_ import settings
    settings.enable_assertions = bool(enabled)


@deprecated('check `tfsnippet.settings.check_numerics` instead')
def should_check_numerics():
    """Whether or not to check numerics?"""
    from .config_ import settings
    return settings.check_numerics


@deprecated('set `tfsnippet.settings.check_numerics = True/False` instead')
def set_check_numerics(enabled):
    """
    Set whether or not to check numerics?

    By checking numerics, one can figure out where the NaNs and Infinities
    originate from.  This affects the behavior of :func:`maybe_check_numerics`,
    and the default behavior of :class:`tfsnippet.distributions.Distribution`
    sub-classes.
    """
    from .config_ import settings
    settings.check_numerics = bool(enabled)


@add_name_arg_doc
def maybe_check_numerics(tensor, message, name=None):
    """
    Check the numerics of `tensor`, if ``should_check_numerics()``.

    Args:
        tensor: The tensor to be checked.
        message: The message to display when numerical issues occur.

    Returns:
        tf.Tensor: The tensor, whose numerics have been checked.
    """
    from .config_ import settings
    tensor = tf.convert_to_tensor(tensor)
    if settings.check_numerics:
        return tf.check_numerics(tensor, message, name=name)
    else:
        return tensor


@contextmanager
def assert_deps(assert_ops):
    """
    If ``is_assertion_enabled() == True``, open a context that will run
    `assert_ops` on exit.  Otherwise do nothing.

    Args:
        assert_ops (Iterable[tf.Operation or None]): A list of assertion
            operations.  :obj:`None` items will be ignored.

    Yields:
        bool: A boolean indicate whether or not the assertion operations
            are not empty, and are executed.
    """
    from .config_ import settings
    assert_ops = [o for o in assert_ops if o is not None]
    if assert_ops and settings.enable_assertions:
        with tf.control_dependencies(assert_ops):
            yield True
    else:
        for op in assert_ops:
            # let TensorFlow not warn about not using this assertion operation
            if hasattr(op, 'mark_used'):
                op.mark_used()
        yield False
