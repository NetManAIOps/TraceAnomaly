import warnings

import six

from .doc_utils import append_to_doc

__all__ = ['deprecated', 'deprecated_arg']


def _deprecated_warn(message):
    warnings.warn(message, category=DeprecationWarning)


def _name_of(target):
    return target.__name__


class deprecated(object):
    """
    Decorate a class, a method or a function to be deprecated.

    Usage::

        @deprecated()
        def some_function():
            ...

        @deprecated()
        class SomeClass:
            ...
    """

    def __init__(self, message='', version=None):
        """
        Construct a new :class:`deprecated` object, which can be
        used to decorate a class, a method or a function.

        Args:
            message: The deprecation message to display.  It will be appended
                to the end of auto-generated message, i.e., the final message
                would be "`<name>` is deprecated; " + message.
            version: The version since which the decorated target is deprecated.
        """
        self._message = message
        self._version = version

    def __call__(self, target):
        if isinstance(target, six.class_types):
            return self._deprecate_class(target)
        else:
            return self._deprecate_func(target)

    def _deprecate_class(self, cls):
        msg = 'Class `{}` is deprecated'.format(_name_of(cls))
        if self._message:
            msg += '; {}'.format(self._message)
        else:
            msg += '.'

        # patch the __init__ of the class
        init = cls.__init__

        def wrapped(*args, **kwargs):
            _deprecated_warn(msg)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        for k in ('__module__', '__name__', '__qualname__', '__annotations__'):
            if hasattr(init, k):
                setattr(wrapped, k, getattr(init, k))

        if six.PY2:
            wrapped.__doc__ = self._update_doc(init.__doc__)
        else:
            cls.__doc__ = self._update_doc(cls.__doc__)

        return cls

    def _deprecate_func(self, func):
        msg = 'Function `{}` is deprecated'.format(_name_of(func))
        if self._message:
            msg += '; {}'.format(self._message)
        else:
            msg += '.'

        @six.wraps(func)
        def wrapped(*args, **kwargs):
            _deprecated_warn(msg)
            return func(*args, **kwargs)

        wrapped.__doc__ = self._update_doc(wrapped.__doc__)
        # Add a reference to the wrapped function so that we can introspect
        # on function arguments in Python 2 (already works in Python 3)
        wrapped.__wrapped__ = func

        return wrapped

    def _update_doc(self, doc):
        def add_indent(s, spaces):
            return '\n'.join(spaces + l if l.strip() else ''
                             for l in s.split('\n'))

        appendix = '.. deprecated::'
        if self._version:
            appendix += ' {}'.format(self._version)
        if self._message:
            appendix += '\n' + add_indent(self._message, '  ')

        return append_to_doc(doc, appendix)


def deprecated_arg(old_arg, new_arg=None, version=None):
    since = ' since {}'.format(version) if version else ''

    if new_arg is None:
        def wrapper(method):
            msg = 'In function `{}`: argument `' + str(old_arg) + \
                  '` is deprecated' + since + '.'
            msg = msg.format(_name_of(method))

            @six.wraps(method)
            def wrapped(*args, **kwargs):
                if old_arg in kwargs:
                    _deprecated_warn(msg)
                return method(*args, **kwargs)
            return wrapped

    else:
        def wrapper(method):
            msg = 'In function `{}`: argument `' + str(old_arg) + \
                  '` is deprecated' + since + ', use `' + str(new_arg) + \
                  '` instead.'
            msg = msg.format(_name_of(method))

            @six.wraps(method)
            def wrapped(*args, **kwargs):
                if old_arg in kwargs:
                    if new_arg in kwargs:
                        raise TypeError(
                            'You should not specify the deprecated argument '
                            '`{}` and its replacement `{}` at the same time.'.
                            format(old_arg, new_arg)
                        )
                    else:
                        _deprecated_warn(msg)
                return method(*args, **kwargs)
            return wrapped

    return wrapper
