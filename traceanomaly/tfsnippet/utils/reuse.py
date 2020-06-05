import functools
import inspect
import weakref
from contextlib import contextmanager

import six
import tensorflow as tf

from .doc_utils import DocInherit
from .scope import root_variable_scope, get_default_scope_name
from .misc import ContextStack, camel_to_underscore
from .tfver import is_tensorflow_version_higher_or_equal

__all__ = [
    'get_reuse_stack_top', 'instance_reuse', 'global_reuse',
    'VarScopeObject'
]


def require_at_least_tensorflow_1_5():
    if not is_tensorflow_version_higher_or_equal('1.5.0'):  # pragma: no cover
        raise RuntimeError('The reuse utilities are only tested for '
                           'TensorFlow >= 1.5.0.  Using these utilities with '
                           'any lower versions of TensorFlow are totally '
                           'not allowed.')


_reuse_stack = ContextStack()  # stack to track the opened reuse scopes


def get_reuse_stack_top():
    """
    Get the top of the reuse scope stack.

    Returns:
        tf.VaribleScope: The top of the reuse scope stack.
    """
    return _reuse_stack.top()


@contextmanager
def _reuse_context(vs):
    _reuse_stack.push(vs)
    try:
        yield
    finally:
        _reuse_stack.pop()


def instance_reuse(method_or_scope=None, _sentinel=None, scope=None):
    """
    Decorate an instance method to reuse a variable scope automatically.

    This decorator should be applied to unbound instance methods, and
    the instance that owns the methods should have :attr:`variable_scope`
    attribute.  The first time to enter a decorated method will open
    a new variable scope under the `variable_scope` of the instance.
    This variable scope will be reused the next time to enter this method.
    For example:

    .. code-block:: python

        class Foo(object):

            def __init__(self, name):
                with tf.variable_scope(name) as vs:
                    self.variable_scope = vs

            @instance_reuse
            def bar(self):
                return tf.get_variable('bar', ...)

        foo = Foo()
        bar = foo.bar()
        bar_2 = foo.bar()
        assert(bar is bar_2)  # should be True

    By default the name of the variable scope should be chosen according to
    the name of the decorated method.  You can change this behavior by
    specifying an alternative name, for example:

    .. code-block:: python

        class Foo(object):

            @instance_reuse('scope_name')
            def foo(self):
                # name will be self.variable_scope.name + '/foo/bar'
                return tf.get_variable('bar', ...)

    Unlike the behavior of :func:`global_reuse`, if you have two methods
    sharing the same scope name, they will indeed use the same variable scope.
    For example:

    .. code-block:: python

        class Foo(object):

            @instance_reuse('foo')
            def foo_1(self):
                return tf.get_variable('bar', ...)

            @instance_reuse('foo')
            def foo_2(self):
                return tf.get_variable('bar', ...)


            @instance_reuse('foo')
            def foo_2(self):
                return tf.get_variable('bar2', ...)

        foo = Foo()
        foo.foo_1()  # its name should be variable_scope.name + '/foo/bar'
        foo.foo_2()  # should raise an error, because 'bar' variable has
                     # been created, but the decorator of `foo_2` does not
                     # aware of this, so has not set ``reused = True``.
        foo.foo_3()  # its name should be variable_scope.name + '/foo/bar2'

    The reason to take this behavior is because the TensorFlow seems to have
    some absurd behavior when using ``tf.variable_scope(..., default_name=?)``
    to uniquify the variable scope name.  In some cases we the following
    absurd behavior would appear:

    .. code-block:: python

        @global_reuse
        def foo():
            with tf.variable_scope(None, default_name='bar') as vs:
                return vs

        vs1 = foo()  # vs.name == 'foo/bar'
        vs2 = foo()  # still expected to be 'foo/bar', but sometimes would be
                     # 'foo/bar_1'. this absurd behavior is related to the
                     # entering and exiting of variable scopes, which is very
                     # hard to diagnose.

    In order to compensate such behavior, if you have specified the
    ``scope`` argument of a :class:`VarScopeObject`, then it will always take
    the desired variable scope.  Also, constructing a `VarScopeObject` within
    a method or a function decorated by `global_reuse` or `instance_reuse` has
    been totally disallowed.

    See Also:
        :class:`tfsnippet.utils.VarScopeObject`,
        :func:`tfsnippet.utils.global_reuse`
    """
    require_at_least_tensorflow_1_5()

    if _sentinel is not None:  # pragma: no cover
        raise TypeError('`scope` must be specified as named argument.')

    if isinstance(method_or_scope, six.string_types):
        scope = method_or_scope
        method = None
    else:
        method = method_or_scope

    if method is None:
        return functools.partial(instance_reuse, scope=scope)

    scope = scope or method.__name__

    if '/' in scope:
        raise ValueError('`instance_reuse` does not support "/" in scope name.')

    # check whether or not `method` looks like an instance method
    if six.PY2:
        getargspec = inspect.getargspec
    else:
        getargspec = inspect.getfullargspec

    if inspect.ismethod(method):
        raise TypeError('`method` is expected to be unbound instance method')
    argspec = getargspec(method)
    if not argspec.args or argspec.args[0] != 'self':
        raise TypeError('`method` seems not to be an instance method '
                        '(whose first argument should be `self`)')

    # determine the scope name
    scope = scope or method.__name__

    # Until now, we have checked all the arguments, such that `method`
    # is the function to be decorated, and `scope` is the base name
    # for the variable scope.  We can now generate the closure used
    # to track the variable scopes.
    variable_scopes = weakref.WeakKeyDictionary()

    @six.wraps(method)
    def wrapped(*args, **kwargs):
        # get the instance from the arguments and its variable scope
        obj = args[0]
        obj_vs = obj.variable_scope

        if not isinstance(obj_vs, tf.VariableScope):
            raise TypeError('`variable_scope` attribute of the instance {!r} '
                            'is expected to be a `tf.VariableScope`, but got '
                            '{!r}'.format(obj, obj_vs))

        # now ready to create the variable scope for the method
        if obj not in variable_scopes:
            graph = tf.get_default_graph()

            # Branch #1.1: first time to enter the method, and we are not
            #   in the object's variable scope.  We should first pick up
            #   the object's variable scope before creating our desired
            #   variable scope.  However, if we execute the method right after
            #   we obtain the new variable scope, we will not be in the correct
            #   name scope.  So we should exit the scope, then re-enter our
            #   desired variable scope.
            if graph.get_name_scope() + '/' != obj_vs.original_name_scope or \
                    tf.get_variable_scope().name != obj_vs.name:
                with tf.variable_scope(obj_vs, auxiliary_name_scope=False):
                    with tf.name_scope(obj_vs.original_name_scope):
                        # now we are here in the object's variable scope, and
                        # its original name scope.  Thus we can now create the
                        # method's variable scope.
                        with tf.variable_scope(scope) as vs:
                            variable_scopes[obj] = vs

                with tf.variable_scope(vs), _reuse_context(vs):
                    return method(*args, **kwargs)

            # Branch #1.2: first time to enter the method, and we are just
            #   in the object's variable scope.  So we can happily create a new
            #   variable scope, and just call the method immediately.
            else:
                with tf.variable_scope(scope) as vs, _reuse_context(vs):
                    variable_scopes[obj] = vs
                    return method(*args, **kwargs)

        else:
            # Branch #2: not the first time to enter the method, so we
            #   should reopen the variable scope with reuse set to `True`.
            vs = variable_scopes[obj]
            with tf.variable_scope(vs, reuse=True), _reuse_context(vs):
                return method(*args, **kwargs)

    return wrapped


def global_reuse(method_or_scope=None, _sentinel=None, scope=None):
    """
    Decorate a function to reuse a variable scope automatically.

    The first time to enter a function decorated by this utility will
    open a new variable scope under the root variable scope.
    This variable scope will be reused the next time to enter this function.
    For example:

    .. code-block:: python

        @global_reuse
        def foo():
            return tf.get_variable('bar', ...)

        bar = foo()
        bar_2 = foo()
        assert(bar is bar_2)  # should be True

    By default the name of the variable scope should be chosen according to
    the name of the decorated method.  You can change this behavior by
    specifying an alternative name, for example:

    .. code-block:: python

        @global_reuse('dense')
        def dense_layer(inputs):
            w = tf.get_variable('w', ...)  # name will be 'dense/w'
            b = tf.get_variable('b', ...)  # name will be 'dense/b'
            return tf.matmul(w, inputs) + b

    If you have two functions sharing the same scope name, they will not
    use the same variable scope.  Instead, one of these two functions will
    have its scope name added with a suffix '_?', for example:

    .. code-block:: python

        @global_reuse('foo')
        def foo_1():
            return tf.get_variable('bar', ...)

        @global_reuse('foo')
        def foo_2():
            return tf.get_variable('bar', ...)

        assert(foo_1().name == 'foo/bar')
        assert(foo_2().name == 'foo_1/bar')

    The variable scope name will depend on the calling order of these
    two functions, so you should better not guess the scope name by yourself.

    Note:
        If you use Keras, you SHOULD NOT create a Keras layer inside a
        `global_reuse` decorated function.  Instead, you should create it
        outside the function, and pass it into the function.

    See Also:
        :func:`tfsnippet.utils.instance_reuse`
    """
    require_at_least_tensorflow_1_5()

    if _sentinel is not None:  # pragma: no cover
        raise TypeError('`scope` must be specified as named argument.')

    if isinstance(method_or_scope, six.string_types):
        scope = method_or_scope
        method = None
    else:
        method = method_or_scope

    if method is None:
        return functools.partial(global_reuse, scope=scope)

    scope = scope or method.__name__
    if '/' in scope:
        raise ValueError('`global_reuse` does not support "/" in scope name.')

    # Until now, we have checked all the arguments, such that `method`
    # is the function to be decorated, and `scope` is the base name
    # for the variable scope.  We can now generate the closure used
    # to track the variable scopes.
    variable_scopes = weakref.WeakKeyDictionary()

    @six.wraps(method)
    def wrapped(*args, **kwargs):
        graph = tf.get_default_graph()

        if graph not in variable_scopes:
            # Branch #1.1: first time to enter the function, and we are not
            #   in the root variable scope.  We should pick up the root
            #   variable scope before creating our desired variable scope.
            #   However, if we execute the method right after we obtain the
            #   new variable scope, we will not be in the correct name scope.
            #   So we should exit the scope, then re-enter our desired
            #   variable scope.
            if graph.get_name_scope() or tf.get_variable_scope().name:
                with root_variable_scope():
                    with tf.variable_scope(None, default_name=scope) as vs:
                        variable_scopes[graph] = vs

                with tf.variable_scope(vs), _reuse_context(vs):
                    return method(*args, **kwargs)

            # Branch #1.2: first time to enter the function, and we are just
            #   in the root variable scope.  So we can happily create a new
            #   variable scope, and just call the method immediately.
            else:
                with tf.variable_scope(None, default_name=scope) as vs, \
                        _reuse_context(vs):
                    variable_scopes[graph] = vs
                    return method(*args, **kwargs)

        else:
            # Branch #2: not the first time to enter the function, so we
            #   should reopen the variable scope with reuse set to `True`.
            vs = variable_scopes[graph]
            with tf.variable_scope(vs, reuse=True), _reuse_context(vs):
                return method(*args, **kwargs)

    return wrapped


@DocInherit
class VarScopeObject(object):
    """
    Base class for objects that own a variable scope.

    The :class:`VarScopeObject` can be used along with :func:`instance_reuse`,
    for example::

        class YourVarScopeObject(VarScopeObject):

            @instance_reuse
            def foo(self):
                return tf.get_variable('bar', ...)

        o = YourVarScopeObject('object_name')
        o.foo()  # You should get a variable with name "object_name/foo/bar"

    To build variables in the constructor of derived classes, you may use
    ``reopen_variable_scope(self.variable_scope)`` to open the original
    variable scope and its name scope, right after the constructor of
    :class:`VarScopeObject` has been called, for example::

        class YourVarScopeObject(VarScopeObject):

            def __init__(self, name=None, scope=None):
                super(YourVarScopeObject, self).__init__(name=name, scope=scope)
                with reopen_variable_scope(self.variable_scope):
                    self.w = tf.get_variable('w', ...)

    See Also:
        :func:`tfsnippet.utils.instance_reuse`.
    """

    def __init__(self, name=None, scope=None):
        """
        Construct the :class:`VarScopeObject`.

        Args:
            name (str): Default name of the variable scope.  Will be uniquified.
                If not specified, generate one according to the class name.
            scope (str): The name of the variable scope.
        """
        scope = scope or None
        name = name or None

        if not scope and not name:
            default_name = get_default_scope_name(
                camel_to_underscore(self.__class__.__name__))
        else:
            default_name = name

        with tf.variable_scope(scope, default_name=default_name) as vs:
            self._variable_scope = vs       # type: tf.VariableScope
            self._name = name

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.variable_scope.name)

    @property
    def name(self):
        """Get the name of this object."""
        return self._name

    @property
    def variable_scope(self):
        """Get the variable scope of this object."""
        return self._variable_scope
