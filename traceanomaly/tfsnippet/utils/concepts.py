__all__ = [
    'AutoInitAndCloseable',
    'Disposable',
    'NoReentrantContext',
    'DisposableContext',
]


class AutoInitAndCloseable(object):
    """
    Classes with :meth:`init()` to initialize its internal states, and also
    :meth:`close()` to destroy these states.  The :meth:`init()` method can
    be repeatedly called, which will cause initialization only at the first
    call.  Thus other methods may always call :meth:`init()` at beginning,
    which can bring auto-initialization to the class.

    A context manager is implemented: :meth:`init()` is explicitly called
    when entering the context, while :meth:`destroy()` is called when
    exiting the context.
    """

    _initialized = False

    def _init(self):
        """Override this method to initialize the internal states."""
        raise NotImplementedError()

    def init(self):
        """Ensure the internal states are initialized."""
        if not self._initialized:
            self._init()
            self._initialized = True

    def __enter__(self):
        """Ensure the internal states are initialized."""
        self.init()
        return self

    def _close(self):
        """Override this method to destroy the internal states."""
        raise NotImplementedError()

    def close(self):
        """Ensure the internal states are destroyed."""
        if self._initialized:
            try:
                self._close()
            finally:
                self._initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup the internal states."""
        self.close()


class Disposable(object):
    """
    Classes which can only be used once.
    """

    _already_used = False

    def _check_usage_and_set_used(self):
        """
        Check whether the usage flag, ensure the object has not been used,
        and then set it to be used.
        """
        if self._already_used:
            raise RuntimeError('Disposable object cannot be used twice: {!r}.'.
                               format(self))
        self._already_used = True


class NoReentrantContext(object):
    """
    Base class for contexts which are not reentrant (i.e., if there is
    a context opened by ``__enter__``, and it has not called ``__exit__``,
    the ``__enter__`` cannot be called again).
    """

    _is_entered = False

    def _enter(self):
        """
        Enter the context.  Subclasses should override this instead of
        the true ``__enter__`` method.
        """
        raise NotImplementedError()

    def _exit(self, exc_type, exc_val, exc_tb):
        """
        Exit the context.  Subclasses should override this instead of
        the true ``__exit__`` method.
        """
        raise NotImplementedError()

    def _require_entered(self):
        """
        Require the context to be entered.

        Raises:
            RuntimeError: If the context is not entered.
        """
        if not self._is_entered:
            raise RuntimeError('Context is required be entered: {!r}.'.
                               format(self))

    def __enter__(self):
        if self._is_entered:
            raise RuntimeError('Context is not reentrant: {!r}.'.
                               format(self))
        ret = self._enter()
        self._is_entered = True
        return ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_entered:
            self._is_entered = False
            return self._exit(exc_type, exc_val, exc_tb)


class DisposableContext(NoReentrantContext):
    """
    Base class for contexts which can only be entered once.
    """

    _has_entered = False

    def __enter__(self):
        if self._has_entered:
            raise RuntimeError(
                'A disposable context cannot be entered twice: {!r}.'.
                format(self))
        ret = super(DisposableContext, self).__enter__()
        self._has_entered = True
        return ret
