__all__ = [
    'HookPriority', 'HookEntry', 'HookList',
]


class HookPriority(object):
    """
    Pre-defined hook priorities for :class:`~tfsnippet.trainer.BaseTrainer`
    and :class:`~tfsnippet.trainer.Evaluator`.

    Smaller values take higher priorities.
    """

    EVALUATION = VALIDATION = 500
    DEFAULT = 1000
    ANNEALING = 1500
    LOGGING = 10000


class HookEntry(object):
    """Configurations of a hook entry in :class:`HookList`."""

    def __init__(self, callback, freq, priority, birth):
        """
        Construct a new :class:`HookEntry`.

        Args:
            callback (() -> any): The callable object, as the hook callback.
            freq (int): The frequency for this callback to be called.
            priority (int): The hook priority.  Smaller number has higher
                priority when the hooks are called.
            birth (int): The counter of birth, as an additional key for
                sorting the hook entries, such that old hooks will be
                placed in front of newly added hooks, if they have the
                same priority.
        """
        self.callback = callback
        self.freq = freq
        self.priority = priority
        self.counter = freq
        self.birth = birth

    def reset_counter(self):
        """Reset the `counter` to `freq`, its initial value."""
        self.counter = self.freq

    def maybe_call(self):
        """
        Decrease the `counter`, and call the `callback` if `counter` is less
        than 1.  The counter will be reset to `freq` after then.
        """
        self.counter -= 1
        if self.counter < 1:
            # put this statement before calling the callback, such that
            # the remaining counter would be correctly updated even if
            # any error occurs
            self.counter = self.freq
            self.callback()

    def sort_key(self):
        """Get the key for sorting this hook entry."""
        return self.priority, self.birth


class HookList(object):
    """
    Class for managing hooks in :class:`~tfsnippet.trainer.BaseTrainer`
    and :class:`~tfsnippet.trainer.Evaluator`.

    A hook is a registered callback that the trainers will call at certain
    time, during the training process.  Apart from the callback method,
    each hook has a `freq` and a `priority`.

    *  The `freq` controls how often the particular hook should be called,
       e.g., every 2 epochs.
    *  The `priority` determines the priority (order) of calling the hooks.
       Smaller number corresponds to higher priority.
    """

    def __init__(self):
        """Construct a new :class:`HookList`."""
        self._hooks = []  # type: list[HookEntry]
        self._birth_counter = 0  # to enforce stable ordering

    def add_hook(self, callback, freq=1, priority=HookPriority.DEFAULT):
        """
        Add a hook into the list.

        Args:
            callback (() -> any): The callable object, as the hook callback.
            freq (int): The frequency for this callback to be called.
            priority (int): The hook priority.  Smaller number has higher
                priority when the hooks are called.
        """
        freq = int(freq)
        if freq < 1:
            raise ValueError('`freq` must be at least 1.')
        self._birth_counter += 1
        self._hooks.append(HookEntry(
            callback=callback, freq=freq, priority=priority,
            birth=self._birth_counter
        ))
        self._hooks.sort(key=lambda e: e.sort_key())

    def call_hooks(self):
        """
        Call all the registered hooks.

        If any of the hook raises an error, it will stop the calling chain,
        and propagate the error to upper caller.
        """
        for e in self._hooks:
            e.maybe_call()

    def reset(self):
        """Reset the frequency counter of all hooks."""
        for e in self._hooks:
            e.reset_counter()

    def remove(self, callback):
        """
        Remove all hooks having the specified `callback`.

        Args:
            callback: The callback of the hooks to be removed.

        Returns:
            int: The number of removed hooks.
        """
        return self.remove_if(lambda c, f, t: c == callback)

    def remove_all(self):
        """
        Remove all hooks.

        Returns:
            int: The number of removed hooks.
        """
        pre_count = len(self._hooks)
        self._hooks = []
        return pre_count

    def remove_by_priority(self, priority):
        """
        Remove all hooks having the specified `priority`.

        Args:
            priority (int): The priority of the hooks to be removed.

        Returns:
            int: The number of removed hooks.
        """
        return self.remove_if(lambda c, f, t: t == priority)

    def remove_if(self, condition):
        """
        Remove all hooks matching the specified `condition`.

        Args:
            condition ((callback, freq, priority) -> bool): A callable object
                to tell whether or not a hook should be removed.

        Returns:
            int: The number of removed hooks.
        """
        pre_count = len(self._hooks)
        self._hooks = [
            e for e in self._hooks
            if not condition(e.callback, e.freq, e.priority)
        ]
        return pre_count - len(self._hooks)

    def __repr__(self):
        payload = ','.join(
            '{!r}:{}'.format(e.callback, e.freq)
            for e in self._hooks
        )
        if payload:
            return 'HookList({})'.format(payload)
        else:
            return 'HookList()'
