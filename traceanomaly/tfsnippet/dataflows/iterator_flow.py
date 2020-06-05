from .base import DataFlow

__all__ = ['IteratorFactoryFlow']


class IteratorFactoryFlow(DataFlow):
    """
    Data flow constructed from an iterator factory.

    Usage::

        x_flow = DataFlow.arrays([x], batch_size=256)
        y_flow = DataFlow.arrays([y], batch_size=256)
        xy_flow = DataFlow.iterator_factory(lambda: (
            (x, y) for (x,), (y,) in zip(x_flow, y_flow)
        ))
    """

    def __init__(self, factory):
        """
        Construct an :class:`IteratorFlow`.

        Args:
            factory (() -> Iterator or Iterable): A factory method for
                constructing the mini-batch iterators for each epoch.
        """
        self._factory = factory

    def _minibatch_iterator(self):
        for batch in self._factory():
            yield batch
