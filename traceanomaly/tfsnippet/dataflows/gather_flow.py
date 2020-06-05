from .base import DataFlow

__all__ = ['GatherFlow']


class GatherFlow(DataFlow):
    """
    Gathering multiple data flows into a single flow.

    Usage::

        x_flow = DataFlow.arrays([x], batch_size=256)
        y_flow = DataFlow.arrays([y], batch_size=256)
        xy_flow = DataFlow.gather([x_flow, y_flow])
    """

    def __init__(self, flows):
        """
        Construct an :class:`IteratorFlow`.

        Args:
            flows(Iterable[DataFlow]): The data flows to gather.
                At least one data flow should be specified, otherwise a
                :class:`ValueError` will be raised.

        Raises:
            ValueError: If not even one data flow is specified.
            TypeError: If a specified flow is not a :class:`DataFlow`.
        """
        flows = tuple(flows)
        if not flows:
            raise ValueError('At least one flow must be specified.')
        for flow in flows:
            if not isinstance(flow, DataFlow):
                raise TypeError('Not a DataFlow: {!r}'.format(flow))
        self._flows = flows

    @property
    def flows(self):
        """
        Get the data flows to be gathered.

        Returns:
            tuple[DataFlow]: The data flows to be gathered.
        """
        return self._flows

    def _minibatch_iterator(self):
        for batches in zip(*self._flows):
            yield sum([tuple(b) for b in batches], ())
