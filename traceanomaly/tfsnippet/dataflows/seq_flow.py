import numpy as np

from .array_flow import ArrayFlow

__all__ = ['SeqFlow']


class SeqFlow(ArrayFlow):
    """
    Using number sequence as data source flow.

    This :class:`SeqFlow` is particularly used for generating the `seed`
    number indices, then fetch the actual data by :class:`MapperFlow`
    according to the seed numbers.

    Usage::

        seq_flow = DataFlow.seq(0, len(x), batch_size=256)
        mapper_flow = seq_flow.map(lambda idx: np.stack(
            [fetch_data_by_index(i) for i in idx]
        ))
    """

    def __init__(self, start, stop, step=1, batch_size=None, shuffle=False,
                 skip_incomplete=False, dtype=np.int32, random_state=None):
        """
        Construct a :class:`SeqFlow`.

        Args:
            start: The starting number of the sequence.
            stop: The ending number of the sequence.
            step: The step of the sequence. (default ``1``)
            batch_size: Batch size of the data flow. Required.
            shuffle (bool): Whether or not to shuffle the numbers before
                iterating? (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            dtype: Data type of the numbers. (default ``np.int32``)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).
        """
        # check the parameters
        if batch_size is None:
            raise ValueError('`batch_size` is required.')

        # memorize the parameters
        super(SeqFlow, self).__init__(
            arrays=[np.arange(start, stop, step, dtype=dtype)],
            batch_size=batch_size,
            shuffle=shuffle,
            skip_incomplete=skip_incomplete,
            random_state=random_state
        )
        self._start = start
        self._stop = stop
        self._step = step

    @property
    def start(self):
        """Get the starting number of the sequence."""
        return self._start

    @property
    def stop(self):
        """Get the ending number of the sequence."""
        return self._stop

    @property
    def step(self):
        """Get the step of the sequence."""
        return self._step
