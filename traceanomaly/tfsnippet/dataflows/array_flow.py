import numpy as np
from numpy.random import RandomState

from tfsnippet.utils import minibatch_slices_iterator
from .base import ExtraInfoDataFlow

__all__ = ['ArrayFlow']


def _make_readonly(arr):
    arr = np.asarray(arr)
    arr.setflags(write=False)
    return arr


class ArrayFlow(ExtraInfoDataFlow):
    """
    Using numpy-like arrays as data source flow.

    Usage::

        array_flow = DataFlow.arrays([x, y], batch_size=256, shuffle=True,
                                     skip_incomplete=True)
        for batch_x, batch_y in array_flow:
            ...
    """

    def __init__(self, arrays, batch_size,
                 shuffle=False, skip_incomplete=False, random_state=None):
        """
        Construct an :class:`ArrayFlow`.

        Args:
            arrays: List of numpy-like arrays, to be iterated through
                mini-batches.  These arrays should be at least 1-d,
                with identical first dimension.
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).
        """
        # validate parameters
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty.')
        for a in arrays:
            if not hasattr(a, 'shape'):
                raise ValueError('`arrays` must be numpy-like arrays.')
            if len(a.shape) < 1:
                raise ValueError('`arrays` must be at least 1-d arrays.')
        data_length = len(arrays[0])
        for a in arrays[1:]:
            if len(a) != data_length:
                raise ValueError('`arrays` must have the same data length.')

        # memorize the parameters
        super(ArrayFlow, self).__init__(
            array_count=len(arrays),
            data_length=data_length,
            data_shapes=tuple(a.shape[1:] for a in arrays),
            batch_size=batch_size,
            skip_incomplete=skip_incomplete,
            is_shuffled=shuffle
        )
        self._arrays = arrays
        self._random_state = random_state or np.random

        # internal indices buffer
        self._indices_buffer = None

    @property
    def the_arrays(self):
        """Get the tuple of arrays accessed by this :class:`ArrayFlow`."""
        return self._arrays

    def _minibatch_iterator(self):
        # shuffle the source arrays if necessary
        if self.is_shuffled:
            if self._indices_buffer is None:
                t = np.int32 if self._data_length < (1 << 31) else np.int64
                self._indices_buffer = np.arange(self._data_length, dtype=t)
            self._random_state.shuffle(self._indices_buffer)

            def get_slice(s):
                return tuple(
                    _make_readonly(a[self._indices_buffer[s]])
                    for a in self.the_arrays
                )
        else:
            def get_slice(s):
                return tuple(_make_readonly(a[s]) for a in self.the_arrays)

        # now iterator through the mini-batches
        for batch_s in minibatch_slices_iterator(
                length=self.data_length,
                batch_size=self.batch_size,
                skip_incomplete=self.skip_incomplete):
            yield get_slice(batch_s)
