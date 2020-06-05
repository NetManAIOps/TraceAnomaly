import numpy as np

from tfsnippet.utils import DocInherit
from .base import DataFlow

__all__ = [
    'DataMapper', 'SlidingWindow'
]


@DocInherit
class DataMapper(object):
    """
    Base class for all data mappers.

    A :class:`DataMapper` is a callable object, which maps input arrays
    into outputs arrays.  Instances of :class:`DataMapper` are usually
    used as the ``mapper`` of a :class:`tfsnippet.dataflows.MapperFlow`.
    """

    def _transform(self, *args):
        """Subclasses should override this to implement the transformation."""
        raise NotImplementedError()

    def __call__(self, *arrays):
        """
        Transform the input arrays into outputs.

        Args:
            *arrays: Arrays to be transformed.

        Returns:
            tuple[np.ndarray]: The output arrays.
        """
        ret = self._transform(*arrays)
        if not isinstance(ret, (tuple, list)):
            raise TypeError('The output of {} is neither a tuple, nor a list.'.
                            format(self.__class__.__name__))
        return tuple(ret)


class SlidingWindow(DataMapper):
    """
    :class:`DataMapper` for producing sliding windows according to indices.

    Usage::

        data = np.arange(1000)
        sw = SlidingWindow(data, window_size=100)

        # construct a DataFlow from this SlidingWindow
        sw_flow = sw.as_flow(batch_size=64)
        # or equivalently
        sw_flow = DataFlow.seq(
            0, len(data) - sw.window_size + 1, batch_size=64).map(sw)
    """

    def __init__(self, data_array, window_size):
        """
        Construct a :class:`SlidingWindow`.

        Args:
            data_array (np.ndarray): The array from which to extract
                sliding windows.
            window_size (int): Size of each window.
        """
        self._data_array = data_array
        self._window_size = window_size
        offset_dtype = (np.int32 if window_size < (1 << 32) else np.int64)
        self._offset = np.arange(0, window_size, 1, dtype=offset_dtype)

    def as_flow(self, batch_size, shuffle=False, skip_incomplete=False):
        """
        Get a :class:`DataFlow` which iterates through mini-batches of
        sliding windows upon ``data_array``.

        Args:
            batch_size (int): Batch size of the data flow. Required.
            shuffle (bool): Whether or not to shuffle the numbers before
                iterating? (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)

        Returns:
            DataFlow: The data flow for sliding windows.
        """
        data_length = len(self.data_array)
        seq_dtype = (np.int32 if data_length < (1 << 32) else np.int64)
        seq_flow = DataFlow.seq(
            0, data_length - self.window_size + 1, 1, batch_size=batch_size,
            shuffle=shuffle, skip_incomplete=skip_incomplete, dtype=seq_dtype
        )
        return seq_flow.map(self)

    @property
    def data_array(self):
        """Get the data array."""
        return self._data_array

    @property
    def window_size(self):
        """Get the window size."""
        return self._window_size

    def _transform(self, indices):
        return (
            self._data_array[
                indices.reshape(indices.shape + (1,)) + self._offset
            ],
        )
