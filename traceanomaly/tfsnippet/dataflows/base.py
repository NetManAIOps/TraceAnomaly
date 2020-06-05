import numpy as np

__all__ = ['DataFlow', 'ExtraInfoDataFlow']


class DataFlow(object):
    """
    Data flows are objects for constructing mini-batch iterators.

    There are two major types of :class:`DataFlow` classes: data sources
    and data transformers.  Data sources, like the :class:`ArrayFlow`,
    produce mini-batches from underlying data sources.  Data transformers,
    like :class:`MapperFlow`, produce mini-batches by transforming arrays
    from the source.

    All :class:`DataFlow` subclasses shipped by :mod:`tfsnippet.dataflows`
    can be constructed by factory methods of this base class.  For example::

        # :class:`ArrayFlow` from arrays
        array_flow = DataFlow.arrays([x, y], batch_size=256, shuffle=True)

        # :class:`MapperFlow` by adding the two arrays from `array_flow`
        mapper_flow = array_flow.map(lambda x, y: (x + y,))
    """

    _is_iter_entered = False
    _implicit_iterator = None  # tracking the iterator for :meth:`next_batch()`
    _current_batch = None  # tracking the result of last :meth:`next_batch()`

    def _minibatch_iterator(self):
        """
        Get the mini-batch iterator.  Subclasses should override this to
        implement the data flow.

        Yields:
            tuple[np.ndarray]: Mini-batches of tuples of numpy arrays.
                The arrays might be read-only.
        """
        raise NotImplementedError()

    def __iter__(self):
        """
        Iterate through the mini-batches.  Not reentrant.

        Some subclasses may also inherit from :class:`NoReentrantContext`,
        thus a context must be firstly entered before using such data flows
        as iterators, for example::

            with DataFlow.threaded(...) as df:
                for epoch in epochs:
                    for batch_x, batch_y in df:
                        ...

        Yields:
            tuple[np.ndarray]: Mini-batches of tuples of numpy arrays.
                The arrays might be read-only.
        """
        if self._is_iter_entered:
            raise RuntimeError('{}.__iter__ is not reentrant.'.
                               format(self.__class__.__name__))
        self._is_iter_entered = True
        try:
            for b in self._minibatch_iterator():
                yield b
        finally:
            self._is_iter_entered = False

    def get_arrays(self):
        """
        Iterate through the data-flow, collecting mini-batches into arrays.

        Returns:
            tuple[np.ndarray]: The collected arrays.

        Raises:
            ValueError: If this data-flow is empty.
        """
        arrays_buf = []
        it = iter(self)
        try:
            batch = next(it)
        except StopIteration:
            raise ValueError('{!r} is empty, cannot convert to arrays'.
                             format(self))
        try:
            arrays_buf = [[arr] for arr in batch]
            while True:
                batch = next(it)
                for i, arr in enumerate(batch):
                    arrays_buf[i].append(arr)
        except StopIteration:
            pass
        return tuple(np.concatenate(arr) for arr in arrays_buf)

    def to_arrays_flow(self, batch_size, shuffle=False,
                       skip_incomplete=False, random_state=None):
        """
        Convert this data-flow to a :class:`~tfsnippet.dataflows.ArrayFlow`.

        This method will iterate through the data-flow, collecting mini-batches
        into arrays, and then construct an ArrayFlow.

        Args:
            batch_size (int): Size of each mini-batch.
            shuffle (bool): Whether or not to shuffle data before iterating?
                (default :obj:`False`)
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete? (default :obj:`False`)
            random_state (RandomState): Optional numpy RandomState for
                shuffling data before each epoch.  (default :obj:`None`,
                use the global :class:`RandomState`).

        Returns:
            tfsnippet.dataflow.ArrayFlow: The constructed ArrayFlow.
        """
        from .array_flow import ArrayFlow
        return ArrayFlow(self.get_arrays(), batch_size=batch_size,
                         shuffle=shuffle, skip_incomplete=skip_incomplete,
                         random_state=random_state)

    @property
    def current_batch(self):
        """
        Get the the result of current batch (last call to :meth:`next_batch()`,
        if the implicit iterator has been opened and the last call to
        :meth:`next_batch()` does not raise a :class:`StopIteration`).

        Returns:
            tuple[np.ndarray] or None: The arrays of current batch.
        """
        return self._current_batch

    def next_batch(self):
        """
        Get the arrays of next mini-batch from the implicit iterator.

        Returns:
            tuple[np.ndarray]: The arrays of mini-batch.

        Raises:
            StopIteration: If the implicit iterator is exhausted.
                Note that this error will only be triggered once at the
                end of an epoch.  The next time calling this method, a new
                epoch will be opened.
        """
        if self._implicit_iterator is None:
            self._implicit_iterator = iter(self)
        try:
            self._current_batch = next(self._implicit_iterator)
            return self._current_batch
        except StopIteration:
            self._implicit_iterator = None
            self._current_batch = None
            raise

    # -------- here starts the transforming methods --------
    def map(self, mapper, array_indices=None):
        """
        Construct a :class:`~tfsnippet.dataflows.MapperFlow`.

        Args:
            mapper ((\*np.ndarray) -> tuple[np.ndarray])): The mapper
                function, which transforms numpy arrays into a tuple
                of other numpy arrays.
            array_indices (int or Iterable[int]): The indices of the arrays
                to be processed within a mini-batch.

                If specified, will apply the mapper only on these selected
                arrays.  This will require the mapper to produce exactly
                the same number of output arrays as the inputs.

                If not specified, apply the mapper on all arrays, and do
                not require the number of output arrays to match the inputs.

        Returns:
            tfsnippet.dataflow.MapperFlow: The data flow with `mapper` applied.
        """
        from .mapper_flow import MapperFlow
        return MapperFlow(self, mapper, array_indices=array_indices)

    def threaded(self, prefetch):
        """
        Construct a :class:`~tfsnippet.dataflows.ThreadingFlow` from this flow.

        Args:
            prefetch (int): Number of mini-batches to prefetch ahead.
                It should be at least 1.

        Returns:
            tfsnippet.dataflow.ThreadingFlow: The background threaded
                data flow to prefetch mini-batches from this flow.
        """
        from .threading_flow import ThreadingFlow
        return ThreadingFlow(self, prefetch=prefetch)

    def select(self, indices):
        """
        Construct a :class:`DataFlow`, which selects and rearranges arrays
        in each mini-batch.  For example::

            flow = DataFlow.arrays([x, y, z], batch_size=64)
            flow.select([0, 2, 0])  # selects ``(x, z, x)`` in each mini-batch

        Args:
            indices (Iterable[int]): The indices of arrays to select.

        Returns:
            DataFlow: The data flow with selected arrays in each mini-batch.
        """
        indices = tuple(indices)
        return self.map(lambda *arrays: tuple(arrays[i] for i in indices))

    # -------- here starts the factory methods for data flows --------
    @staticmethod
    def gather(flows):
        """
        Gather multiple data flows into a single flow.

        Args:
            flows(Iterable[DataFlow]): The data flows to gather.
                At least one data flow should be specified, otherwise a
                :class:`ValueError` will be raised.

        Returns:
            tfsnippet.dataflow.GatherFlow: The gathered data flow.

        Raises:
            ValueError: If not even one data flow is specified.
            TypeError: If a specified flow is not a :class:`DataFlow`.
        """
        from .gather_flow import GatherFlow
        return GatherFlow(tuple(flows))

    @staticmethod
    def seq(start, stop, step=1, batch_size=None, shuffle=False,
            skip_incomplete=False, dtype=np.int32, random_state=None):
        """
        Construct a :class:`~tfsnippet.dataflows.SeqFlow`.

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

        Returns:
            tfsnippet.dataflow.SeqFlow: The data flow from number sequence.
        """
        from .seq_flow import SeqFlow
        return SeqFlow(
            start=start, stop=stop, step=step, batch_size=batch_size,
            shuffle=shuffle, skip_incomplete=skip_incomplete, dtype=dtype,
            random_state=random_state
        )

    @staticmethod
    def arrays(arrays, batch_size, shuffle=False, skip_incomplete=False,
               random_state=None):
        """
        Construct an :class:`~tfsnippet.dataflows.ArrayFlow`.

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

        Returns:
            tfsnippet.dataflow.ArrayFlow: The data flow from arrays.
        """
        from .array_flow import ArrayFlow
        return ArrayFlow(
            arrays=arrays, batch_size=batch_size, shuffle=shuffle,
            skip_incomplete=skip_incomplete, random_state=random_state
        )

    @staticmethod
    def iterator_factory(factory):
        """
        Construct a :class:`~tfsnippet.dataflows.IteratorFactoryFlow`.

        Args:
            factory (() -> Iterator or Iterable): A factory method for
                constructing the mini-batch iterators for each epoch.

        Returns:
            tfsnippet.dataflow.IteratorFactoryFlow: The data flow.
        """
        from .iterator_flow import IteratorFactoryFlow
        return IteratorFactoryFlow(factory)


class ExtraInfoDataFlow(DataFlow):
    """
    Base class for :class:`DataFlow` subclasses with auxiliary information
    about the mini-batches.
    """

    def __init__(self, array_count, data_length, data_shapes, batch_size,
                 skip_incomplete, is_shuffled):
        """
        Construct an :class:`ExtraInfoDataFlow`.

        Args:
            array_count (int): The count of arrays in each mini-batch.
            data_length (int): The total length of the data.
            data_shapes (tuple[tuple[int]]): The shapes of data in a
                mini-batch.  The batch dimension is not included.
            batch_size (int): Size of each mini-batch.
            skip_incomplete (bool): Whether or not to exclude the last
                mini-batch if it is incomplete?
            is_shuffled (bool): Whether or not the data are first shuffled
                before iterated through mini-batches?
        """
        self._array_count = array_count
        self._data_length = data_length
        self._data_shapes = data_shapes
        self._batch_size = batch_size
        self._skip_incomplete = skip_incomplete
        self._is_shuffled = is_shuffled

    @property
    def array_count(self):
        """
        Get the count of arrays in each mini-batch.

        Returns:
            int: The count of arrays in each mini-batch.
        """
        return self._array_count

    @property
    def data_length(self):
        """
        Get the total length of the data.

        Returns:
            int: The total length of the data.
        """
        return self._data_length

    @property
    def data_shapes(self):
        """
        Get the shapes of the data in each mini-batch.

        Returns:
            tuple[tuple[int]]: The shapes of data in a mini-batch.
                The batch dimension is not included.
        """
        return self._data_shapes

    @property
    def batch_size(self):
        """
        Get the size of each mini-batch.

        Returns:
            int: The size of each mini-batch.
        """
        return self._batch_size

    @property
    def skip_incomplete(self):
        """
        Whether or not to exclude the last mini-batch if it is incomplete?
        """
        return self._skip_incomplete

    @property
    def is_shuffled(self):
        """
        Whether or not the data are first shuffled before iterated through
        mini-batches?
        """
        return self._is_shuffled
