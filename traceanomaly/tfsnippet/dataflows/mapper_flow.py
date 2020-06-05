from .base import DataFlow

__all__ = ['MapperFlow']


class MapperFlow(DataFlow):
    """
    Data flow which transforms the mini-batch arrays from source flow
    by a specified mapper function.

    Usage::

        source_flow = Data.arrays([x, y], batch_size=256)
        mapper_flow = source_flow.map(lambda x, y: (x + y,))
    """

    def __init__(self, source, mapper, array_indices=None):
        """
        Construct a :class:`MapperFlow`.

        Args:
            source (DataFlow): The source data flow.
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
        """
        if array_indices is not None:
            try:
                array_indices = (int(array_indices),)
            except TypeError:
                array_indices = tuple(map(int, array_indices))
        self._source = source
        self._mapper = mapper
        self._array_indices = array_indices

    @property
    def source(self):
        """Get the source data flow."""
        return self._source

    @property
    def array_indices(self):
        """Get the indices of the arrays to be processed."""
        return self._array_indices

    def _validate_outputs(self, outputs):
        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif not isinstance(outputs, tuple):
            raise TypeError('The output of the mapper is expected to '
                            'be a tuple or a list, but got a {}.'.
                            format(outputs.__class__.__name__))
        return outputs

    def _minibatch_iterator(self):
        for batch in self._source:
            if self._array_indices is not None:
                mapped_b = list(batch)
                inputs = [mapped_b[i] for i in self._array_indices]
                outputs = self._validate_outputs(self._mapper(*inputs))
                if len(outputs) != len(inputs):
                    raise ValueError('The number of output arrays of the '
                                     'mapper is required to match the inputs, '
                                     'since `array_indices` is specified: '
                                     'outputs {} != inputs {}.'.
                                     format(len(outputs), len(inputs)))
                for i, o in zip(self._array_indices, outputs):
                    mapped_b[i] = o
                mapped_b = tuple(mapped_b)
            else:
                mapped_b = self._validate_outputs(self._mapper(*batch))
            yield mapped_b
