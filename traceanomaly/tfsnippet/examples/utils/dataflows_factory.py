import numpy as np

from tfsnippet.dataflows import DataFlow
from tfsnippet.preprocessing import BernoulliSampler, UniformNoiseSampler

__all__ = ['bernoulli_flow', 'quantized_flow']


def _create_sampled_dataflow(arrays, sampler, sample_now, **kwargs):
    if sample_now:
        arrays = sampler(*arrays)
    df = DataFlow.arrays(arrays, **kwargs)
    if not sample_now:
        df = df.map(sampler)
    return df


def bernoulli_flow(x, batch_size, shuffle=False, skip_incomplete=False,
                   sample_now=False, dtype=np.int32, random_state=None):
    """
    Construct a new :class:`DataFlow`, which samples 0/1 binary images
    according to the given `x` array.

    Args:
        x: The `train_x` or `test_x` of an image dataset.  The pixel values
            must be 8-bit integers, having the range of ``[0, 255]``.
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether or not to shuffle data before iterating?
            (default :obj:`False`)
        skip_incomplete (bool): Whether or not to exclude the last
            mini-batch if it is incomplete? (default :obj:`False`)
        sample_now (bool): Whether or not to sample immediately instead
            of sampling at the beginning of each epoch? (default :obj:`False`)
        dtype: The data type of the sampled array.  Default `np.int32`.
        random_state (RandomState): Optional numpy RandomState for
            shuffling data before each epoch.  (default :obj:`None`,
            use the global :class:`RandomState`).

    Returns:
        DataFlow: The Bernoulli `x` flow.
    """
    x = np.asarray(x)

    # prepare the sampler
    x = x / np.asarray(255., dtype=x.dtype)
    sampler = BernoulliSampler(dtype=dtype, random_state=random_state)

    # compose the data flow
    return _create_sampled_dataflow(
        [x], sampler, sample_now, batch_size=batch_size, shuffle=shuffle,
        skip_incomplete=skip_incomplete, random_state=random_state
    )


def quantized_flow(x, batch_size, shuffle=False, skip_incomplete=False,
                   normalize=False, sample_now=False, dtype=np.float32,
                   random_state=None):
    """
    Construct a new :class:`DataFlow`, which adds uniform noises onto
    the given `x` array.

    Args:
        x: The `train_x` or `test_x` of an image dataset.  The pixel values
            must be 8-bit integers, having the range of ``[0, 255]``.
        batch_size (int): Size of each mini-batch.
        shuffle (bool): Whether or not to shuffle data before iterating?
            (default :obj:`False`)
        skip_incomplete (bool): Whether or not to exclude the last
            mini-batch if it is incomplete? (default :obj:`False`)
        normalize (bool): Whether or not to normalize the sampled array?
            If :obj:`True`, the sampled array would range in ``[0, 1)``.
            If :obj:`True`, the sampled array would range in ``[0, 256)``.
            Default :obj:`True`.
        sample_now (bool): Whether or not to sample immediately instead
            of sampling at the beginning of each epoch? (default :obj:`False`)
        dtype: The data type of the sampled array.  Default `np.float32`.
        random_state (RandomState): Optional numpy RandomState for
            shuffling data before each epoch.  (default :obj:`None`,
            use the global :class:`RandomState`).

    Returns:
        DataFlow: The quantized `x` flow.
    """
    x = np.asarray(x)

    # prepare the sampler
    if normalize:
        x = x / np.asarray(256., dtype=x.dtype)
        maxval = np.asarray(1 / 256., dtype=x.dtype)
    else:
        maxval = np.asarray(1., dtype=x.dtype)
    minval = np.asarray(0., dtype=x.dtype)
    sampler = UniformNoiseSampler(minval=minval, maxval=maxval, dtype=dtype,
                                  random_state=random_state)

    # compose the data flow
    return _create_sampled_dataflow(
        [x], sampler, sample_now, batch_size=batch_size, shuffle=shuffle,
        skip_incomplete=skip_incomplete, random_state=random_state
    )
