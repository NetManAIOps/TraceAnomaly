import numpy as np

from tfsnippet.ops import space_to_depth, depth_to_space
from tfsnippet.utils import (get_static_shape, get_shape, InputSpec,
                             reshape_tail, add_name_and_scope_arg_doc)
from .base import BaseFlow
from .utils import ZeroLogDet

__all__ = ['ReshapeFlow', 'SpaceToDepthFlow']


class ReshapeFlow(BaseFlow):
    """
    A flow which reshapes the last `x_value_ndims` of `x` into `y_value_shape`.

    Usage::

        # to reshape a conv2d output into dense input
        flow = ReshapeFlow(x_value_ndims=3, y_value_shape=[-1])
        x = tf.random_normal(shape=[2, 3, 4, 5])
        y, log_det = flow.transform(x)

        # y == tf.reshape(x, [2, -1])
        # log_det == tf.zeros([2])
    """

    @add_name_and_scope_arg_doc
    def __init__(self, x_value_ndims, y_value_shape, require_batch_dims=False,
                 name=None, scope=None):
        """
        Construct a new :class:`ReshapeFlow`.

        Args:
            x_value_ndims (int): Number of value dimensions in `x`.
                `x.ndims - x_value_ndims == log_det.ndims`.
            y_value_shape (Iterable[int]): The value shape of `y`.
                May contain one un-deterministic dimension `-1`.
            require_batch_dims (bool): If :obj:`True`, `x` are required
                to have at least `x_value_ndims + 1` dimensions, and `y`
                are required to have at least `y_value_ndims + 1` dimensions.

                If :obj:`False`, `x` are required to have at least
                `x_value_ndims` dimensions, and `y` are required to have
                at least `y_value_ndims` dimensions.
        """
        y_value_shape = tuple(int(s) for s in y_value_shape)
        neg_one_count = 0
        for s in y_value_shape:
            if s <= 0:
                if s == -1:
                    if neg_one_count > 0:
                        raise ValueError('`shape` is not a valid shape: at '
                                         'most one `-1` can be specified.')
                    else:
                        neg_one_count += 1
                else:
                    raise ValueError('`shape` is not a valid shape: {} is '
                                     'not allowed.'.format(s))

        self._x_value_shape = None
        self._y_value_shape = y_value_shape

        super(ReshapeFlow, self).__init__(
            x_value_ndims=x_value_ndims,
            y_value_ndims=len(y_value_shape),
            require_batch_dims=require_batch_dims,
            name=name,
            scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        shape = get_static_shape(input)
        dtype = input.dtype.base_dtype
        assert(shape is not None and len(shape) >= self.x_value_ndims)

        # re-build the x input spec
        x_shape_spec = []
        if self.x_value_ndims > 0:
            x_shape_spec = list(shape)[-self.x_value_ndims:]
        if self.require_batch_dims:
            x_shape_spec = ['?'] + x_shape_spec
        x_shape_spec = ['...'] + x_shape_spec
        self._x_input_spec = InputSpec(shape=x_shape_spec, dtype=dtype)

        # infer the dynamic value shape of x, and store it for inverse transform
        x_value_shape = []
        if self.x_value_ndims > 0:
            x_value_shape = list(shape)[-self.x_value_ndims:]

        neg_one_count = 0
        for i, s in enumerate(x_value_shape):
            if s is None:
                if neg_one_count > 0:
                    x_value_shape = get_shape(input)
                    if self.x_value_ndims > 0:
                        x_value_shape = x_value_shape[-self.x_value_ndims:]
                    break
                else:
                    x_value_shape[i] = -1
                    neg_one_count += 1

        self._x_value_shape = x_value_shape

        # now infer the y value shape according to new info obtained from x
        y_value_shape = list(self._y_value_shape)
        if isinstance(x_value_shape, list) and -1 not in x_value_shape:
            x_value_size = int(np.prod(x_value_shape))
            y_value_size = int(np.prod([s for s in y_value_shape if s != -1]))

            if (-1 in y_value_shape and x_value_size % y_value_size != 0) or \
                    (-1 not in y_value_shape and x_value_size != y_value_size):
                raise ValueError(
                    'Cannot reshape the tail dimensions of `x` into `y`: '
                    'x value shape {!r}, y value shape {!r}.'.
                    format(x_value_shape, y_value_shape)
                )

            if -1 in y_value_shape:
                y_value_shape[y_value_shape.index(-1)] = \
                    x_value_size // y_value_size

            assert(-1 not in y_value_shape)
            self._y_value_shape = tuple(y_value_shape)

        # re-build the y input spec
        y_shape_spec = list(y_value_shape)
        if self.require_batch_dims:
            y_shape_spec = ['?'] + y_shape_spec
        y_shape_spec = ['...'] + y_shape_spec
        self._y_input_spec = InputSpec(shape=y_shape_spec, dtype=dtype)

    def _transform(self, x, compute_y, compute_log_det):
        assert (len(get_static_shape(x)) >= self.x_value_ndims)

        # compute y
        y = None
        if compute_y:
            y = reshape_tail(x, self.x_value_ndims, self._y_value_shape)

        # compute log_det
        log_det = None
        if compute_log_det:
            dst_shape = get_shape(x)
            if self.x_value_ndims > 0:
                dst_shape = dst_shape[:-self.x_value_ndims]
            log_det = ZeroLogDet(dst_shape, dtype=x.dtype.base_dtype)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        assert (len(get_static_shape(y)) >= self.y_value_ndims)

        # compute y
        x = None
        if compute_x:
            x = reshape_tail(y, self.y_value_ndims, self._x_value_shape)

        # compute log_det
        log_det = None
        if compute_log_det:
            dst_shape = get_shape(y)
            if self.y_value_ndims > 0:
                dst_shape = dst_shape[:-self.y_value_ndims]
            log_det = ZeroLogDet(dst_shape, dtype=y.dtype.base_dtype)

        return x, log_det


class SpaceToDepthFlow(BaseFlow):
    """
    A flow which computes ``y = space_to_depth(x)``, and conversely
    ``x = depth_to_space(y)``.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, block_size, channels_last=True, name=None, scope=None):
        """
        Construct a new :class:`SpaceToDepthFlow`.

        Args:
            block_size (int): An int >= 2, the size of the spatial block.
            channels_last (bool): Whether or not the channels axis
                is the last axis in the input tensor?
        """
        block_size = int(block_size)
        if block_size < 2:
            raise ValueError('`block_size` must be at least 2.')

        self._block_size = block_size
        self._channels_last = bool(channels_last)
        super(SpaceToDepthFlow, self).__init__(
            x_value_ndims=3, y_value_ndims=3, require_batch_dims=True,
            name=name, scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        # TODO: maybe add more shape check here.
        pass

    def _transform(self, x, compute_y, compute_log_det):
        # compute y
        y = None
        if compute_y:
            y = space_to_depth(x, block_size=self._block_size,
                               channels_last=self._channels_last)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(shape=get_shape(x)[:-3],
                                 dtype=x.dtype.base_dtype)

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # compute x
        x = None
        if compute_x:
            x = depth_to_space(y, block_size=self._block_size,
                               channels_last=self._channels_last)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = ZeroLogDet(shape=get_shape(y)[:-3],
                                 dtype=y.dtype.base_dtype)

        return x, log_det
