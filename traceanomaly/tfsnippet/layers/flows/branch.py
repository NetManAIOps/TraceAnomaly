import tensorflow as tf

from tfsnippet.utils import get_static_shape, InputSpec, get_shape, assert_deps
from .base import BaseFlow, sum_log_det

__all__ = ['SplitFlow']


class SplitFlow(BaseFlow):
    """
    A flow which splits input `x` into halves, apply different flows on each
    half, then concat the output together.

    Basically, a :class:`SplitFlow` performs the following transformation::

        x1, x2 = split(x, axis=split_axis)
        y1, log_det1 = left.transform(x1)
        if right is not None:
            y2, log_det2 = right.transform(x2)
        else:
            y2, log_det2 = x2, 0.
        y = concat([y1, y2], axis=join_axis)
        log_det = log_det1 + log_det2
    """

    def __init__(self, split_axis, left, join_axis=None, right=None,
                 name=None, scope=None):
        """
        Construct a new :class:`SplitFlow`.

        Args:
            split_axis (int): Along which axis to split `x`.
            left (BaseFlow): The `left` flow (see above).
            join_axis (int): Along which axis to join `y`.
                If not specified, use `split_axis`.
                Must be specified if `left.x_value_ndims != left.y_value_ndims`.
            right (BaseFlow): The `right` flow (see above).
                `right.x_value_ndims` must equal to `left.x_value_ndims`, and
                `right.y_value_ndims` must equal to `left.y_value_ndims`.
                If not specified, the right flow will be identity.
                Must be specified if `left.x_value_ndims != left.y_value_ndims`.
        """
        split_axis = int(split_axis)
        if join_axis is not None:
            join_axis = int(join_axis)

        if not isinstance(left, BaseFlow):
            raise TypeError('`left` must be an instance of `BaseFlow`: '
                            'got {!r}.'.format(left))
        x_value_ndims = left.x_value_ndims
        y_value_ndims = left.y_value_ndims

        if right is not None:
            if not isinstance(right, BaseFlow):
                raise TypeError('`right` must be an instance of `BaseFlow`: '
                                'got {!r}.'.format(right))
            if right.x_value_ndims != x_value_ndims or \
                    right.y_value_ndims != y_value_ndims:
                raise ValueError('`left` and `right` must have same '
                                 '`x_value_ndims` and `y_value_ndims`: '
                                 'left {!r} vs right {!r}.'.format(left, right))

        if x_value_ndims != y_value_ndims:
            if join_axis is None:
                raise ValueError('`x_value_ndims` != `y_value_ndims`, thus '
                                 '`join_axis` must be specified.')

            if right is None:
                raise ValueError('`x_value_ndims` != `y_value_ndims`, thus '
                                 '`right` must be specified.')

        self._split_axis = split_axis
        self._join_axis = join_axis
        self._left = left
        self._right = right
        self._x_n_left = self._x_n_right = None
        self._y_n_left = self._y_n_right = None

        super(SplitFlow, self).__init__(
            x_value_ndims=x_value_ndims, y_value_ndims=y_value_ndims,
            require_batch_dims=False, name=name, scope=scope
        )

    @property
    def explicitly_invertible(self):
        return True

    def _build(self, input=None):
        shape = get_static_shape(input)
        dtype = input.dtype.base_dtype

        # resolve the split axis
        split_axis = self._split_axis
        if split_axis < 0:
            split_axis += len(shape)
        if split_axis < 0 or split_axis < len(shape) - self.x_value_ndims:
            raise ValueError(
                '`split_axis` out of range, or not covered by `x_value_ndims`: '
                'split_axis {}, x_value_ndims {}, input {}'.
                format(self._split_axis, self.x_value_ndims, input)
            )
        split_axis -= len(shape)

        err_msg = (
            'The split axis of `input` must be at least 2: input {}, axis {}.'.
            format(input, split_axis)
        )
        if shape[split_axis] is not None:
            x_n_features = shape[split_axis]
            if x_n_features < 2:
                raise ValueError(err_msg)
            x_n_left = x_n_features // 2
            x_n_right = x_n_features - x_n_left
        else:
            x_n_features = get_shape(input)[split_axis]
            x_n_left = x_n_features // 2
            x_n_right = x_n_features - x_n_left

            with assert_deps([
                        tf.assert_greater_equal(
                            x_n_features, 2, message=err_msg)
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    x_n_left = tf.identity(x_n_left)
                    x_n_right = tf.identity(x_n_right)

            x_n_features = None

        self._split_axis = split_axis
        self._x_n_left = x_n_left
        self._x_n_right = x_n_right

        # build the x spec
        shape_spec = ['?'] * self.x_value_ndims
        if x_n_features is not None:
            shape_spec[self._split_axis] = x_n_features
        assert(not self.require_batch_dims)
        shape_spec = ['...'] + shape_spec
        self._x_input_spec = InputSpec(shape=shape_spec, dtype=dtype)

    def _transform(self, x, compute_y, compute_log_det):
        # split the input x
        x1, x2 = tf.split(x, [self._x_n_left, self._x_n_right],
                          axis=self._split_axis)
        do_compute_y = compute_y or (self._y_n_left is None)

        # apply the left transformation
        y1, log_det1 = self._left.transform(
            x1, compute_y=do_compute_y, compute_log_det=compute_log_det)

        # apply the right transformation
        if self._right is not None:
            y2, log_det2 = self._right.transform(
                x2, compute_y=do_compute_y, compute_log_det=compute_log_det)
        else:
            y2, log_det2 = x2, None

        # check the outputs
        y1_shape = get_static_shape(y1)
        y2_shape = get_static_shape(y2)

        if len(y1_shape) != len(y2_shape):
            raise RuntimeError('`y_left.ndims` != `y_right.ndims`: y_left {} '
                               'vs y_right {}'.format(y1, y2))

        # build the y spec if not built
        join_axis = self._join_axis

        if self._y_n_left is None:
            # resolve the join axis
            if join_axis is None:
                join_axis = self._split_axis
            if join_axis < 0:
                join_axis += len(y1_shape)

            if join_axis < 0 or join_axis < len(y1_shape) - self.y_value_ndims:
                raise ValueError(
                    '`join_axis` out of range, or not covered by `y_value_ndims'
                    '`: split_axis {}, y_value_ndims {}, y_left {}, y_right {}'.
                    format(self._split_axis, self.y_value_ndims, y1, y2)
                )
            join_axis -= len(y1_shape)

            err_msg = (
                '`y_left.shape[join_axis] + y_right.shape[join_axis]` must '
                'be at least 2: y_left {}, y_right {} axis {}.'.
                format(y1, y2, join_axis)
            )
            y_n_left = y1_shape[join_axis]
            y_n_right = y2_shape[join_axis]
            if y_n_left is not None and y_n_right is not None:
                y_n_features = y_n_left + y_n_right
                assert(y_n_features >= 2)
            else:
                y_n_left = get_shape(y1)[join_axis]
                y_n_right = get_shape(y2)[join_axis]
                y_n_features = None
                with assert_deps([
                            tf.assert_greater_equal(
                                y_n_left + y_n_right, 2, message=err_msg)
                        ]) as asserted:
                    if asserted:  # pragma: no cover
                        y_n_left = tf.identity(y_n_left)
                        y_n_right = tf.identity(y_n_right)

            self._join_axis = join_axis
            self._y_n_left = y_n_left
            self._y_n_right = y_n_right

            # build the y spec
            dtype = self._x_input_spec.dtype
            y_shape_spec = ['?'] * self.y_value_ndims
            if y_n_features is not None:
                y_shape_spec[self._split_axis] = y_n_features
            assert(not self.require_batch_dims)
            y_shape_spec = ['...'] + y_shape_spec
            self._y_input_spec = InputSpec(shape=y_shape_spec, dtype=dtype)

        assert(join_axis is not None)

        # compute y
        y = None
        if compute_y:
            y = tf.concat([y1, y2], axis=join_axis)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = log_det1
            if log_det2 is not None:
                log_det = sum_log_det([log_det, log_det2])

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        assert(self._y_n_left is not None)

        # split the y
        y1, y2 = tf.split(y, [self._y_n_left, self._y_n_right],
                          axis=self._join_axis)

        # apply the left transformation
        x1, log_det1 = self._left.inverse_transform(
            y1, compute_x=compute_x, compute_log_det=compute_log_det)

        # apply the right transformation
        if self._right is not None:
            x2, log_det2 = self._right.inverse_transform(
                y2, compute_x=compute_x, compute_log_det=compute_log_det)
        else:
            x2, log_det2 = y2, None

        # compute x
        x = None
        if compute_x:
            x = tf.concat([x1, x2], axis=self._split_axis)

        # compute log_det
        log_det = None
        if compute_log_det:
            log_det = log_det1
            if log_det2 is not None:
                log_det = sum_log_det([log_det, log_det2])

        return x, log_det
