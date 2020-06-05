import functools

import numpy as np
import tensorflow as tf

from .debugging import assert_deps
from .doc_utils import add_name_arg_doc
from .type_utils import is_tensor_object

__all__ = [
    'get_static_shape', 'get_batch_size', 'get_rank', 'get_shape',
    'get_dimensions_size',
    'flatten_to_ndims', 'unflatten_from_ndims',
    'resolve_negative_axis',  'concat_shapes', 'is_shape_equal',
    'broadcast_to_shape', 'broadcast_to_shape_strict',
    'transpose_conv2d_axis', 'transpose_conv2d_channels_last_to_x',
    'transpose_conv2d_channels_x_to_last',
    'reshape_tail',
]


def get_static_shape(tensor):
    """
    Get the the static shape of specified `tensor` as a tuple.

    Args:
        tensor: The tensor object.

    Returns:
        tuple[int or None] or None: The static shape tuple, or :obj:`None`
            if the dimensions of `tensor` is not deterministic.
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tensor.get_shape()
    if shape.ndims is None:
        shape = None
    else:
        shape = tuple((int(v) if v is not None else None)
                      for v in shape.as_list())
    return shape


def resolve_negative_axis(ndims, axis):
    """
    Resolve all negative `axis` indices according to `ndims` into positive.

    Usage::

        resolve_negative_axis(4, [0, -1, -2])  # output: (0, 3, 2)

    Args:
        ndims (int): Number of total dimensions.
        axis (Iterable[int]): The axis indices.

    Returns:
        tuple[int]: The resolved positive axis indices.

    Raises:
        ValueError: If any index in `axis` is out of range.
    """
    axis = tuple(int(a) for a in axis)
    ret = []
    for a in axis:
        if a < 0:
            a += ndims
        if a < 0 or a >= ndims:
            raise ValueError('`axis` out of range: {} vs ndims {}.'.
                             format(axis, ndims))
        ret.append(a)
    if len(set(ret)) != len(ret):
        raise ValueError('`axis` has duplicated elements after resolving '
                         'negative axis: ndims {}, axis {}.'.
                         format(ndims, axis))
    return tuple(ret)


@add_name_arg_doc
def flatten_to_ndims(x, ndims, name=None):
    """
    Flatten the front dimensions of `x`, such that the resulting tensor
    will have at most `ndims` dimensions.

    Args:
        x (Tensor): The tensor to be flatten.
        ndims (int): The maximum number of dimensions for the resulting tensor.

    Returns:
        (tf.Tensor, tuple[int or None], tuple[int] or tf.Tensor) or (tf.Tensor, None, None):
            (The flatten tensor, the static front shape, and the front shape),
            or (the original tensor, None, None)
    """
    x = tf.convert_to_tensor(x)
    if ndims < 1:
        raise ValueError('`k` must be greater or equal to 1.')
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < ndims:
        raise ValueError('`k` is {}, but `x` only has rank {}.'.
                         format(ndims, len(shape)))
    if len(shape) == ndims:
        return x, None, None

    with tf.name_scope(name, default_name='flatten', values=[x]):
        if ndims == 1:
            static_shape = shape
            if None in shape:
                shape = tf.shape(x)
            return tf.reshape(x, [-1]), static_shape, shape
        else:
            front_shape, back_shape = shape[:-(ndims - 1)], shape[-(ndims - 1):]
            static_front_shape = front_shape
            static_back_shape = back_shape
            if None in front_shape or None in back_shape:
                dynamic_shape = tf.shape(x)
                if None in front_shape:
                    front_shape = dynamic_shape[:-(ndims - 1)]
                if None in back_shape:
                    back_shape = dynamic_shape[-(ndims - 1):]
            if isinstance(back_shape, tuple):
                x = tf.reshape(x, [-1] + list(back_shape))
            else:
                x = tf.reshape(x, tf.concat([[-1], back_shape], axis=0))
                x.set_shape(tf.TensorShape([None] + list(static_back_shape)))
            return x, static_front_shape, front_shape


@add_name_arg_doc
def unflatten_from_ndims(x, static_front_shape, front_shape, name=None):
    """
    The inverse transformation of :func:`flatten`.

    If both `static_front_shape` is None and `front_shape` is None,
    `x` will be returned without any change.

    Args:
        x (Tensor): The tensor to be unflatten.
        static_front_shape (tuple[int or None] or None): The static front shape.
        front_shape (tuple[int] or tf.Tensor or None): The front shape.

    Returns:
        tf.Tensor: The unflatten x.
    """
    x = tf.convert_to_tensor(x)
    if static_front_shape is None and front_shape is None:
        return x
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < 1:
        raise ValueError('`x` only has rank {}, required at least 1.'.
                         format(len(shape)))
    if not is_tensor_object(front_shape):
        front_shape = tuple(front_shape)

    with tf.name_scope(name, default_name='unflatten', values=[x]):
        back_shape = shape[1:]
        static_back_shape = back_shape
        if None in back_shape:
            back_shape = tf.shape(x)[1:]
        if isinstance(front_shape, tuple) and isinstance(back_shape, tuple):
            x = tf.reshape(x, front_shape + back_shape)
        else:
            x = tf.reshape(x, tf.concat([front_shape, back_shape], axis=0))
            x.set_shape(tf.TensorShape(list(static_front_shape) +
                                       list(static_back_shape)))
        return x


@add_name_arg_doc
def get_batch_size(tensor, axis=0, name=None):
    """
    Infer the mini-batch size according to `tensor`.

    Args:
        tensor (tf.Tensor): The input placeholder.
        axis (int): The axis of mini-batches.  Default is 0.

    Returns:
        int or tf.Tensor: The batch size.
    """
    tensor = tf.convert_to_tensor(tensor)
    axis = int(axis)
    with tf.name_scope(name, default_name='get_batch_size', values=[tensor]):
        batch_size = None
        shape = get_static_shape(tensor)
        if shape is not None:
            batch_size = shape[axis]
        if batch_size is None:
            batch_size = tf.shape(tensor)[axis]
    return batch_size


@add_name_arg_doc
def get_rank(tensor, name=None):
    """
    Get the rank of the tensor.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        name: TensorFlow name scope of the graph nodes.

    Returns:
        int or tf.Tensor: The rank.
    """
    tensor_shape = get_static_shape(tensor)
    if tensor_shape is not None:
        return len(tensor_shape)
    return tf.rank(tensor, name=name)


@add_name_arg_doc
def get_dimensions_size(tensor, axis=None, name=None):
    """
    Get the size of `tensor` of specified `axis`.

    If `axis` is :obj:`None`, select the size of all dimensions.

    Args:
        tensor (tf.Tensor): The tensor to be tested.
        axis (Iterable[int] or None): The dimensions to be selected.

    Returns:
        tuple[int] or tf.Tensor: A tuple of integers if all selected
            dimensions have static sizes.  Otherwise a tensor.
    """
    tensor = tf.convert_to_tensor(tensor)
    if axis is not None:
        axis = tuple(axis)
        if not axis:
            return ()

    with tf.name_scope(name, default_name='get_dimensions_size',
                       values=[tensor]):
        shape = get_static_shape(tensor)

        if shape is not None and axis is not None:
            shape = tuple(shape[a] for a in axis)

        if shape is None or None in shape:
            dynamic_shape = tf.shape(tensor)
            if axis is None:
                shape = dynamic_shape
            else:
                shape = tf.stack([dynamic_shape[i] for i in axis], axis=0)

        return shape


get_shape = functools.partial(get_dimensions_size, axis=None)


@add_name_arg_doc
def concat_shapes(shapes, name=None):
    """
    Concat shapes from `shapes`.

    Args:
        shapes (Iterable[tuple[int] or tf.Tensor]): List of shape tuples
            or tensors.

    Returns:
        tuple[int] or tf.Tensor: The concatenated shape.
    """
    shapes = tuple(shapes)
    if any(is_tensor_object(s) for s in shapes):
        shapes = [
            s if is_tensor_object(s) else tf.constant(s, dtype=tf.int32)
            for s in shapes
        ]
        with tf.name_scope(name, default_name='concat_shapes', values=shapes):
            return tf.concat(shapes, axis=0)
    else:
        return sum((tuple(s) for s in shapes), ())


@add_name_arg_doc
def is_shape_equal(x, y, name=None):
    """
    Check whether the shape of `x` equals to `y`.

    Args:
        x: A tensor.
        y: Another tensor, to compare with `x`.

    Returns:
        bool or tf.Tensor: The static or dynamic comparison result.
    """
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    x_shape = get_static_shape(x)
    y_shape = get_static_shape(y)

    # both shapes have deterministic dimensions, we can perform a fast check
    if x_shape is not None and y_shape is not None:
        # dimension mismatch, cannot be equal
        if len(x_shape) != len(y_shape):
            return False

        # gather the axis to check
        axis_to_check = []
        for i, (a, b) in enumerate(zip(x_shape, y_shape)):
            if a is None or b is None:
                axis_to_check.append(i)
            else:
                if a != b:
                    return False

        # no dynamic axis to check, confirm equality
        if not axis_to_check:
            return True

        # generate the dynamic check
        with tf.name_scope(name or 'is_shape_equal', values=[x, y]):
            x_shape = get_shape(x)
            y_shape = get_shape(y)
            return tf.reduce_all([tf.equal(x_shape[a], y_shape[a])
                                  for a in axis_to_check])

    # either one of the shapes has non-deterministic dimensions
    with tf.name_scope(name or 'is_shape_equal', values=[x, y]):
        x_shape = get_shape(x)
        y_shape = get_shape(y)
        return tf.cond(
            tf.equal(tf.rank(x), tf.rank(y)),
            lambda: tf.reduce_all(
                tf.equal(
                    tf.concat([x_shape, y_shape], axis=0),
                    tf.concat([y_shape, x_shape], axis=0)
                )
            ),
            lambda: tf.constant(False, dtype=tf.bool)
        )


def broadcast_to_shape(x, shape, name=None):
    """
    Broadcast `x` to match `shape`.

    If ``rank(x) > len(shape)``, only the tail dimensions will be broadcasted
    to match `shape`.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    from tfsnippet.ops import smart_cond

    # check the parameters
    x = tf.convert_to_tensor(x)
    x_shape = get_static_shape(x)
    ns_values = [x]
    if is_tensor_object(shape):
        shape = tf.convert_to_tensor(shape)
        ns_values.append(shape)
    else:
        shape = tuple(int(s) for s in shape)

    with tf.name_scope(name=name or 'broadcast_to_shape', values=ns_values):
        cannot_broadcast_msg = (
            '`x` cannot be broadcasted to match `shape`: x {!r} vs shape {!r}'.
            format(x, shape)
        )

        # fast routine: shape is tuple[int] and x_shape is all known,
        # we can use reshape + tile to do the broadcast, which should be faster
        # than using ``x * ones(shape)``.
        if isinstance(shape, tuple) and x_shape is not None and \
                all(s is not None for s in x_shape):
            # reshape to have the same dimension
            if len(x_shape) < len(shape):
                x_shape = (1,) * (len(shape) - len(x_shape)) + x_shape
                x = tf.reshape(x, x_shape)

            # tile to have the same shape
            tile = []
            i = -1
            while i > -len(shape) - 1:
                a, b = x_shape[i], shape[i]
                if a == 1 and b > 1:
                    tile.append(b)
                elif a != b:
                    raise ValueError(cannot_broadcast_msg)
                else:
                    tile.append(1)
                i -= 1
            tile = [1] * (len(x_shape) - len(shape)) + list(reversed(tile))
            if any(s > 1 for s in tile):
                x = tf.tile(x, tile)

            return x

        # slow routine: we may need ``x * ones(shape)`` to do the broadcast
        assertions = []
        post_assert_shape = False
        static_shape = tf.TensorShape(None)

        if isinstance(shape, tuple) and x_shape is not None:
            need_multiply_ones = False

            # it should always broadcast if len(x_shape) < len(shape)
            if len(x_shape) < len(shape):
                need_multiply_ones = True

            # check the consistency of x and shape
            static_shape_hint = []  # list to gather the static shape hint
            axis_to_check = []  # list to gather the axis to check
            i = -1
            while i >= -len(shape) and i >= -len(x_shape):
                a, b = x_shape[i], shape[i]
                if a is None:
                    axis_to_check.append(i)
                else:
                    if a != b:
                        if a == 1:
                            need_multiply_ones = True
                        else:
                            raise ValueError(cannot_broadcast_msg)
                static_shape_hint.append(b)
                i -= 1

            # compose the static shape hint
            if len(shape) < len(x_shape):
                static_shape = x_shape[:-len(shape)]
            elif len(shape) > len(x_shape):
                static_shape = shape[:-len(x_shape)]
            else:
                static_shape = ()
            static_shape = tf.TensorShape(
                static_shape + tuple(reversed(static_shape_hint)))

            # compose the assertion operations and the multiply flag
            if axis_to_check:
                need_multiply_flags = []
                x_dynamic_shape = tf.shape(x)

                for i in axis_to_check:
                    assertions.append(tf.assert_equal(
                        tf.logical_or(
                            tf.equal(x_dynamic_shape[i], shape[i]),
                            tf.equal(x_dynamic_shape[i], 1),
                        ),
                        True,
                        message=cannot_broadcast_msg
                    ))
                    if len(x_shape) >= len(shape):
                        need_multiply_flags.append(
                            tf.not_equal(x_dynamic_shape[i], shape[i]))

                if not need_multiply_ones:
                    need_multiply_ones = \
                        tf.reduce_any(tf.stack(need_multiply_flags))

        else:
            # we have no ideal about what `shape` is here, thus we need to
            # assert the shape after ``x * ones(shape)``.
            need_multiply_ones = True
            post_assert_shape = True

        # do broadcast if `x_shape` != `shape`
        def multiply_branch():
            with assert_deps(assertions):
                ones_template = tf.ones(shape, dtype=x.dtype.base_dtype)
            try:
                return x * ones_template
            except ValueError:  # pragma: no cover
                raise ValueError(cannot_broadcast_msg)

        def identity_branch():
            with assert_deps(assertions) as asserted:
                if asserted:
                    return tf.identity(x)
                else:  # pragma: no cover
                    return x

        t = smart_cond(need_multiply_ones, multiply_branch, identity_branch)
        t.set_shape(static_shape)

        if post_assert_shape:
            post_assert_op = tf.assert_equal(
                tf.reduce_all(tf.equal(tf.shape(t)[-tf.size(shape):], shape)),
                True,
                message=cannot_broadcast_msg
            )
            with assert_deps([post_assert_op]) as asserted:
                if asserted:
                    t = tf.identity(t)

        return t


@add_name_arg_doc
def broadcast_to_shape_strict(x, shape, name=None):
    """
    Broadcast `x` to match `shape`.

    This method requires `rank(x)` to be less than or equal to `len(shape)`.
    You may use :func:`broadcast_to_shape` instead, to allow the cases where
    ``rank(x) > len(shape)``.

    Args:
        x: A tensor.
        shape (tuple[int] or tf.Tensor): Broadcast `x` to match this shape.

    Returns:
        tf.Tensor: The broadcasted tensor.
    """
    from tfsnippet.ops import assert_rank

    # check the parameters
    x = tf.convert_to_tensor(x)
    x_shape = get_static_shape(x)
    ns_values = [x]
    if is_tensor_object(shape):
        shape = tf.convert_to_tensor(shape)
        ns_values.append(shape)
    else:
        shape = tuple(int(s) for s in shape)

    with tf.name_scope(name=name or 'broadcast_to_shape', values=ns_values):
        cannot_broadcast_msg = (
            '`x` cannot be broadcasted to match `shape`: x {!r} vs shape {!r}'.
            format(x, shape)
        )

        # assert ``rank(x) <= len(shape)``
        if isinstance(shape, tuple) and x_shape is not None:
            if len(x_shape) > len(shape):
                raise ValueError(cannot_broadcast_msg)
        elif isinstance(shape, tuple):
            with assert_deps([
                        tf.assert_less_equal(
                            tf.rank(x),
                            len(shape),
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    x = tf.identity(x)
        else:
            with assert_deps([
                        assert_rank(
                            shape,
                            1,
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    shape = tf.identity(shape)

            with assert_deps([
                        tf.assert_less_equal(
                            tf.rank(x),
                            tf.size(shape),
                            message=cannot_broadcast_msg
                        )
                    ]) as asserted:
                if asserted:  # pragma: no cover
                    x = tf.identity(x)

        # do broadcast
        return broadcast_to_shape(x, shape)


@add_name_arg_doc
def transpose_conv2d_axis(input, from_channels_last, to_channels_last,
                          name=None):
    """
    Ensure the channels axis of `input` tensor to be placed at the desired axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        from_channels_last (bool): Whether or not the channels axis
            is the last axis in `input`? (i.e., the data format is "NHWC")
        to_channels_last (bool): Whether or not the channels axis
            should be the last axis in the output tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    from .tensor_spec import InputSpec
    if from_channels_last:
        input_spec = InputSpec(shape=('...', '?', '?', '?', '*'))
    else:
        input_spec = InputSpec(shape=('...', '?', '*', '?', '?'))
    input = input_spec.validate('input', input)
    input_shape = get_static_shape(input)
    sample_and_batch_axis = [i for i in range(len(input_shape) - 3)]

    # check whether or not axis should be transpose
    if from_channels_last and not to_channels_last:
        transpose_axis = [-1, -3, -2]
    elif not from_channels_last and to_channels_last:
        transpose_axis = [-2, -1, -3]
    else:
        transpose_axis = None

    # transpose the axis
    if transpose_axis is not None:
        transpose_axis = [i + len(input_shape) for i in transpose_axis]
        input = tf.transpose(input, sample_and_batch_axis + transpose_axis,
                             name=name or 'transpose_conv2d_axis')

    return input


@add_name_arg_doc
def transpose_conv2d_channels_last_to_x(input, channels_last, name=None):
    """
    Ensure the channels axis (known to be the last axis) of `input` tensor
    to be placed at the desired axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channels axis
            should be the last axis in the output tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    return transpose_conv2d_axis(
        input, from_channels_last=True, to_channels_last=channels_last,
        name=name
    )


@add_name_arg_doc
def transpose_conv2d_channels_x_to_last(input, channels_last, name=None):
    """
    Ensure the channels axis of `input` tensor to be placed at the last axis.

    Args:
        input (tf.Tensor): The input tensor, at least 4-d.
        channels_last (bool): Whether or not the channels axis
            is the last axis in the `input` tensor?

    Returns:
        tf.Tensor: The (maybe) transposed output tensor.
    """
    return transpose_conv2d_axis(
        input, from_channels_last=channels_last, to_channels_last=True,
        name=name
    )


@add_name_arg_doc
def reshape_tail(input, ndims, shape, name=None):
    """
    Reshape the tail (last) `ndims` into specified `shape`.

    Usage::

        x = tf.zeros([2, 3, 4, 5, 6])
        reshape_tail(x, 3, [-1])  # output: zeros([2, 3, 120])
        reshape_tail(x, 1, [3, 2])  # output: zeros([2, 3, 4, 5, 3, 2])

    Args:
        input (Tensor): The input tensor, at least `ndims` dimensions.
        ndims (int): To reshape this number of dimensions at tail.
        shape (Iterable[int] or tf.Tensor): The shape of the new tail.

    Returns:
        tf.Tensor: The reshaped tensor.
    """
    from tfsnippet.ops import assert_rank_at_least

    input = tf.convert_to_tensor(input)
    if not is_tensor_object(shape):
        shape = list(int(s) for s in shape)
        neg_one_count = 0
        for s in shape:
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

    with tf.name_scope(name or 'reshape_tail', values=[input]):
        # assert the dimension
        with assert_deps([
                    assert_rank_at_least(
                        input, ndims,
                        message='rank(input) must be at least ndims')
                ]) as asserted:
            if asserted:  # pragma: no cover
                input = tf.identity(input)

        # compute the static shape
        static_input_shape = get_static_shape(input)
        static_output_shape = None

        if static_input_shape is not None:
            if ndims > 0:
                left_shape = static_input_shape[:-ndims]
                right_shape = static_input_shape[-ndims:]
            else:
                left_shape = static_input_shape
                right_shape = ()

            # attempt to resolve "-1" in `shape`
            if isinstance(shape, list):
                if None not in right_shape:
                    shape_size = int(np.prod([s for s in shape if s != -1]))
                    right_shape_size = int(np.prod(right_shape))

                    if (-1 not in shape and shape_size != right_shape_size) or \
                            (-1 in shape and right_shape_size % shape_size != 0):
                        raise ValueError(
                            'Cannot reshape the tail dimensions of '
                            '`input` into `shape`: input {!r}, ndims '
                            '{}, shape {}.'.format(input, ndims, shape)
                        )

                    if -1 in shape:
                        pos = shape.index(-1)
                        shape[pos] = right_shape_size // shape_size

                static_output_shape = left_shape + \
                    tuple(s if s != -1 else None for s in shape)

        static_output_shape = tf.TensorShape(static_output_shape)

        # compute the dynamic shape
        input_shape = get_shape(input)
        if ndims > 0:
            output_shape = concat_shapes([input_shape[:-ndims], shape])
        else:
            output_shape = concat_shapes([input_shape, shape])

        # do reshape
        output = tf.reshape(input, output_shape)
        output.set_shape(static_output_shape)
        return output
