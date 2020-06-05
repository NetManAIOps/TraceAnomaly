import tensorflow as tf

from tfsnippet.utils import (DocInherit, add_name_and_scope_arg_doc,
                             get_default_scope_name, get_static_shape,
                             InputSpec, is_integer, assert_deps)
from ..base import BaseLayer
from .utils import assert_log_det_shape_matches_input, ZeroLogDet

__all__ = ['BaseFlow', 'MultiLayerFlow', 'FeatureMappingFlow']


@DocInherit
class BaseFlow(BaseLayer):
    """
    The basic class for normalizing flows.

    A normalizing flow transforms a random variable `x` into `y` by an
    (implicitly) invertible mapping :math:`y = f(x)`, whose Jaccobian matrix
    determinant :math:`\\det \\frac{\\partial f(x)}{\\partial x} \\neq 0`, thus
    can derive :math:`\\log p(y)` from given :math:`\\log p(x)`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 x_value_ndims,
                 y_value_ndims=None,
                 require_batch_dims=False,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`BaseFlow`.

        Args:
            x_value_ndims (int): Number of value dimensions in `x`.
                `x.ndims - x_value_ndims == log_det.ndims`.
            y_value_ndims (int): Number of value dimensions in `y`.
                `y.ndims - y_value_ndims == log_det.ndims`.
                If not specified, use `x_value_ndims`.
            require_batch_dims (bool): If :obj:`True`, `x` are required
                to have at least `x_value_ndims + 1` dimensions, and `y`
                are required to have at least `y_value_ndims + 1` dimensions.

                If :obj:`False`, `x` are required to have at least
                `x_value_ndims` dimensions, and `y` are required to have
                at least `y_value_ndims` dimensions.
        """
        x_value_ndims = int(x_value_ndims)
        if y_value_ndims is None:
            y_value_ndims = x_value_ndims
        else:
            y_value_ndims = int(y_value_ndims)

        super(BaseFlow, self).__init__(name=name, scope=scope)
        self._x_value_ndims = x_value_ndims
        self._y_value_ndims = y_value_ndims
        self._require_batch_dims = bool(require_batch_dims)

        self._x_input_spec = None  # type: InputSpec
        self._y_input_spec = None  # type: InputSpec

    def invert(self):
        """
        Get the inverted flow from this flow.

        The :meth:`transform()` will become the :meth:`inverse_transform()`
        in the inverted flow, and the :meth:`inverse_transform()` will become
        the :meth:`transform()` in the inverted flow.

        If the current flow has not been initialized, it must be initialized
        via :meth:`inverse_transform()` in the new flow.

        Returns:
            tfsnippet.layers.InvertFlow: The inverted flow.
        """
        from .invert import InvertFlow
        return InvertFlow(self)

    @property
    def x_value_ndims(self):
        """
        Get the number of value dimensions in `x`.

        Returns:
            int: The number of value dimensions in `x`.
        """
        return self._x_value_ndims

    @property
    def y_value_ndims(self):
        """
        Get the number of value dimensions in `y`.

        Returns:
            int: The number of value dimensions in `y`.
        """
        return self._y_value_ndims

    @property
    def require_batch_dims(self):
        """Whether or not this flow requires batch dimensions."""
        return self._require_batch_dims

    @property
    def explicitly_invertible(self):
        """
        Whether or not this flow is explicitly invertible?

        If a flow is not explicitly invertible, then it only supports to
        transform `x` into `y`, and corresponding :math:`\\log p(x)` into
        :math:`\\log p(y)`.  It cannot compute :math:`\\log p(y)` directly
        without knowing `x`, nor can it transform `x` back into `y`.

        Returns:
            bool: A boolean indicating whether or not the flow is explicitly
                invertible.
        """
        raise NotImplementedError()

    def _build_input_spec(self, input):
        batch_ndims = int(self.require_batch_dims)
        dtype = input.dtype.base_dtype

        x_input_shape = ['...'] + ['?'] * (self.x_value_ndims + batch_ndims)
        y_input_shape = ['...'] + ['?'] * (self.y_value_ndims + batch_ndims)

        self._x_input_spec = InputSpec(shape=x_input_shape, dtype=dtype)
        self._y_input_spec = InputSpec(shape=y_input_shape, dtype=dtype)

    def build(self, input=None):
        # check the input.
        if input is None:
            raise ValueError('`input` is required to build {}.'.
                             format(self.__class__.__name__))

        input = tf.convert_to_tensor(input)
        shape = get_static_shape(input)
        require_ndims = self.x_value_ndims + int(self.require_batch_dims)
        require_ndims_text = ('x_value_ndims + 1'
                              if self.require_batch_dims else 'x_value_ndims')

        if shape is None or len(shape) < require_ndims:
            raise ValueError('`x.ndims` must be known and >= `{}`: x '
                             '{} vs ndims `{}`.'.
                             format(require_ndims_text, input, require_ndims))

        # build the input spec
        self._build_input_spec(input)

        # build the layer
        return super(BaseFlow, self).build(input)

    def _transform(self, x, compute_y, compute_log_det):
        raise NotImplementedError()

    def transform(self, x, compute_y=True, compute_log_det=True, name=None):
        """
        Transform `x` into `y`, and compute the log-determinant of `f` at `x`
        (i.e., :math:`\\log \\det \\frac{\\partial f(x)}{\\partial x}`).

        Args:
            x (Tensor): The samples of `x`.
            compute_y (bool): Whether or not to compute :math:`y = f(x)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.

        Returns:
            (tf.Tensor, tf.Tensor): `y` and the (maybe summed) log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_y` and `compute_log_det` are set
                to :obj:`False`.
        """
        if not compute_y and not compute_log_det:
            raise ValueError('At least one of `compute_y` and '
                             '`compute_log_det` should be True.')

        x = tf.convert_to_tensor(x)
        if not self._has_built:
            self.build(x)

        x = self._x_input_spec.validate('x', x)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('transform', self),
                values=[x]):
            y, log_det = self._transform(x, compute_y, compute_log_det)

            if compute_log_det:
                with assert_deps([
                            assert_log_det_shape_matches_input(
                                log_det=log_det,
                                input=x,
                                value_ndims=self.x_value_ndims
                            )
                        ]) as asserted:
                    if asserted:  # pragma: no cover
                        log_det = tf.identity(log_det)

            return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        raise NotImplementedError()

    def inverse_transform(self, y, compute_x=True, compute_log_det=True,
                          name=None):
        """
        Transform `y` into `x`, and compute the log-determinant of `f^{-1}` at
        `y` (i.e.,
        :math:`\\log \\det \\frac{\\partial f^{-1}(y)}{\\partial y}`).

        Args:
            y (Tensor): The samples of `y`.
            compute_x (bool): Whether or not to compute :math:`x = f^{-1}(y)`?
                Default :obj:`True`.
            compute_log_det (bool): Whether or not to compute the
                log-determinant?  Default :obj:`True`.
            name (str): If specified, will use this name as the TensorFlow
                operational name scope.

        Returns:
            (tf.Tensor, tf.Tensor): `x` and the (maybe summed) log-determinant.
                The items in the returned tuple might be :obj:`None`
                if corresponding `compute_?` argument is set to :obj:`False`.

        Raises:
            RuntimeError: If both `compute_x` and `compute_log_det` are set
                to :obj:`False`.
            RuntimeError: If the flow is not explicitly invertible.
        """
        if not self.explicitly_invertible:
            raise RuntimeError('The flow is not explicitly invertible: {!r}'.
                               format(self))
        if not compute_x and not compute_log_det:
            raise ValueError('At least one of `compute_x` and '
                             '`compute_log_det` should be True.')
        if not self._has_built:
            raise RuntimeError('`inverse_transform` cannot be called before '
                               'the flow has been built; it can be built by '
                               'calling `build`, `apply` or `transform`: '
                               '{!r}'.format(self))

        y = tf.convert_to_tensor(y)
        y = self._y_input_spec.validate('y', y)

        with tf.name_scope(
                name,
                default_name=get_default_scope_name('inverse_transform', self),
                values=[y]):
            x, log_det = self._inverse_transform(y, compute_x, compute_log_det)

            if compute_log_det:
                with assert_deps([
                            assert_log_det_shape_matches_input(
                                log_det=log_det,
                                input=y,
                                value_ndims=self.y_value_ndims
                            )
                        ]) as asserted:
                    if asserted:  # pragma: no cover
                        log_det = tf.identity(log_det)

            return x, log_det

    def _apply(self, x):
        y, _ = self.transform(x, compute_y=True, compute_log_det=False)
        return y


def sum_log_det(log_det_list, name='sum_log_det'):
    """
    Carefully sum up `log_det_list`, to allow :class:`ZeroLogDet` to opt-out
    some unnecessary zero tensors.
    """
    assert(not not log_det_list)
    with tf.name_scope(name):
        log_det = log_det_list[0]
        for t in log_det_list[1:]:
            # adjust the summation order, to allow `ZeroLogDet` to opt-out
            if isinstance(log_det, ZeroLogDet):
                log_det = log_det + t
            else:
                log_det = t + log_det
        return log_det


class MultiLayerFlow(BaseFlow):
    """Base class for multi-layer normalizing flows."""

    @add_name_and_scope_arg_doc
    def __init__(self, n_layers, **kwargs):
        """
        Construct a new :class:`MultiLayerFlow`.

        Args:
            n_layers (int): Number of flow layers.
            \\**kwargs: Other named arguments passed to :class:`BaseFlow`.
        """
        n_layers = int(n_layers)
        if n_layers < 1:
            raise ValueError('`n_layers` must be larger than 0.')
        self._n_layers = n_layers
        self._layer_params = []

        super(MultiLayerFlow, self).__init__(**kwargs)

    @property
    def n_layers(self):
        """
        Get the number of flow layers.

        Returns:
            int: The number of flow layers.
        """
        return self._n_layers

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        raise NotImplementedError()

    def _transform(self, x, compute_y, compute_log_det):
        # apply transformation of each layer
        log_det_list = []
        for i in range(self._n_layers):
            with tf.name_scope('_{}'.format(i)):
                x, log_det = self._transform_layer(
                    layer_id=i,
                    x=x,
                    compute_y=True if i < self._n_layers - 1 else compute_y,
                    compute_log_det=compute_log_det
                )
                log_det_list.append(log_det)

        # compose the return values
        y = x if compute_y else None
        return y, sum_log_det(log_det_list)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        raise NotImplementedError()

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # apply transformation of each layer
        log_det_list = []
        for i in range(self._n_layers - 1, -1, -1):
            with tf.name_scope('_{}'.format(i)):
                y, log_det = self._inverse_transform_layer(
                    layer_id=i,
                    y=y,
                    compute_x=True if i > 0 else compute_x,
                    compute_log_det=compute_log_det
                )
                log_det_list.append(log_det)

        # compose the return values
        x = y if compute_x else None
        return x, sum_log_det(log_det_list)


class FeatureMappingFlow(BaseFlow):
    """
    Base class for flows mapping input features to output features.

    A :class:`FeatureMappingFlow` must not change the value dimensions,
    i.e., `x_value_ndims == y_value_ndims`.  Thus, one single argument
    `value_ndims` replaces the `x_value_ndims` and `y_value_ndims` arguments.

    The :class:`FeatureMappingFlow` transforms a specified axis or a list
    of specified axes.  The axis/axes is/are specified via the argument
    `axis`.  All the `axis` must be covered by `value_ndims`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, axis, value_ndims, **kwargs):
        """
        Construct a new :class:`FeatureMappingFlow`.

        Args:
            axis (int or Iterable[int]): The feature axis/axes, on which to
                apply the transformation.
            value_ndims (int): Number of value dimensions in both `x` and `y`.
                `x.ndims - value_ndims == log_det.ndims` and
                `y.ndims - value_ndims == log_det.ndims`.
            \\**kwargs: Other named arguments passed to :class:`BaseFlow`.
        """
        # check the arguments
        if is_integer(axis):
            axis = int(axis)
        else:
            axis = tuple(int(a) for a in axis)
            if not axis:
                raise ValueError('`axis` must not be empty.')

        if 'x_value_ndims' in kwargs or 'y_value_ndims' in kwargs:
            raise ValueError('Specifying `x_value_ndims` or `y_value_ndims` '
                             'for a `FeatureMappingFlow` is not allowed.')
        value_ndims = int(value_ndims)

        # construct the layer
        super(FeatureMappingFlow, self).__init__(
            x_value_ndims=value_ndims, y_value_ndims=value_ndims, **kwargs)
        self._axis = axis

    @property
    def axis(self):
        """
        Get the feature axis/axes.

        Returns:
            int or tuple[int]: The feature axis/axes, as is specified
                in the constructor.
        """
        return self._axis

    @property
    def value_ndims(self):
        """
        Get the number of value dimensions in both `x` and `y`.

        Returns:
            int: The number of value dimensions in both `x` and `y`.
        """
        assert(self.x_value_ndims == self.y_value_ndims)
        return self.x_value_ndims

    def _build_input_spec(self, input):
        super(FeatureMappingFlow, self)._build_input_spec(input)

        dtype = input.dtype.base_dtype
        shape = get_static_shape(input)

        # These facts should have been checked in `BaseFlow.build`.
        assert (shape is not None)
        assert (len(shape) >= self.value_ndims)

        # validate the feature axis, ensure it is covered by `value_ndims`.
        axis = self._axis
        axis_is_int = is_integer(axis)
        if axis_is_int:
            axis = [axis]
        else:
            axis = list(axis)

        for i, a in enumerate(axis):
            if a < 0:
                a += len(shape)
            if a < 0 or a < len(shape) - self.value_ndims:
                raise ValueError('`axis` out of range, or not covered by '
                                 '`value_ndims`: axis {}, value_ndims {}, '
                                 'input {}'.
                                 format(self._axis, self.value_ndims, input))
            if shape[a] is None:
                raise ValueError('The feature axis of `input` is not '
                                 'deterministic: input {}, axis {}'.
                                 format(input, self._axis))

            # Store the negative axis, such that when new inputs can have more
            # dimensions than this `input`, the axis can still be correctly
            # resolved.
            axis[i] = a - len(shape)

        if axis_is_int:
            assert(len(axis) == 1)
            self._axis = axis[0]
        else:
            axis_len = len(axis)
            axis = tuple(sorted(set(axis)))
            if len(axis) != axis_len:
                raise ValueError('Duplicated elements after resolving negative '
                                 '`axis` with respect to the `input`: '
                                 'input {}, axis {}'.format(input, self._axis))
            self._axis = tuple(axis)

        # re-build the input spec
        batch_ndims = int(self.require_batch_dims)
        shape_spec = ['...'] + ['?'] * (self.value_ndims + batch_ndims)
        for a in axis:
            shape_spec[a] = shape[a]
        self._y_input_spec = self._x_input_spec = InputSpec(shape=shape_spec,
                                                            dtype=dtype)
        self._x_input_spec.validate('input', input)
