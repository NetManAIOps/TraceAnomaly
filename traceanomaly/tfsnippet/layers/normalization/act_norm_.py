import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tfsnippet.layers.flows.utils import (broadcast_log_det_against_input,
                                          ExpScale, LinearScale)
from tfsnippet.utils import (InputSpec, ParamSpec, add_name_and_scope_arg_doc,
                             get_static_shape, maybe_check_numerics,
                             validate_int_tuple_arg, get_dimensions_size,
                             validate_enum_arg, model_variable)
from ..flows import FeatureMappingFlow

__all__ = ['ActNorm', 'act_norm']


class ActNorm(FeatureMappingFlow):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    `y = (x + bias) * scale; log_det = y / scale - bias`

    `bias` and `scale` are initialized such that `y` will have zero mean and
    unit variance for the initial mini-batch of `x`.
    It can be initialized only through the forward pass.  You may need to use
    :meth:`BaseFlow.invert()` to get a inverted flow if you need to initialize
    the parameters via the opposite direction.
    """

    _build_require_input = True

    @add_name_and_scope_arg_doc
    def __init__(self,
                 axis=-1,
                 value_ndims=1,
                 initialized=False,
                 scale_type='exp',
                 bias_regularizer=None,
                 bias_constraint=None,
                 log_scale_regularizer=None,
                 log_scale_constraint=None,
                 scale_regularizer=None,
                 scale_constraint=None,
                 trainable=True,
                 epsilon=1e-6,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`ActNorm` instance.

        Args:
            axis (int or Iterable[int]): The axis to apply ActNorm.
                Dimensions not in `axis` will be averaged out when computing
                the mean of activations. Default `-1`, the last dimension.
                All items of the `axis` should be covered by `value_ndims`.
            value_ndims (int): Number of value dimensions in both `x` and `y`.
                `x.ndims - value_ndims == log_det.ndims` and
                `y.ndims - value_ndims == log_det.ndims`.
            initialized (bool): Whether or not the variables have been
                initialized?  If :obj:`False`, the first input `x` in the
                forward pass will be used to initialize the variables.

                Normally, it should take the default value, :obj:`False`.
                Setting it to :obj:`True` only if you're constructing a
                :class:`ActNorm` instance inside some reused variable scope.
            scale_type: One of {"exp", "linear"}.
                If "exp", ``y = (x + bias) * tf.exp(log_scale)``.
                If "linear", ``y = (x + bias) * scale``.
                Default is "exp".
            bias_regularizer: The regularizer for `bias`.
            bias_constraint: The constraint for `bias`.
            log_scale_regularizer: The regularizer for `log_scale`.
            log_scale_constraint: The constraint for `log_scale`.
            scale_regularizer: The regularizer for `scale`.
            scale_constraint: The constraint for `scale`.
            trainable (bool): Whether or not the variables are trainable?
            epsilon: Small float to avoid dividing by zero or taking
                logarithm of zero.
        """
        axis = validate_int_tuple_arg('axis', axis)
        self._scale_type = validate_enum_arg(
            'scale_type', scale_type, ['exp', 'linear'])
        self._initialized = bool(initialized)
        self._bias_regularizer = bias_regularizer
        self._bias_constraint = bias_constraint
        self._log_scale_regularizer = log_scale_regularizer
        self._log_scale_constraint = log_scale_constraint
        self._scale_regularizer = scale_regularizer
        self._scale_constraint = scale_constraint
        self._trainable = bool(trainable)
        self._epsilon = epsilon

        super(ActNorm, self).__init__(axis=axis, value_ndims=value_ndims,
                                      name=name, scope=scope)

    def _build(self, input=None):
        # check the input.
        input = tf.convert_to_tensor(input)
        dtype = input.dtype.base_dtype
        shape = get_static_shape(input)

        # These facts should have been checked in `BaseFlow.build`.
        assert(shape is not None)
        assert(len(shape) >= self.value_ndims)

        # compute var spec and input spec
        min_axis = min(self.axis)
        shape_spec = [None] * len(shape)
        for a in self.axis:
            shape_spec[a] = shape[a]
        shape_spec = shape_spec[min_axis:]
        assert(not not shape_spec)
        assert(self.value_ndims >= len(shape_spec))

        self._y_input_spec = self._x_input_spec = InputSpec(
            shape=(('...',) +
                   ('?',) * (self.value_ndims - len(shape_spec)) +
                   tuple(shape_spec)),
            dtype=dtype
        )
        # the shape of variables must only have necessary dimensions,
        # such that we can switch freely between `channels_last = True`
        # (in which case `input.shape = (..., *,)`, and `channels_last = False`
        # (in which case `input.shape = (..., *, 1, 1)`.
        self._var_shape = tuple(s for s in shape_spec if s is not None)
        # and we still need to compute the aligned variable shape, such that
        # we can immediately reshape the variables into this aligned shape,
        # then compute `scale * input + bias`.
        self._var_shape_aligned = tuple(s or 1 for s in shape_spec)
        self._var_spec = ParamSpec(self._var_shape)

        # validate the input
        self._x_input_spec.validate('input', input)

        # build the variables
        self._bias = model_variable(
            'bias',
            dtype=dtype,
            shape=self._var_shape,
            regularizer=self._bias_regularizer,
            constraint=self._bias_constraint,
            trainable=self._trainable
        )
        if self._scale_type == 'exp':
            self._pre_scale = model_variable(
                'log_scale',
                dtype=dtype,
                shape=self._var_shape,
                regularizer=self._log_scale_regularizer,
                constraint=self._log_scale_constraint,
                trainable=self._trainable
            )
        else:
            self._pre_scale = model_variable(
                'scale',
                dtype=dtype,
                shape=self._var_shape,
                regularizer=self._scale_regularizer,
                constraint=self._scale_constraint,
                trainable=self._trainable
            )

    @property
    def explicitly_invertible(self):
        return True

    def _transform(self, x, compute_y, compute_log_det):
        # check the argument
        dtype = x.dtype.base_dtype
        shape = get_static_shape(x)
        assert(-len(shape) <= -self.value_ndims <= min(self.axis))
        reduce_axis = tuple(sorted(
            set(range(-len(shape), 0)).difference(self.axis)))

        # prepare for the parameters
        if not self._initialized:
            if len(shape) == len(self._var_shape_aligned):
                raise ValueError('Initializing ActNorm requires multiple '
                                 '`x` samples, thus `x` must have at least '
                                 'one more dimension than the variable shape: '
                                 'x {} vs variable shape {}.'.
                                 format(x, self._var_shape_aligned))

            with tf.name_scope('initialization'):
                x_mean, x_var = tf.nn.moments(x, reduce_axis)
                x_mean = tf.reshape(x_mean, self._var_shape)
                x_var = maybe_check_numerics(
                    tf.reshape(x_var, self._var_shape),
                    'numeric issues in computed x_var'
                )

                bias = self._bias.assign(-x_mean)
                if self._scale_type == 'exp':
                    pre_scale = self._pre_scale.assign(
                        -tf.constant(.5, dtype=dtype) *
                        tf.log(tf.maximum(x_var, self._epsilon))
                    )
                    pre_scale = maybe_check_numerics(
                        pre_scale, 'numeric issues in initializing log_scale')
                else:
                    assert(self._scale_type == 'linear')
                    pre_scale = self._pre_scale.assign(
                        tf.constant(1., dtype=dtype) /
                        tf.sqrt(tf.maximum(x_var, self._epsilon))
                    )
                    pre_scale = maybe_check_numerics(
                        pre_scale, 'numeric issues in initializing scale')
            self._initialized = True
        else:
            bias = self._bias
            pre_scale = self._pre_scale

        # align the shape of variables, and create the scale object
        bias = tf.reshape(bias, self._var_shape_aligned)
        pre_scale = tf.reshape(pre_scale, self._var_shape_aligned)

        if self._scale_type == 'exp':
            scale = ExpScale(pre_scale, self._epsilon)
        else:
            assert(self._scale_type == 'linear')
            scale = LinearScale(pre_scale, self._epsilon)

        # compute y
        y = None
        if compute_y:
            y = (x + bias) * scale

        # compute log_det
        log_det = None
        if compute_log_det:
            with tf.name_scope('log_det'):
                log_det = scale.log_scale()
                reduce_ndims1 = min(
                    self.value_ndims, len(self._var_shape_aligned))
                reduce_ndims2 = self.value_ndims - reduce_ndims1

                # reduce the last `min(value_ndims, len(var_shape))` dimensions
                if reduce_ndims1 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims1, 0)))

                    # the following axis have been averaged out during
                    # computation, and will be directly summed up without
                    # getting broadcasted. Thus we need to multiply a factor
                    # to the log_det by the count of reduced elements.
                    reduce_axis1 = tuple(filter(
                        lambda a: (a >= -reduce_ndims1),
                        reduce_axis
                    ))
                    reduce_shape1 = get_dimensions_size(x, reduce_axis1)
                    if isinstance(reduce_shape1, tuple):
                        log_det *= np.prod(reduce_shape1, dtype=np.float32)
                    else:
                        log_det *= tf.cast(
                            tf.reduce_prod(reduce_shape1),
                            dtype=log_det.dtype
                        )

                # we need to broadcast `log_det` to match the shape of `x`
                log_det = broadcast_log_det_against_input(
                    log_det, x, value_ndims=reduce_ndims1)

                # reduce the remaining dimensions
                if reduce_ndims2 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims2, 0)))

        return y, log_det

    def _inverse_transform(self, y, compute_x, compute_log_det):
        # `BaseFlow` ensures `build` is called before `inverse_transform`.
        # In `ActNorm`, `build` can only be called by `apply` or `transform`.
        # Thus it should always have been initialized.
        assert(self._initialized)

        # check the argument
        shape = get_static_shape(y)
        assert (-len(shape) <= -self.value_ndims <= min(self.axis))
        reduce_axis = tuple(sorted(
            set(range(-len(shape), 0)).difference(self.axis)))

        # align the shape of variables, and create the scale object
        bias = tf.reshape(self._bias, self._var_shape_aligned)
        pre_scale = tf.reshape(self._pre_scale, self._var_shape_aligned)

        if self._scale_type == 'exp':
            scale = ExpScale(pre_scale, self._epsilon)
        else:
            assert(self._scale_type == 'linear')
            scale = LinearScale(pre_scale, self._epsilon)

        # compute x
        x = None
        if compute_x:
            x = y / scale - bias

        # compute log_det
        log_det = None
        if compute_log_det:
            with tf.name_scope('log_det'):
                log_det = scale.neg_log_scale()
                reduce_ndims1 = min(
                    self.value_ndims, len(self._var_shape_aligned))
                reduce_ndims2 = self.value_ndims - reduce_ndims1

                # reduce the last `min(value_ndims, len(var_shape))` dimensions
                if reduce_ndims1 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims1, 0)))

                    # the following axis have been averaged out during
                    # computation, and will be directly summed up without
                    # getting broadcasted. Thus we need to multiply a factor
                    # to the log_det by the count of reduced elements.
                    reduce_axis1 = tuple(filter(
                        lambda a: (a >= -reduce_ndims1),
                        reduce_axis
                    ))
                    reduce_shape1 = get_dimensions_size(y, reduce_axis1)
                    if isinstance(reduce_shape1, tuple):
                        log_det *= np.prod(reduce_shape1, dtype=np.float32)
                    else:
                        log_det *= tf.cast(
                            tf.reduce_prod(reduce_shape1),
                            dtype=log_det.dtype
                        )

                # we need to broadcast `log_det` to match the shape of `y`
                log_det = broadcast_log_det_against_input(
                    log_det, y, value_ndims=reduce_ndims1)

                # reduce the remaining dimensions
                if reduce_ndims2 > 0:
                    log_det = tf.reduce_sum(
                        log_det, axis=list(range(-reduce_ndims2, 0)))

        return x, log_det


@add_arg_scope
@add_name_and_scope_arg_doc
def act_norm(input,
             axis=-1,
             value_ndims=1,
             initializing=False,
             scale_type='exp',
             bias_regularizer=None,
             bias_constraint=None,
             log_scale_regularizer=None,
             log_scale_constraint=None,
             scale_regularizer=None,
             scale_constraint=None,
             trainable=True,
             epsilon=1e-6,
             name=None,
             scope=None):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    Examples::

        import tfsnippet as spt

        # apply act_norm on a dense layer
        x = spt.layers.dense(x, units, activation_fn=tf.nn.relu,
                             normalizer_fn=functools.partial(
                                 act_norm, initializing=initializing))

        # apply act_norm on a conv2d layer
        x = spt.layers.conv2d(x, out_channels, (3, 3),
                              channels_last=channels_last,
                              activation_fn=tf.nn.relu,
                              normalizer_fn=functools.partial(
                                  act_norm,
                                  axis=-1 if channels_last else -3,
                                  value_ndims=3,
                                  initializing=initializing,
                              ))

    Args:
        input (tf.Tensor): The input tensor.
        axis (int or Iterable[int]): The axis to apply ActNorm.
            Dimensions not in `axis` will be averaged out when computing
            the mean of activations. Default `-1`, the last dimension.
            All items of the `axis` should be covered by `value_ndims`.
        value_ndims (int): Number of dimensions to be considered as the
            value dimensions.  `x.ndims - value_ndims == log_det.ndims`.
        initializing (bool): Whether or not to use the input `x` to initialize
            the layer parameters? (default :obj:`True`)
        scale_type: One of {"exp", "linear"}.
            If "exp", ``y = (x + bias) * tf.exp(log_scale)``.
            If "linear", ``y = (x + bias) * scale``.
            Default is "exp".
        bias_regularizer: The regularizer for `bias`.
        bias_constraint: The constraint for `bias`.
        log_scale_regularizer: The regularizer for `log_scale`.
        log_scale_constraint: The constraint for `log_scale`.
        scale_regularizer: The regularizer for `scale`.
        scale_constraint: The constraint for `scale`.
        trainable (bool): Whether or not the variables are trainable?
        epsilon: Small float to avoid dividing by zero or taking
            logarithm of zero.

    Returns:
        tf.Tensor: The output after the ActNorm has been applied.
    """
    layer = ActNorm(
        axis=axis,
        value_ndims=value_ndims,
        initialized=not initializing,
        scale_type=scale_type,
        bias_regularizer=bias_regularizer,
        bias_constraint=bias_constraint,
        log_scale_regularizer=log_scale_regularizer,
        log_scale_constraint=log_scale_constraint,
        scale_regularizer=scale_regularizer,
        scale_constraint=scale_constraint,
        trainable=trainable,
        epsilon=epsilon,
        name=name,
        scope=scope
    )
    return layer.apply(input)
