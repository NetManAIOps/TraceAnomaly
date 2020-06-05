import tensorflow as tf

from tfsnippet.utils import (flatten_to_ndims, unflatten_from_ndims,
                             add_name_and_scope_arg_doc, assert_deps,
                             validate_positive_int_arg, get_static_shape,
                             model_variable)
from .base import FeatureMappingFlow
from .sequential import SequentialFlow
from .utils import assert_log_det_shape_matches_input

__all__ = [
    'PlanarNormalizingFlow', 'planar_normalizing_flows',
]


class PlanarNormalizingFlow(FeatureMappingFlow):
    """
    A single layer Planar Normalizing Flow (Danilo 2016) with `tanh` activation
    function, as well as the invertible trick.  The `x` and `y` are assumed to
    be 1-D random variable (i.e., ``value_ndims == 1``)

    .. math::

        \\begin{aligned}
            \\mathbf{y} &= \\mathbf{x} +
                \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b) \\\\
            \\mathbf{\\hat{u}} &= \\mathbf{u} +
                \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
                       (\\mathbf{w}^\\top \\mathbf{u})\\right]
                \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\\\
            m(a) &= -1 + \\log(1+\\exp(a))
        \\end{aligned}
    """

    @add_name_and_scope_arg_doc
    def __init__(self,
                 w_initializer=tf.random_normal_initializer(0., 0.01),
                 w_regularizer=None,
                 b_initializer=tf.zeros_initializer(),
                 b_regularizer=None,
                 u_initializer=tf.random_normal_initializer(0., 0.01),
                 u_regularizer=None,
                 trainable=True,
                 name=None,
                 scope=None):
        """
        Construct a new :class:`PlanarNormalizingFlow`.

        Args:
            w_initializer: The initializer for parameter `w`.
            w_regularizer: The regularizer for parameter `w`.
            b_regularizer: The regularizer for parameter `b`.
            b_initializer: The initializer for parameter `b`.
            u_regularizer: The regularizer for parameter `u`.
            u_initializer: The initializer for parameter `u`.
            trainable (bool): Whether or not the parameters are trainable?
                (default :obj:`True`)
        """
        self._w_initializer = w_initializer
        self._w_regularizer = w_regularizer
        self._b_initializer = b_initializer
        self._b_regularizer = b_regularizer
        self._u_initializer = u_initializer
        self._u_regularizer = u_regularizer
        self._trainable = bool(trainable)

        super(PlanarNormalizingFlow, self).__init__(
            axis=-1, value_ndims=1, require_batch_dims=True,
            name=name, scope=scope
        )

    def _build(self, input=None):
        dtype = input.dtype.base_dtype
        n_units = get_static_shape(input)[self.axis]

        w = model_variable(
            'w',
            shape=[1, n_units],
            dtype=dtype,
            initializer=self._w_initializer,
            regularizer=self._w_regularizer,
            trainable=self._trainable
        )
        b = model_variable(
            'b',
            shape=[1],
            dtype=dtype,
            initializer=self._b_initializer,
            regularizer=self._b_regularizer,
            trainable=self._trainable
        )
        u = model_variable(
            'u',
            shape=[1, n_units],
            dtype=dtype,
            initializer=self._u_initializer,
            regularizer=self._u_regularizer,
            trainable=self._trainable
        )
        wu = tf.matmul(w, u, transpose_b=True)  # wu.shape == [1]
        u_hat = u + (-1 + tf.nn.softplus(wu) - wu) * \
            w / tf.reduce_sum(tf.square(w))  # shape == [1, n_units]

        self._w, self._b, self._u, self._u_hat = w, b, u, u_hat

    @property
    def explicitly_invertible(self):
        return False

    def _transform(self, x, compute_y, compute_log_det):
        w, b, u, u_hat = self._w, self._b, self._u, self._u_hat

        # flatten x for better performance
        x_flatten, s1, s2 = flatten_to_ndims(x, 2)  # x.shape == [?, n_units]
        wxb = tf.matmul(x_flatten, w, transpose_b=True) + b  # shape == [?, 1]
        tanh_wxb = tf.tanh(wxb)  # shape == [?, 1]

        # compute y = f(x)
        y = None
        if compute_y:
            y = x_flatten + u_hat * tanh_wxb  # shape == [?, n_units]
            y = unflatten_from_ndims(y, s1, s2)

        # compute log(det|df/dz|)
        log_det = None
        if compute_log_det:
            grad = 1. - tf.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
            phi = grad * w  # shape == [?, n_units]
            u_phi = tf.matmul(phi, u_hat, transpose_b=True)  # shape == [?, 1]
            det_jac = 1. + u_phi  # shape == [?, 1]
            log_det = tf.log(tf.abs(det_jac))  # shape == [?, 1]
            log_det = unflatten_from_ndims(tf.squeeze(log_det, -1), s1, s2)

        # now returns the transformed sample and log-determinant
        return y, log_det

    # provide this method to avoid abstract class warning
    def _inverse_transform(self, y, compute_x, compute_log_det):
        raise RuntimeError('Should never be called.')  # pragma: no cover


@add_name_and_scope_arg_doc
def planar_normalizing_flows(
        n_layers=1,
        w_initializer=tf.random_normal_initializer(0., 0.01),
        w_regularizer=None,
        b_initializer=tf.zeros_initializer(),
        b_regularizer=None,
        u_initializer=tf.random_normal_initializer(0., 0.01),
        u_regularizer=None,
        trainable=True,
        name=None,
        scope=None):
    """
    Construct a sequential of :class`PlanarNormalizingFlow`.

    Args:
        n_layers (int): The number of :class`PlanarNormalizingFlow`.
        w_initializer: The initializer for parameter `w`.
        w_regularizer: The regularizer for parameter `w`.
        b_regularizer: The regularizer for parameter `b`.
        b_initializer: The initializer for parameter `b`.
        u_regularizer: The regularizer for parameter `u`.
        u_initializer: The initializer for parameter `u`.
        trainable (bool): Whether or not the parameters are trainable?
            (default :obj:`True`)

    Returns:
        SequentialFlow or PlanarNormalizingFlow: A :class:`SequentialFlow`
            if `n_layers > 1`, or a :class:`PlanarNormalizingFlow` if
            `n_layers == 1`.

    See Also:
        :class:`tfsnippet.layers.PlanarNormalizingFlow`
    """
    n_layers = validate_positive_int_arg('n_layers', n_layers)
    flow_kwargs = {
        'w_initializer': w_initializer, 'w_regularizer': w_regularizer,
        'b_initializer': b_initializer, 'b_regularizer': b_regularizer,
        'u_initializer': u_initializer, 'u_regularizer': u_regularizer,
        'trainable': trainable
    }

    if n_layers == 1:
        return PlanarNormalizingFlow(name=name, scope=scope, **flow_kwargs)

    else:
        with tf.variable_scope(
                scope, default_name=name or 'planar_normalizing_flows'):
            flows = []
            for i in range(n_layers):
                flows.append(PlanarNormalizingFlow(
                    name='_{}'.format(i), **flow_kwargs))
            return SequentialFlow(flows)
