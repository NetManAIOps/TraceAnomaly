import tensorflow as tf

from tfsnippet.utils import (DocInherit,
                             TensorWrapper,
                             register_tensor_wrapper_class,
                             get_default_session_or_error)

__all__ = ['ScheduledVariable', 'AnnealingVariable']


@DocInherit
class ScheduledVariable(TensorWrapper):
    """
    A non-trainable :class:`tf.Variable`, whose value might need to be changed
    as training goes by.
    """

    def __init__(self, name, initial_value, dtype=tf.float32, model_var=False,
                 collections=None, **kwargs):
        """
        Construct a new :class:`ScheduledVariable`.

        Args:
            name (str): Name of the variable.
            initial_value: Initial value of the variable.
            dtype (tf.DType): Data type of the variable.
            model_var (bool): If :obj:`True`, will add the variable to
                the `MODEL_VARIABLES` collection.
            collections (Iterable[str]): Add the variable to these graph
                collections, in addition to the `MODEL_VARIABLES` and
                `GLOBAL_VARIABLES` collections.
            \\**kwargs: Additional named arguments passed to :meth:`_init()`.
        """
        with tf.name_scope('ScheduledVariable.init'):
            dtype = tf.as_dtype(dtype)

            initial_value = tf.convert_to_tensor(initial_value)
            if initial_value.dtype != dtype:
                initial_value = tf.cast(initial_value, dtype=dtype)

            collections = list(collections or ())
            collections += [tf.GraphKeys.GLOBAL_VARIABLES]
            if model_var:
                collections += [tf.GraphKeys.MODEL_VARIABLES]
            collections = list(set(collections))

            self._init(name, initial_value, dtype, collections, **kwargs)

    def _init(self, name, initial_value, dtype, collections, **kwargs):
        """
        Actually construct the :class:`ScheduledVariable`.

        Derived classes may override this to do more initialization.

        Args:
            name (str): Name of the variable.
            initial_value (tf.Tensor): The initial value, casted into
                :class:`tf.Tensor` with appropriate dtype.
            dtype (tf.DType): Data type of the variable.
            collections (list[str]): The variable collections to add.
        """
        self._self_var = tf.get_variable(
            name, dtype=dtype, initializer=initial_value, trainable=False,
            collections=collections
        )
        self._self_read_op = tf.convert_to_tensor(self._self_var)
        self._self_assign_ph = tf.placeholder(
            dtype=dtype, shape=self._self_var.get_shape())
        self._self_assign_op = tf.assign(
            self._self_var, self._self_assign_ph)

    @property
    def tensor(self):
        return self._self_read_op

    @property
    def variable(self):
        """
        Get the TensorFlow variable object.

        Returns:
            tf.Variable: The TensorFlow variable object.
        """
        return self._self_var

    def get(self):
        """Get the current value of the variable."""
        return get_default_session_or_error().run(self._self_read_op)

    def set(self, value):
        """
        Set the value of the variable.

        Args:
            value: The value to be assigned to the variable.
        """
        get_default_session_or_error().run(
            self._self_assign_op, feed_dict={self._self_assign_ph: value})


class AnnealingVariable(ScheduledVariable):
    """
    A non-trainable :class:`tf.Variable`, whose value will be annealed
    as training goes by.
    """

    def __init__(self, name, initial_value, ratio, min_value=None,
                 dtype=tf.float32):
        """
        Construct a new :class:`AnnealingVariable`.

        Args:
            name (str): Name of the variable.
            initial_value: Initial value of the variable.
            ratio: A number or a tensor, the ratio of annealing at each time.
            min_value: Optional, a number, the minimum value.
            dtype (tf.DType): Data type of the variable.
        """
        super(AnnealingVariable, self).__init__(
            name=name, initial_value=initial_value, dtype=dtype,
            ratio=ratio, min_value=min_value
        )

    def _init(self, name, initial_value, dtype, collections, ratio, min_value):
        ratio = tf.convert_to_tensor(ratio)
        if ratio.dtype != dtype:
            ratio = tf.cast(ratio, dtype=dtype)

        if min_value is not None:
            min_value = tf.convert_to_tensor(min_value)
            if min_value.dtype != dtype:
                min_value = tf.cast(min_value, dtype=dtype)
            initial_value = tf.maximum(initial_value, min_value)

        super(AnnealingVariable, self)._init(
            name, initial_value, dtype, collections)

        with tf.name_scope('anneal_op'):
            if min_value is not None:
                self._self_anneal_op = tf.assign(
                    self._self_var,
                    tf.maximum(min_value, self._self_var * ratio)
                )
            else:
                self._self_anneal_op = tf.assign(
                    self._self_var, self._self_var * ratio)

    def anneal(self):
        """Anneal the value."""
        get_default_session_or_error().run(self._self_anneal_op)


register_tensor_wrapper_class(ScheduledVariable)
