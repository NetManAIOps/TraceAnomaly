import six
import tensorflow as tf

__all__ = [
    'create_session',
    'get_default_session_or_error',
    'get_variables_as_dict',
    'get_uninitialized_variables',
    'ensure_variables_initialized',
    'get_variable_ddi',
]


def create_session(lock_memory=True,
                   log_device_placement=False,
                   allow_soft_placement=True,
                   **kwargs):
    """
    A convenient method to create a TensorFlow session.

    Args:
        lock_memory (True or False or float):

            * If :obj:`True`, lock all free memory.

            * If :obj:`False`, set `allow_growth` to True, i.e., not to lock
                all free memory.

            * If float, lock this portion of memory.

            (default :obj:`None`)

        log_device_placement (bool): Whether to log the placement of graph
            nodes.   (default :obj:`False`)
        allow_soft_placement (bool): Whether or not to allow soft placement?
            (default :obj:`True`)
        \\**kwargs: Other named parameters to be passed to `tf.ConfigProto`.

    Returns:
        tf.Session: The TensorFlow session.
    """
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement,
                            **kwargs)
    if lock_memory is False:
        config.gpu_options.allow_growth = True
    elif isinstance(lock_memory, float):
        config.gpu_options.per_process_gpu_memory_fraction = lock_memory
    elif lock_memory is not True:
        raise TypeError('`lock_memory` must be True, False or float.')
    session = tf.Session(config=config)
    return session


def get_default_session_or_error():
    """
    Get the default session.

    Returns:
        tf.Session: The default session.

    Raises:
        RuntimeError: If there's no active session.
    """
    ret = tf.get_default_session()
    if ret is None:
        raise RuntimeError('No session is active')
    return ret


def get_variables_as_dict(scope=None, collection=tf.GraphKeys.GLOBAL_VARIABLES):
    """
    Get TensorFlow variables as dict.

    Args:
        scope (str or tf.VariableScope or None): If :obj:`None`, will collect
            all the variables within current graph.  If a :class:`str` or a
            :class:`tf.VariableScope`, will collect the variables only from
            this scope. (default :obj:`None`)
        collection (str): Collect the variables only from this collection.
            (default ``tf.GraphKeys.GLOBAL_VARIABLES``)

    Returns:
        dict[str, tf.Variable]: Dict which maps from names to TensorFlow
            variables.  The names will be the full names of variables if
            `scope` is not specified, or the `relative names` within the
            `scope` otherwise. By `relative names` we mean the variable names
            without the common scope name prefix.
    """
    # get the common prefix to be stripped
    if isinstance(scope, tf.VariableScope):
        scope_name = scope.name
    else:
        scope_name = scope
    if scope_name and not scope_name.endswith('/'):
        scope_name += '/'
    scope_name_len = len(scope_name) if scope_name else 0

    # get the variables and strip the prefix
    variables = tf.get_collection(collection, scope_name)
    return {
        var.name[scope_name_len:].rsplit(':', 1)[0]: var
        for var in variables
    }


def get_uninitialized_variables(variables=None, name=None):
    """
    Get uninitialized variables as a list.

    Args:
        variables (list[tf.Variable]): Collect only uninitialized variables
            within this list. If not specified, will collect all uninitialized
            variables within ``tf.GraphKeys.GLOBAL_VARIABLES`` collection.
        name (str): TensorFlow name scope of the graph nodes.

    Returns:
        list[tf.Variable]: Uninitialized variables.
    """
    sess = get_default_session_or_error()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)
    with tf.name_scope(name, default_name='get_uninitialized_variables'):
        init_flag = sess.run(tf.stack(
            [tf.is_variable_initialized(v) for v in variables]
        ))
    return [v for v, f in zip(variables, init_flag) if not f]


def ensure_variables_initialized(variables=None, name=None):
    """
    Ensure variables are initialized.

    Args:
        variables (list[tf.Variable] or dict[str, tf.Variable]): Ensure only
            the variables within this collection to be initialized. If not
            specified, will ensure all variables within the collection
            `tf.GraphKeys.GLOBAL_VARIABLES` to be initialized.
        name (str): TensorFlow name scope of the graph nodes. (default
            `ensure_variables_initialized`)
    """
    with tf.name_scope(name, default_name='ensure_variables_initialized'):
        if isinstance(variables, dict):
            variables = list(six.itervalues(variables))
        uninitialized = get_uninitialized_variables(variables)
        if uninitialized:
            sess = get_default_session_or_error()
            sess.run(tf.variables_initializer(uninitialized))


def get_variable_ddi(name,
                     initial_value,
                     shape=None,
                     dtype=tf.float32,
                     initializing=False,
                     regularizer=None,
                     constraint=None,
                     trainable=True,
                     collections=None,
                     **kwargs):
    """
    Wraps :func:`tf.get_variable` to support data-dependent initialization.

    Args:
        name: Name of the variable.
        initial_value: The data-dependent initial value of the variable.
        shape: Shape of the variable.
        dtype: Data type of the variable.
        initializing (bool): Whether or not it is building the graph for
            data-dependent initialization? Ignored if `initial_value` is absent.
        regularizer: Regularizer of the variable.
        constraint: Constraint of the variable.
        trainable (bool): Whether or not to the variable is trainable?
        collections (Iterable[str]): Add the variable to these collections.
        \\**kwargs: Other named parameters passed to :func:`tf.get_variable`.

    Returns:
        tf.Variable or tf.Tensor: The variable or the tensor.
    """
    # TODO: detect shape from `initial_value` if not specified
    v = tf.get_variable(
        name, shape=shape, dtype=dtype, regularizer=regularizer,
        constraint=constraint, trainable=trainable, collections=collections,
        **kwargs
    )
    if initializing:
        v = v.assign(initial_value)
    return v
