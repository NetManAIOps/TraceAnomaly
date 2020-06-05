import tensorflow as tf

__all__ = ['model_variable', 'get_model_variables']


def model_variable(name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   constraint=None,
                   trainable=True,
                   collections=None,
                   **kwargs):
    """
    Get or create a model variable.

    When the variable is created, it will be added to both `GLOBAL_VARIABLES`
    and `MODEL_VARIABLES` collection.

    Args:
        name: Name of the variable.
        shape: Shape of the variable.
        dtype: Data type of the variable.
        initializer: Initializer of the variable.
        regularizer: Regularizer of the variable.
        constraint: Constraint of the variable.
        trainable (bool): Whether or not the variable is trainable?
        collections: In addition to `GLOBAL_VARIABLES` and `MODEL_VARIABLES`,
            also add the variable to these collections.
        \\**kwargs: Other named arguments passed to :func:`tf.get_variable`.

    Returns:
        tf.Variable: The variable.
    """
    collections = list(set(
        list(collections or ()) +
        [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.MODEL_VARIABLES]
    ))
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        constraint=constraint,
        trainable=trainable,
        collections=collections,
        **kwargs
    )


def get_model_variables(scope=None):
    """
    Get all model variables (i.e., variables in `MODEL_VARIABLES` collection).

    Args:
        scope: If specified, will obtain variables only within this scope.

    Returns:
        list[tf.Variable]: The model variables.
    """
    return tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=scope)
