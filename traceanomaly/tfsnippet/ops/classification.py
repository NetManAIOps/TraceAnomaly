import tensorflow as tf

from tfsnippet.utils import add_name_arg_doc, InputSpec, get_static_shape

__all__ = [
    'classification_accuracy',
    'softmax_classification_output',
]


@add_name_arg_doc
def classification_accuracy(y_pred, y_true, name=None):
    """
    Compute the classification accuracy for `y_pred` and `y_true`.

    Args:
        y_pred: The predicted labels.
        y_true: The ground truth labels.  Its shape must match `y_pred`.

    Returns:
        tf.Tensor: The accuracy.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = InputSpec(shape=get_static_shape(y_pred)). \
        validate('y_true', y_true)
    with tf.name_scope(name, default_name='classification_accuracy',
                       values=[y_pred, y_true]):
        return tf.reduce_mean(
            tf.cast(tf.equal(y_pred, y_true), dtype=tf.float32))


@add_name_arg_doc
def softmax_classification_output(logits, name=None):
    """
    Get the most possible softmax classification output for each logit.

    Args:
        logits: The softmax logits.  Its last dimension will be treated
            as the softmax logits dimension, and will be reduced.

    Returns:
        tf.Tensor: tf.int32 tensor, the class label for each logit.
    """
    logits = InputSpec(shape=('...', '?', '?')).validate('logits', logits)
    with tf.name_scope(name, default_name='softmax_classification_output',
                       values=[logits]):
        return tf.argmax(logits, axis=-1, output_type=tf.int32)
