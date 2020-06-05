from tfsnippet.utils import deprecated
from .evaluator import Evaluator, auto_batch_weight

__all__ = ['Validator']


@deprecated('use :class:`Evaluator` instead.', version='0.1')
class Validator(Evaluator):
    """Class to compute validation loss and other metrics."""

    def __init__(self, loop, metrics, inputs, data_flow, feed_dict=None,
                 time_metric_name='valid_time',
                 batch_weight_func=auto_batch_weight):  # pragma: no cover
        super(Validator, self).__init__(
            loop=loop, metrics=metrics, inputs=inputs, data_flow=data_flow,
            feed_dict=feed_dict, time_metric_name=time_metric_name,
            batch_weight_func=batch_weight_func
        )
