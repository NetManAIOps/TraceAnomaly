import warnings

from tfsnippet.scaffold import TrainLoop
from tfsnippet.utils import deprecated, deprecated_arg
from .trainer import Trainer
from .feed_dict import merge_feed_dict

__all__ = ['LossTrainer']


@deprecated('use :class:`Trainer` instead.', version='0.1')
class LossTrainer(Trainer):
    """
    A subclass of :class:`BaseTrainer`, which optimizes a single loss.
    """

    def __init__(self, loop, loss, train_op, inputs, data_flow, feed_dict=None,
                 metric_name='loss'):
        """
        Construct a new :class:`LossTrainer`.

        Args:
            loop (TrainLoop): The training loop object.
            loss (tf.Tensor): The training loss.
            train_op (tf.Operation): The training operation.
            inputs (list[tf.Tensor]): The input placeholders. The number of
                tensors, and the order of tensors, should both match the arrays
                of each mini-batch data, provided by `data_flow`.
            data_flow (DataFlow): The training data flow. Each mini-batch must
                contain one array for each placeholder in `inputs`.
            feed_dict: The feed dict for training.  It will be merged with
                the arrays provided by `data_flow` in each step.
                (default :obj:`None`)
            metric_name (str): The metric name for collecting training loss.
        """
        super(LossTrainer, self).__init__(
            loop=loop, train_op=train_op, inputs=inputs, data_flow=data_flow,
            feed_dict=feed_dict, metrics={metric_name: loss}
        )

    @property
    def loss(self):
        """Get the training loss."""
        return list(self.metrics.values())[0]

    @property
    def metric_name(self):
        """Get the metric name for collecting training loss."""
        return list(self.metrics.keys())[0]

    @deprecated_arg('feed_dict', version='0.1')
    def run(self, feed_dict=None):
        """
        Run training loop.

        Args:
            feed_dict: DEPRECATED.  The extra feed dict to be merged with
                the already configured dict.  (default :obj:`None`)
        """
        old_feed_dict = self._feed_dict
        try:
            if feed_dict is not None:  # pragma: no cover
                self._feed_dict = merge_feed_dict(self._feed_dict, feed_dict)
            super(LossTrainer, self).run()
        finally:
            self._feed_dict = old_feed_dict
