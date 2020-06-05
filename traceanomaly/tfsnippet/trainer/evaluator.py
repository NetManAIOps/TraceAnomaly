from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import six
import tensorflow as tf

from tfsnippet.dataflows import DataFlow
from tfsnippet.utils import get_default_session_or_error
from tfsnippet.scaffold import TrainLoop

from .feed_dict import resolve_feed_dict, merge_feed_dict
from .hooks import HookList

__all__ = ['auto_batch_weight', 'Evaluator']


def auto_batch_weight(*batch_arrays):
    """
    Automatically inspect the metric weight for an evaluation mini-batch.

    Args:
        *batch_arrays: Mini-batch arrays.  The ``.size`` of the first array
            will be used as the weight.

    Returns:
        The inspected weight, or 1. if any error occurs during inspection.
    """
    try:
        return batch_arrays[0].size
    except Exception:
        return 1.


class Evaluator(object):
    """
    Class to compute evaluation metrics.

    It is a common practice to compute one or more metrics for evaluation
    and validation during the training process.  This class provides a
    convenient interface for computing metrics by mini-batches.
    """

    def __init__(self, loop, metrics, inputs, data_flow, feed_dict=None,
                 time_metric_name='eval_time',
                 batch_weight_func=auto_batch_weight):
        """
        Construct a new :class:`Evaluator`.

        Args:
            loop (TrainLoop): The training loop object.
            metrics (Tensor or dict[str, Tensor]):
                The validation loss metric, or a dict of metrics.
                All the metrics must be 0-d tensors.

                If only a loss is specified, the default validation loss
                name ``loop.valid_metric_name`` will be used as its name.
                Otherwise if a dict is specified, the keys will be used
                as the names of each metric.
            inputs (list[tf.Tensor]): The input placeholders.
                The number of tensors, and the order of tensors, should
                both match the arrays of each mini-batch data, provided
                by `data_flow`.
            data_flow (DataFlow): The validation data flow.
            feed_dict (dict[tf.Tensor, any]): The fixed feed dict for
                validation.  It will be merged with `inputs` and the
                argument of ``run(feed_dict)``. (default :obj:`None`)
            time_metric_name (None or str): The metric name for collecting
                evaluation time usage.  Specify :obj:`None` to suppress
                the time usage metric. (default "eval_time")
            batch_weight_func ((\*arrays) -> float or None): Specify how
                to compute the metric weight for each mini-batch.  If
                :obj:`None`, will use 1. as the metric weight.
                (default :func:`auto_batch_weight`)
        """
        if not isinstance(metrics, (dict, OrderedDict)):
            metrics = {loop.valid_metric_name: metrics}
        metrics = OrderedDict([
            (str(k),
             tf.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v)
            for k, v in six.iteritems(metrics)
        ])
        for v in six.itervalues(metrics):
            if v.get_shape() is not None and len(v.get_shape()) != 0:
                raise ValueError('Metric is not a scalar tensor: {!r}'.
                                 format(v))

        self._before_run = HookList()
        self._after_run = HookList()
        self._loop = loop
        self._metrics = metrics
        self._inputs = list(inputs or ())
        self._data_flow = data_flow
        self._feed_dict = dict(feed_dict or ())
        self._time_metric_name = time_metric_name
        self._batch_weight_func = batch_weight_func
        self._last_metrics_dict = {}  # store the metrics of last evaluation

    @property
    def before_run(self):
        """
        Get the hooks run before evaluation.

        Returns:
            HookList: The hook list.
        """
        return self._before_run

    @property
    def after_run(self):
        """
        Get the hooks run after evaluation.

        Returns:
            HookList: The hook list.
        """
        return self._after_run

    @property
    def loop(self):
        """
        Get the training loop object.

        Returns:
            TrainLoop: The training loop object.
        """
        return self._loop

    @property
    def metrics(self):
        """
        Get the metrics to compute.

        Returns:
            OrderedDict[str, tf.Tensor]: The metrics to compute.
        """
        return self._metrics

    @property
    def inputs(self):
        """
        Get the input placeholders.

        Returns:
            list[tf.Tensor]: The input placeholders.
        """
        return self._inputs

    @property
    def data_flow(self):
        """
        Get the validation data flow.

        Returns:
            DataFlow: The validation data flow.
        """
        return self._data_flow

    @property
    def feed_dict(self):
        """
        Get the fixed feed dict.

        Returns:
            dict[tf.Tensor, any]: The fixed feed dict.
        """
        return self._feed_dict

    @property
    def time_metric_name(self):
        """Get the metric name for collecting evaluation time usage."""
        return self._time_metric_name

    @property
    def batch_weight_func(self):
        """Get the function to compute the metric weight for each mini-batch."""
        return self._batch_weight_func

    @property
    def last_metrics_dict(self):
        """
        Get the metric values from last evaluation.

        Returns:
            dict[str, any]: The metric values dict.
        """
        return self._last_metrics_dict

    def _run_batch(self, session, feed_dict):
        return session.run(list(six.itervalues(self.metrics)),
                           feed_dict=feed_dict)

    def run(self, feed_dict=None):
        """
        Run evaluation.

        Args:
            feed_dict: The extra feed dict to be merged with the already
                configured dict.  (default :obj:`None`)
        """
        @contextmanager
        def timeit():
            if self.time_metric_name is not None:
                with self.loop.timeit(self.time_metric_name):
                    yield
            else:
                yield

        session = get_default_session_or_error()
        metric_tensors = list(six.itervalues(self.metrics))
        metric_names = list(six.iterkeys(self.metrics))
        metric_values = []
        metric_weights = []

        with timeit():
            # run before evaluation hooks
            self.before_run.call_hooks()

            for batch_data in self.data_flow:
                # prepare for the batch feed dict
                feed_dict = resolve_feed_dict(
                    merge_feed_dict(
                        self.feed_dict,
                        feed_dict,
                        zip(self.inputs, batch_data)
                    )
                )

                # inspect the batch weight
                if self._batch_weight_func is not None:
                    batch_weight = self._batch_weight_func(*batch_data)
                else:
                    batch_weight = 1.
                metric_weights.append(batch_weight)

                # run the mini-batch
                batch_values = self._run_batch(session, feed_dict)
                for i, v in enumerate(batch_values):
                    if len(np.asarray(v).shape) != 0:  # pragma: no cover
                        raise ValueError(
                            'Metric is not a scalar: tensor {!r}, value {!r}.'.
                            format(v, metric_tensors[i])
                        )

                # accumulate the metrics
                metric_values.append(np.asarray(batch_values))

            # now merge all batch metrics and do logging
            if metric_values:
                metric_values = np.average(
                    np.stack(metric_values, axis=0),
                    axis=0,
                    weights=np.asarray(metric_weights),
                )
                assert(len(metric_names) == len(metric_values))
                self._last_metrics_dict = metrics_dict = {
                    k: v for k, v in zip(metric_names, metric_values)
                }
                self.loop.collect_metrics(metrics_dict)

            # run after evaluation hooks
            self.after_run.call_hooks()
