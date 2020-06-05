# -*- coding: utf-8 -*-
import functools
import re
from collections import defaultdict, OrderedDict
from itertools import chain

import numpy as np
import six
import tensorflow as tf
from natsort import natsorted

from tfsnippet.utils import (humanize_duration,
                             StatisticsCollector,
                             get_default_session_or_error,
                             DocInherit)

__all__ = [
    'MetricFormatter',
    'DefaultMetricFormatter',
    'MetricLogger',
    'summarize_variables',
]


@DocInherit
class MetricFormatter(object):
    """
    Base class for a training metrics formatter.

    A training metric formatter determines the order of metrics, and the way
    to display the values of these metrics, in :class:`MetricLogger`.
    """

    def sort_metrics(self, names):
        """
        Sort the names of metrics.

        Args:
            names: Iterable metric names.

        Returns:
            list[str]: Sorted metric names.
        """
        raise NotImplementedError()

    def format_metric(self, name, value):
        """
        Format the value of specified metric.

        Args:
            name: Name of the metric.
            value: Value of the metric.

        Returns:
            str: Human readable string representation of the metric value.
        """
        raise NotImplementedError()


class DefaultMetricFormatter(MetricFormatter):
    """
    Default training metric formatter.

    This class sorts the metrics as follows:

    1.  The metrics are first divided into groups according to the suffices
        of their names as follows:

        1.  Names ending with "time" or "timer" should come the first;

        2.  Other metrics should come the second;

        3.  Names ending with "loss" or "cost" should come the third;

        4.  Names ending with "acc", "accuracy", "nll", "lb" or "lower_bound"
            should come the fourth.

    2.  The metrics are then sorted according to their names, within each group.

    The values of the metrics would be formatted into 6-digit real numbers,
    except for metrics with "time" or "timer" as suffices in their names,
    which would be formatted using :func:`~tfsnippet.utils.humanize_duration`.
    """

    METRIC_ORDERS = (
        (-1, re.compile(r'.*timer?$')),
        (998, re.compile(r'.*(loss|cost)$')),
        (999, re.compile(r'(.*(acc(uracy)?|lower_bound))|((^|.*_)(nll|lb))$')),
    )

    def sort_metrics(self, names):
        def sort_key(name):
            for priority, pattern in self.METRIC_ORDERS:
                if pattern.match(name):
                    return priority, name
            return 0, name

        return sorted(names, key=sort_key)

    def format_metric(self, name, value):
        if name.endswith('time') or name.endswith('timer'):
            return humanize_duration(float(value))
        else:
            return '{:.6g}'.format(float(value))


class MetricLogger(object):
    """
    Logger for the training metrics.

    This class provides convenient methods for logging training metrics,
    and for writing metrics onto disk via TensorFlow summary writer.
    The statistics of the metrics could be formatted into human readable
    strings via :meth:`format_logs`.

    An example of using this logger is:

    .. code-block:: python

        logger = MetricLogger(tf.summary.FileWriter(log_dir))
        global_step = 1

        for epoch in range(1, max_epoch+1):
            for batch in DataFlow.arrays(...):
                loss, _ = session.run([loss, train_op], ...)
                logger.collect_metrics({'loss': loss}, global_step)
                global_step += 1

            valid_loss = session.run([loss], ...)
            logger.collect_metrics({'valid_loss': valid_loss}, global_step)
            print('Epoch {}, step {}: {}'.format(
                epoch, global_step, logger.format_logs()))
            logger.clear()
    """

    def __init__(self, summary_writer=None, summary_metric_prefix='',
                 summary_skip_pattern=None, summary_commit_freqs=None,
                 formatter=None):
        """
        Construct the :class:`MetricLogger`.

        Args:
            summary_writer: TensorFlow summary writer.
            summary_metric_prefix (str): The prefix for the metrics committed
                to `summary_writer`.  This will not affect the summaries
                added via :meth:`add_summary`. (default "")
            summary_skip_pattern (str or regex): Metrics matching this pattern
                will be excluded from `summary_writer`. (default :obj:`None`)
            summary_commit_freqs (dict[str, int] or None): If specified,
                a metric will be committed to `summary_writer` no more frequent
                than ``summary_commit_freqs[metric]``. (default :obj:`None`)
            formatter (MetricFormatter): Metric formatter for this logger.
                If not specified, will use an instance of
                :class:`DefaultMetricFormatter`.
        """
        if formatter is None:
            formatter = DefaultMetricFormatter()
        if summary_skip_pattern is not None:
            summary_skip_pattern = re.compile(summary_skip_pattern)
        self._formatter = formatter
        self._summary_writer = summary_writer
        self._summary_metric_prefix = summary_metric_prefix
        self._summary_skip_pattern = summary_skip_pattern
        self._summary_commit_freqs = dict(summary_commit_freqs or ())

        # accumulators for various metrics
        self._metrics = defaultdict(StatisticsCollector)
        self._metrics_skip_counter = {}
        self.clear()

    def clear(self):
        """Clear all the metric statistics."""
        # Instead of calling ``self._metrics.clear()``, we reset every
        # collector object (so that they can be reused).
        # This may help reduce the time cost on GC.
        for k, v in six.iteritems(self._metrics):
            v.reset()
        self._metrics_skip_counter.clear()
        for k, v in six.iteritems(self._summary_commit_freqs):
            self._metrics_skip_counter[k] = v - 1

    def collect_metrics(self, metrics, global_step=None):
        """
        Collect the statistics of metrics.

        Args:
            metrics (dict[str, float or np.ndarray or ScheduledVariable]):
                Dict from metrics names to their values.
                For :meth:`format_logs`, there is no difference between
                calling :meth:`collect_metrics` only once, with an array
                of metric values; or calling :meth:`collect_metrics` multiple
                times, with one value at each time.
                However, for the TensorFlow summary writer, only the mean of
                the metric values would be recorded, if calling
                :meth:`collect_metrics` with an array.
            global_step (int or tf.Variable or tf.Tensor): The global step
                counter. (optional)
        """
        from tfsnippet.trainer import ScheduledVariable
        tf_summary_values = []
        for k, v in six.iteritems(metrics):
            if isinstance(v, ScheduledVariable):
                v = v.get()
            v = np.asarray(v)
            self._metrics[k].collect(v)

            if self._summary_writer is not None and \
                    (self._summary_skip_pattern is None or
                     not self._summary_skip_pattern.match(k)):
                skip_count = self._metrics_skip_counter.get(k, 0)
                freq_limit = self._summary_commit_freqs.get(k, 1)
                if skip_count + 1 >= freq_limit:
                    self._metrics_skip_counter[k] = 0
                    tag = self._summary_metric_prefix + k
                    tf_summary_values.append(
                        tf.summary.Summary.Value(
                            tag=tag, simple_value=v.mean()
                        )
                    )
                else:
                    self._metrics_skip_counter[k] = skip_count + 1

        if tf_summary_values:
            summary = tf.summary.Summary(value=tf_summary_values)
            if global_step is not None and \
                    isinstance(global_step, (tf.Variable, tf.Tensor)):
                global_step = get_default_session_or_error().run(global_step)
            self._summary_writer.add_summary(summary, global_step=global_step)

    def format_logs(self):
        """
        Format the metric statistics as human readable strings.

        Returns:
            str: The formatted metric statistics.
        """
        buf = []
        for key in self._formatter.sort_metrics(six.iterkeys(self._metrics)):
            metric = self._metrics[key]
            if metric.has_value:
                name = key.replace('_', ' ')
                val = self._formatter.format_metric(key, metric.mean)
                if metric.counter > 1:
                    std = ' (±{})'.format(
                        self._formatter.format_metric(key, metric.stddev))
                else:
                    std = ''
                buf.append('{}: {}{}'.format(name, val, std))
        return '; '.join(buf)


def _var_size(v):
    return int(np.prod(v.get_shape().as_list(), dtype=np.int32))


class _VarDict(object):

    def __init__(self, variables):
        if isinstance(variables, list):
            self.all = OrderedDict([
                (v.name.rsplit(':', 1)[0], v)
                for v in variables
            ])
        else:
            self.all = OrderedDict(variables)

    def select(self, predicate, strip_prefix=0):
        return _VarDict(OrderedDict([
            (k[strip_prefix:], v)
            for k, v in six.iteritems(self.all)
            if predicate(k, v)
        ]))

    def empty(self):
        return not self.all

    def total_size(self):
        return sum(_var_size(v) for v in six.itervalues(self.all))


def _format_title(title, var_size, min_hr_len):
    title = str(title)
    right = '({:,} in total)'.format(var_size)
    length = max(min_hr_len, len(title) + len(right) + 1)
    return '{}{}{}'.format(
        title, ' ' * (length - len(title) - len(right)), right)


def _format_var_table(var_dict, title=None, post_title_hr='-', min_hr_len=0,
                      sort_by_names=False):
    names = list(var_dict.all)
    if sort_by_names:
        names = natsorted(names)
    var_size = var_dict.total_size()
    if not names:
        return ''
    the_title = _format_title(title, var_size, min_hr_len)
    title_len = len(the_title)
    variables = [var_dict.all[n] for n in names]
    shapes = ['{!r}'.format(tuple(v.get_shape().as_list())) for v in variables]
    sizes = ['{:,}'.format(_var_size(v)) for v in variables]
    name_len = max(map(len, names))
    shape_len = max(map(len, shapes))
    size_len = max(map(len, sizes))
    hr_len = max(name_len + shape_len + size_len + 4, title_len, min_hr_len)
    the_title = _format_title(title, var_size, hr_len)
    pad_len = hr_len - (name_len + shape_len + size_len + 4)

    ret = [the_title]
    if post_title_hr:
        ret.append(post_title_hr * hr_len)
    for name, shape, size in zip(names, shapes, sizes):
        ret.append(
            '{name:<{name_len}}  {shape:<{shape_len}}  '
            '{size:>{size_len}}'.format(
                name=name, shape=shape, size=size,
                name_len=name_len + pad_len,
                shape_len=shape_len,
                size_len=size_len
            )
        )
    return '\n'.join(ret)


def summarize_variables(variables,
                        title='Variables Summary',
                        other_variables_title='Other Variables',
                        groups=None,
                        sort_by_names=False):
    """
    Get a formatted summary about the variables.

    Args:
        variables (list[tf.Variable] or dict[str, tf.Variable]): List or
            dict of variables to be summarized.
        title (str): Title of this summary.
        other_variables_title (str): Title of the "Other Variables".
        groups (None or list[str]): List of separated variable groups, each
            summarized in a table.  (default :obj:`None`)
        sort_by_names (bool): Whether or not to sort the variables within
            each group by their names? (if not :obj:`True`, will display
            the variables according to their natural order)

    Returns:
        str: Formatted summary about the variables.
    """
    if not groups:
        return _format_var_table(_VarDict(variables), title=title,
                                 sort_by_names=sort_by_names)
    var_dict = _VarDict(variables)
    groups = [g.rstrip('/') + '/' for g in groups if g.rstrip('/')]

    max_line_len = 0
    buf = []

    # do two-pass, so as to align the length of each group
    for _ in range(2):
        the_title = _format_title(title, var_dict.total_size(), max_line_len)
        title_len = len(the_title)
        buf = [the_title, '']
        matched_k = set()

        def group_filter(k, v, g):
            if not k.startswith(g):
                return False
            matched_k.add(k)
            return True

        for j, g in enumerate(groups):
            g_var_dict = var_dict.select(
                functools.partial(group_filter, g=g), len(g))
            if not g_var_dict.empty():
                g_table = _format_var_table(g_var_dict,
                                            title=g,
                                            min_hr_len=title_len,
                                            sort_by_names=sort_by_names)
                if j > 0:
                    buf.append('')
                buf.append(g_table)
        if not matched_k:
            return summarize_variables(var_dict.all, title=title, groups=None,
                                       sort_by_names=True)

        o_var_dict = var_dict.select(lambda k, v: k not in matched_k)
        if not o_var_dict.empty():
            o_table = _format_var_table(o_var_dict,
                                        title=other_variables_title,
                                        min_hr_len=title_len,
                                        sort_by_names=sort_by_names)
            buf.append('')
            buf.append(o_table)

        max_line_len = max(*map(len, chain(*(b.split('\n') for b in buf))))

    buf[1] = '=' * max_line_len
    return '\n'.join(buf)
