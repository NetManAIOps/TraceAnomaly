import copy
import os
import shutil
import warnings
from logging import getLogger

import tensorflow as tf

from tfsnippet.utils import (DisposableContext, TemporaryDirectory, makedirs,
                             deprecated)
from .variable_saver import VariableSaver

__all__ = ['EarlyStopping', 'EarlyStoppingContext', 'early_stopping']


class EarlyStopping(DisposableContext):
    """
    Early-stopping context object.

    This class provides a object for memorizing the parameters for best
    metric, in an early-stopping context.  An example of using this context:

    .. code-block:: python

        with EarlyStopping(param_vars) as es:
            ...
            es.update(loss, global_step)
            ...

    Where ``es.update(loss, global_step)`` should cause the parameters to
    be saved on disk if `loss` is better than the current best metric.
    One may also get the current best metric via ``es.best_metric``.

    Notes:
        If no loss is given via ``es.update``, then the variables
        would keep their latest values when closing an early-stopping object.
    """

    def __init__(self, param_vars, initial_metric=None, checkpoint_dir=None,
                 smaller_is_better=True, restore_on_error=False,
                 cleanup=True, name=None):
        """
        Construct the :class:`EarlyStopping`.

        Args:
            param_vars (list[tf.Variable] or dict[str, tf.Variable]): List or
                dict of variables to be memorized. If a dict is specified, the
                keys of the dict would be used as the serializations keys via
                :class:`VariableSaver`.
            initial_metric (float or tf.Tensor or tf.Variable): The initial best
                metric (for recovering from previous session).
            checkpoint_dir (str): The directory where to save the checkpoint
                files.  If not specified, will use a temporary directory.
            smaller_is_better (bool): Whether or not it is better to have
                smaller metric values? (default :obj:`True`)
            restore_on_error (bool): Whether or not to restore the memorized
                parameters even on error? (default :obj:`False`)
            cleanup (bool): Whether or not to cleanup the checkpoint directory
                on exit? This argument will be ignored if `checkpoint_dir` is
                :obj:`None`, where the temporary directory will always be
                deleted on exit.
            name (str): Name scope of all TensorFlow operations. (default
                "early_stopping").
        """
        # regularize the parameters
        if not param_vars:
            raise ValueError('`param_vars` must not be empty')

        if isinstance(initial_metric, (tf.Tensor, tf.Variable)):
            initial_metric = initial_metric.eval()

        if checkpoint_dir is not None:
            checkpoint_dir = os.path.abspath(checkpoint_dir)

        # memorize the parameters
        self._param_vars = copy.copy(param_vars)
        self._checkpoint_dir = checkpoint_dir
        self._smaller_is_better = smaller_is_better
        self._restore_on_error = restore_on_error
        self._cleanup = cleanup
        self._name = name

        # internal states of the object
        self._best_metric = initial_metric
        self._ever_updated = False
        self._temp_dir_ctx = None
        self._saver = None  # type: VariableSaver

    def _enter(self):
        # open a temporary directory if the checkpoint dir is not specified
        if self._checkpoint_dir is None:
            self._temp_dir_ctx = TemporaryDirectory()
            self._checkpoint_dir = self._temp_dir_ctx.__enter__()
        else:
            makedirs(self._checkpoint_dir, exist_ok=True)

        # create the variable saver
        self._saver = VariableSaver(self._param_vars, self._checkpoint_dir)

        # return self as the context object
        return self

    def _exit(self, exc_type, exc_val, exc_tb):
        try:
            # restore the variables
            # exc_info = (exc_type, exc_val, exc_tb)
            if exc_type is None or exc_type is KeyboardInterrupt or \
                    self._restore_on_error:
                self._saver.restore(ignore_non_exist=True)

        finally:
            # cleanup the checkpoint directory
            try:
                if self._temp_dir_ctx is not None:
                    self._temp_dir_ctx.__exit__(exc_type, exc_val, exc_tb)
                elif self._cleanup:
                    if os.path.exists(self._checkpoint_dir):
                        shutil.rmtree(self._checkpoint_dir)
            except Exception:  # pragma: no cover
                getLogger(__name__).error(
                    'Failed to cleanup validation save dir %r.',
                    self._checkpoint_dir, exc_info=True
                )

            # warning if metric never updated
            if not self._ever_updated:
                warnings.warn(
                    'Early-stopping metric has never been updated. '
                    'The variables will keep their latest values. '
                    'Did you forget to add corresponding metric?'
                )

    def update(self, metric, global_step=None):
        """
        Update the best metric.

        Args:
            metric (float): New metric value.
            global_step (int): Optional global step counter.

        Returns:
            bool: Whether or not the best loss has been updated?
        """
        self._require_entered()
        self._ever_updated = True
        if self._best_metric is None or \
                (self._smaller_is_better and metric < self._best_metric) or \
                (not self._smaller_is_better and metric > self._best_metric):
            self._saver.save(global_step)
            self._best_metric = metric
            return True
        return False

    @property
    def best_metric(self):
        """Get the current best loss."""
        return self._best_metric

    @property
    def ever_updated(self):
        """Check whether or not `update` method has ever been called."""
        return self._ever_updated


EarlyStoppingContext = EarlyStopping  # legacy alias for EarlyStopping


@deprecated('use :class:`EarlyStopping` instead.', version='0.1')
def early_stopping(*args, **kwargs):  # pragma: no cover
    return EarlyStopping(*args, **kwargs)
