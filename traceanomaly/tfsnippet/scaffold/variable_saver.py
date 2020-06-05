import os
from logging import getLogger

import tensorflow as tf

from tfsnippet.shortcuts import VarScopeObject
from tfsnippet.utils import (makedirs, reopen_variable_scope,
                             get_default_session_or_error)

__all__ = ['VariableSaver']


class VariableSaver(VarScopeObject):
    """Version controlled saving and restoring TensorFlow variables."""

    def __init__(self, variables, save_dir, max_versions=2,
                 filename='variables.dat', latest_file='latest',
                 save_meta=True, name=None, scope=None):
        """
        Construct the :class:`VariableSaver`.

        Args:
            variables (collections.Iterable[tf.Variable] or dict[str, any]):
                List of variables, or dict of variables with explicit keys,
                which should be saved and restored.
            save_dir (str): Directory where to place the saved variables.
            max_versions (int): Maximum versions to keep in the directory
                (Default is 2). At least 2 versions should be kept, in order to
                prevent corrupted checkpoint files caused by IO failure.
            filename (str): Name of the files of variable values (default is
                ``variables.dat``).
            latest_file (str): Name of the file which organizes the checkpoint
                versions (default is ``latest``).
            save_meta (bool): Whether or not to save meta graph (default
                is :obj:`True`).
            name (str): Name of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
            scope (str): Scope of this :class:`VariableSaver`
                (argument of :class:`~tfsnippet.utils.VarScopeObject`).
        """
        if not isinstance(variables, dict):
            variables = list(variables)
        if max_versions < 2:
            raise ValueError('At least 2 versions should be kept')

        self.variables = variables
        self.save_dir = os.path.abspath(save_dir)
        self.filename = filename
        self.max_versions = max_versions
        self.latest_file = latest_file
        self.save_meta = save_meta

        super(VariableSaver, self).__init__(scope, name)

        with reopen_variable_scope(self.variable_scope):
            self._saver = tf.train.Saver(
                var_list=self.variables, max_to_keep=self.max_versions,
                name='saver'
            )

    def get_latest_file(self):
        """Get the latest available checkpoint file."""
        return tf.train.latest_checkpoint(self.save_dir, self.latest_file)

    def save(self, global_step=None):
        """
        Save the checkpoint to file.

        Args:
            global_step (int or tf.Tensor): The global step counter.
        """
        sess = get_default_session_or_error()
        makedirs(self.save_dir, exist_ok=True)
        self._saver.save(
            sess,
            os.path.join(self.save_dir, self.filename),
            global_step=global_step,
            latest_filename=self.latest_file,
            write_meta_graph=self.save_meta
        )

    def restore(self, ignore_non_exist=False):
        """
        Restore the checkpoint from file if it exists.

        Args:
            ignore_non_exist (bool): Whether or not to ignore error if the
                checkpoint file does not exist? (default :obj:`False`)

        Raises:
            IOError: If the checkpoint files do not exist, and
                `ignore_non_exist` is not :obj:`True`.
        """
        file_path = self.get_latest_file()
        if file_path:
            sess = get_default_session_or_error()
            self._saver.restore(sess, file_path)
            getLogger(__name__).debug(
                'Restored from checkpoint file %r.', file_path)
        elif not ignore_non_exist:
            raise IOError('Checkpoint file does not exist in directory {}'.
                          format(self.save_dir))

