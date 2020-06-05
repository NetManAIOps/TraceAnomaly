import json
import os
import sys
from pprint import pformat

import imageio
import numpy as np
import six
from fs import open_fs
from fs.base import FS
from fs.errors import NoSysPath

from tfsnippet.utils import makedirs, get_config_defaults
from .jsonutils import JsonEncoder

__all__ = ['MLResults']


def ensure_unicode_path(path):
    if isinstance(path, six.binary_type):
        path = path.decode('utf-8')
    return path


class MLResults(object):
    """
    Class to help save results of machine learning experiments.

    Usage::

        results = MLResults()
        results.update_metrics(acc=0.9)
    """

    def __init__(self, result_dir=None, script_name=None):
        """
        Construct a new :class:`MLResults` instance.

        Args:
            result_dir (str or fs.base.FS): A local directory path, a URI
                recognizable by `PyFilesystem <https://www.pyfilesystem.org/>`_,
                or an instance of :class:`fs.base.FS`.  It will be used as
                the result directory, while all the result files will be
                stored within it.  If not specified, will create a local
                directory "./results/<script_name>/".
            script_name (str): The name of the main script.
                If not specified, will use the file name (excluding
                the extension ".py") of the main module.
        """
        if result_dir is None:
            if script_name is None:
                script_name = os.path.splitext(
                    os.path.split(
                        os.path.abspath(sys.modules['__main__'].__file__)
                    )[1]
                )[0]

            # The ``env["MLSTORAGE_EXPERIMENT_ID"]`` would be set if the
            # program is run via `mlrun` from MLStorage.  See
            # `MLStorage Server <https://github.com/haowen-xu/mlstorage-server>`_
            # and
            # `MLStorage Client <https://github.com/haowen-xu/mlstorage-client>`_
            # for details.
            if os.environ.get('MLSTORAGE_EXPERIMENT_ID'):
                # use the current working directory as the result directory
                # if run via `mlrun` from MLStorage.
                result_dir = os.getcwd()
            else:
                result_dir = os.path.join('./results', script_name)
                if not os.path.isdir(result_dir):
                    makedirs(result_dir, exist_ok=True)

        if not isinstance(result_dir, FS):
            try:
                # attempt to create the result directory automatically
                self._fs = open_fs(result_dir, create=True)
            except TypeError:
                self._fs = open_fs(result_dir)
        else:
            self._fs = result_dir

        # the dict to collect metrics
        self._metrics_dict = {}

    @property
    def fs(self):
        """
        Get the virtual file system of the result directory.

        Returns:
            FS: The virtual file system of the result directory.
        """
        return self._fs

    def close(self):
        """
        Close the results object (especially the underlying VFS).
        """
        self._fs.close()

    @property
    def metrics_dict(self):
        """
        Get the metrics dict.

        Returns:
            dict[str, any]: The metrics dict.
        """
        return self._metrics_dict

    def _commit_metrics(self):
        json_result = json.dumps(self._metrics_dict, sort_keys=True,
                                 cls=JsonEncoder)
        with self.fs.open('result.json', 'w', encoding='utf-8') as f:
            f.write(json_result)

    def update_metrics(self, metrics=None, **kwargs):
        """
        Update the metric values.

        The updated metrics will be immediately written to "result.json"
        under the result directory.

        Args:
            metrics (dict[str, any]): The metrics dict.
            \\**kwargs: The metrics dict, specified as named arguments.
        """
        def collect_dict(d):
            for k, v in six.iteritems(d):
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                self._metrics_dict[k] = v

        if metrics:
            collect_dict(metrics)
        if kwargs:
            collect_dict(kwargs)
        self._commit_metrics()

    def format_metrics(self):
        """
        Format the metric values as string.

        Returns:
            str: The formatted metric values.
        """
        return pformat(self.metrics_dict)

    def system_path(self, path):
        """
        Resolve `path` into system path.

        Args:
            path (str): The directory path, relative to the result directory.

        Returns:
            str: The resolved system path.

        Raises:
            RuntimeError: If the `path` cannot be resolved as system path.
        """
        try:
            return self.fs.getsyspath(ensure_unicode_path(path))
        except NoSysPath as e:
            raise RuntimeError('`path` cannot be resolved into system absolute '
                               'path: {}'.format(path), e)

    def make_dirs(self, path, exist_ok=False):
        """
        Create a directory and all its parent directories.

        Args:
            path (str): The directory path, relative to the result directory.
            exist_ok (bool): If :obj:`False`, will raise exception if
                `path` already is a directory.
        """
        self.fs.makedirs(ensure_unicode_path(path), recreate=exist_ok)

    makedirs = make_dirs

    def save_image(self, path, im, format=None, **kwargs):
        """
        Save an image into the result directory.

        Args:
            path (str): The path of the image file.
            im: The image object.
            format (str or None): The format of the image.
                If not specified, will guess according to `path`.
            \\**kwargs: Other parameters to be passed to `imageio.imwrite`.
        """
        if format is None:
            _, extension = os.path.splitext(path)
            from imageio import formats

            # search for a format that can write
            for fmt in formats:
                if extension in fmt.extensions:
                    format = fmt.name
                    break

        with self.fs.open(ensure_unicode_path(path), 'wb') as f:
            return imageio.imwrite(f, im, format=format, **kwargs)

    imwrite = save_image

    def save_config(self, config):
        """
        Save `config` object into the result directory.

        This will write `config.json` and `config.defaults.json`.

        Args:
            config (Config): The config object.
        """
        def to_json(d):
            s = json.dumps(d, sort_keys=True, cls=JsonEncoder)
            if not isinstance(s, six.binary_type):
                s = s.encode('utf-8')
            return s

        defaults_dict = get_config_defaults(config)
        config_dict = {k: v for k, v in six.iteritems(config.to_dict())
                       if k not in defaults_dict or defaults_dict[k] != v}

        defaults_dict = to_json(defaults_dict)
        config_dict = to_json(config_dict)

        with self.fs.open(ensure_unicode_path('config.json'), 'wb') as f:
            f.write(config_dict)
        with self.fs.open(ensure_unicode_path('config.defaults.json'),
                          'wb') as f:
            f.write(defaults_dict)
