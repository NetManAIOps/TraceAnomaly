import os
import shutil
from contextlib import contextmanager

import requests
import six
import sys

from filelock import FileLock
from tqdm import tqdm

from .archive_file import Extractor
from .imported import makedirs

if six.PY2:
    from urlparse import urlparse
else:
    from urllib.parse import urlparse

__all__ = [
    'get_cache_root', 'set_cache_root', 'CacheDir',
]

_cache_root = None


@contextmanager
def _maybe_tqdm(tqdm_enabled, **kwargs):
    if tqdm_enabled:
        with tqdm(**kwargs) as t:
            yield t
    else:
        yield None


def get_cache_root():
    """
    Get the cache root directory.

    Returns:
        str: Path of the cache root directory.
    """
    if _cache_root is None:
        return os.path.abspath(
            os.environ.get(
                'TFSNIPPET_CACHE_ROOT',
                os.path.expanduser('~/.tfsnippet/cache')
            )
        )
    return _cache_root


def set_cache_root(cache_root):
    """
    Set the root cache directory.

    Args:
        cache_root (str): The cache root directory.  It will be normalized
            to absolute path.
    """
    global _cache_root
    _cache_root = os.path.abspath(cache_root)


def guess_show_progress_arg(progress_file, show_progress):  # pragma: no cover
    if show_progress is None:
        if hasattr(progress_file, 'isatty'):
            return progress_file.isatty()
        else:
            return False
    else:
        return show_progress


def guess_filename_from_uri(uri):
    parsed_uri = urlparse(uri)
    filename = parsed_uri.path.rsplit('/', 1)[-1]
    if not filename:  # pragma: no cover
        raise ValueError('`filename` cannot be inferred.')
    return filename


def guess_extract_dir_from_filename(filename):
    extract_dir = filename.split('.', 1)[0]
    if not extract_dir:  # pragma: no cover
        raise ValueError('`extract_dir` cannot be inferred.')
    return extract_dir


class CacheDir(object):
    """Class to manipulate a cache directory."""

    def __init__(self, name, cache_root=None):
        """
        Construct a new :class:`CacheDir`.

        Args:
            name (str): The name of the sub-directory under `cache_root`.
            cache_root (str or None): The cache root directory.  If not
                specified, use ``get_cache_root()``.
        """
        if not name:
            raise ValueError('`name` is required.')
        if cache_root is None:
            cache_root = get_cache_root()
        self._name = name
        self._cache_root = os.path.abspath(cache_root)
        self._path = os.path.abspath(os.path.join(self._cache_root, name))

    @property
    def name(self):
        """Get the name of this cache directory under `cache_root`."""
        return self._name

    @property
    def cache_root(self):
        """Get the cache root directory."""
        return self._cache_root

    @property
    def path(self):
        """Get the absolute path of this cache directory."""
        return self._path

    def resolve(self, sub_path):
        """
        Resolve a sub path relative to ``self.path``.

        Args:
            sub_path: The sub path to resolve.

        Returns:
            The resolved absolute path of `sub_path`.
        """
        return os.path.join(self.path, sub_path)

    @contextmanager
    def _lock_file(self, file_path):
        lock_file = file_path + '.lock'
        parent_dir = os.path.split(lock_file)[0]
        if not os.path.isdir(parent_dir):
            makedirs(parent_dir, exist_ok=True)
        with FileLock(lock_file):
            yield

    def _download(self, uri, file_path, show_progress, progress_file):
        if not os.path.isfile(file_path):
            temp_file = file_path + '._downloading_'
            try:
                desc = 'Downloading {}'.format(uri)
                with _maybe_tqdm(tqdm_enabled=show_progress, desc=desc,
                                 unit='B', unit_scale=True, unit_divisor=1024,
                                 miniters=1, file=progress_file) as t, \
                        open(temp_file, 'wb') as f:
                    req = requests.get(uri, stream=True)
                    if req.status_code != 200:
                        raise IOError('HTTP Error {}: {}'.
                                      format(req.status_code, req.content))

                    # detect the total length
                    if t is not None:
                        cont_length = req.headers.get('Content-Length')
                        if cont_length is not None:
                            try:
                                t.total = int(cont_length)
                            except ValueError:  # pragma: no cover
                                pass

                    # do download the content
                    for chunk in req.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                            if t is not None:
                                t.update(len(chunk))
            except BaseException:
                if os.path.isfile(temp_file):  # pragma: no cover
                    os.remove(temp_file)
                raise
            else:
                os.rename(temp_file, file_path)
        return file_path

    def download(self, uri, filename=None, show_progress=None,
                 progress_file=sys.stderr):
        """
        Download a file into this :class:`CacheDir`.

        Args:
            uri (str): The URI to be retrieved.
            filename (str): The filename to use as the downloaded file.
                If `filename` already exists in this :class:`CacheDir`,
                will not download `uri`.  Default :obj:`None`, will
                automatically infer `filename` according to `uri`.
            show_progress (bool): Whether or not to show interactive
                progress bar?  If not specified, will show progress only
                if `progress_file` is `std.stdout` or `std.stderr`, and
                if `progress_file.isatty()` is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)

        Returns:
            str: The absolute path of the downloaded file.

        Raises:
            ValueError: If `filename` cannot be inferred.
        """
        # check the arguments
        show_progress = guess_show_progress_arg(progress_file, show_progress)

        if filename is None:
            filename = guess_filename_from_uri(uri)
        file_path = os.path.abspath(os.path.join(self.path, filename))

        # download the file
        with self._lock_file(file_path):
            return self._download(
                uri, file_path, show_progress=show_progress,
                progress_file=progress_file
            )

    def _extract_file(self, archive_file, extract_path, show_progress,
                      progress_file):
        if not os.path.isdir(extract_path):
            temp_path = extract_path + '._extracting_'
            if show_progress:
                progress_file.write('Extracting {} ... '.format(archive_file))
                progress_file.flush()
            try:
                with Extractor.open(archive_file) as extractor:
                    for name, file_obj in extractor:
                        file_path = os.path.join(temp_path, name)
                        file_dir = os.path.split(file_path)[0]
                        if not os.path.isdir(file_dir):
                            makedirs(file_dir, exist_ok=True)
                        with open(file_path, 'wb') as dst_obj:
                            shutil.copyfileobj(file_obj, dst_obj)
            except BaseException:
                if show_progress:
                    progress_file.write('error\n')
                    progress_file.flush()
                if os.path.isdir(temp_path):  # pragma: no cover
                    shutil.rmtree(temp_path)
                raise
            else:
                if show_progress:
                    progress_file.write('done\n')
                    progress_file.flush()
                os.rename(temp_path, extract_path)
        return extract_path

    def extract_file(self, archive_file, extract_dir=None, show_progress=None,
                     progress_file=sys.stderr):
        """
        Extract an archive file into this :class:`CacheDir`.

        Args:
            archive_file (str): The path of the archive file.
            extract_dir (str): The name to use as the extracted directory.
                If `extract_dir` already exists in this :class:`CacheDir`,
                will not extract `archive_file`.  Default :obj:`None`, will
                automatically infer `extract_dir` according to `archive_file`.
            show_progress (bool): Whether or not to show interactive
                progress bar?  If not specified, will show progress only
                if `progress_file` is `std.stdout` or `std.stderr`, and
                if `progress_file.isatty()` is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)

        Returns:
            str: The absolute path of the extracted directory.

        Raises:
            ValueError: If `extract_dir` cannot be inferred.
        """
        # check the arguments
        show_progress = guess_show_progress_arg(progress_file, show_progress)

        archive_file = os.path.abspath(archive_file)
        filename = os.path.split(archive_file)[-1]
        if extract_dir is None:
            extract_dir = guess_extract_dir_from_filename(filename)
        extract_path = os.path.abspath(os.path.join(self.path, extract_dir))

        # extract the file
        with self._lock_file(archive_file):
            return self._extract_file(
                archive_file, extract_path, show_progress=show_progress,
                progress_file=progress_file
            )

    def download_and_extract(self, uri, filename=None, extract_dir=None,
                             show_progress=None, progress_file=sys.stderr):
        """
        Download a file into this :class:`CacheDir`, and extract it.

        Args:
            uri (str): The URI to be retrieved.
            filename (str): The filename to use as the downloaded file.
                If `filename` already exists in this :class:`CacheDir`,
                will not download `uri`.  Default :obj:`None`, will
                automatically infer `filename` according to `uri`.
            extract_dir (str): The name to use as the extracted directory.
                If `extract_dir` already exists in this :class:`CacheDir`,
                will not extract `archive_file`.  Default :obj:`None`, will
                automatically infer `extract_dir` according to `filename`.
            show_progress (bool): Whether or not to show interactive
                progress bar?  If not specified, will show progress only
                if `progress_file` is `std.stdout` or `std.stderr`, and
                if `progress_file.isatty()` is :obj:`True`.
            progress_file: The file object where to write the progress.
                (default :obj:`sys.stderr`)

        Returns:
            str: The absolute path of the extracted directory.

        Raises:
            ValueError: If `filename` or `extract_dir` cannot be inferred.
        """
        # check the arguments
        show_progress = guess_show_progress_arg(progress_file, show_progress)

        if filename is None:
            filename = guess_filename_from_uri(uri)
        file_path = os.path.abspath(os.path.join(self.path, filename))

        if extract_dir is None:
            extract_dir = guess_extract_dir_from_filename(filename)
        extract_path = os.path.abspath(os.path.join(self.path, extract_dir))

        # download and extract the file
        with self._lock_file(file_path):
            if not os.path.isdir(extract_path):
                archive_file = self._download(
                    uri, file_path, show_progress=show_progress,
                    progress_file=progress_file
                )
                self._extract_file(
                    archive_file, extract_path, show_progress=show_progress,
                    progress_file=progress_file
                )
                # download the archive file if we successfully extracted it.
                os.remove(file_path)
            return extract_path

    def purge_all(self):
        """Delete everything in this :class:`CacheDir`."""
        shutil.rmtree(self.path)
