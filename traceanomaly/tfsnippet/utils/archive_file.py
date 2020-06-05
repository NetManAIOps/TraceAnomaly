import sys
import tarfile
import zipfile

try:
    import rarfile
#    rarfile.PATH_SEP = '/'
except ImportError:  # pragma: no cover
    rarfile = None

__all__ = ['Extractor', 'TarExtractor', 'ZipExtractor', 'RarExtractor']


TAR_FILE_EXTENSIONS = ('.tar',
                       '.tar.gz', '.tgz',
                       '.tar.bz2', '.tbz', '.tbz2', '.tb2')
if sys.version_info[:2] >= (3, 3):
    TAR_FILE_EXTENSIONS = TAR_FILE_EXTENSIONS + ('.tar.xz', '.txz')


def normalize_archive_entry_name(name):
    """
    Get the normalized name of an archive file entry.
    Args:
        name (str): Name of the archive file entry.

    Returns:
        str: The normalized name.
    """
    return name.replace('\\', '/')


class Extractor(object):
    """
    The base class for all archive extractors.

    .. code-block:: python

        from tfsnippet.utils import Extractor, maybe_close

        with Extractor.open('a.zip') as archive_file:
            for name, f in archive_file:
                with maybe_close(f):  # This file object may not be closeable,
                                      # thus we surround it by ``maybe_close()``
                    print('the content of {} is:'.format(name))
                    print(f.read())
    """

    def __init__(self, archive_file):
        """
        Initialize the base :class:`Extractor` class.

        Args:
            archive_file: The archive file object.
        """
        self._archive_file = archive_file

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def __iter__(self):
        return self.iter_extract()

    def close(self):
        """Close the extractor."""
        if self._archive_file:
            self._archive_file.close()
            self._archive_file = None

    def iter_extract(self):
        """
        Extract files from the archive with an iterator.

        You may simply iterate over a :class:`Extractor` object, which is
        same as calling to this method.

        Yields:
            (str, file-like): Tuples of ``(name, file-like object)``, the
                filename and corresponding file-like object for each file
                in the archive.  The returned file-like object may or may
                not be closeable.  You may surround it by ``maybe_close()``.
        """
        raise NotImplementedError()

    @staticmethod
    def open(file_path):
        """
        Create an :class:`Extractor` instance for given archive file.

        Args:
            file_path (str): The path of the archive file.

        Returns:
            Extractor: The specified extractor instance.

        Raises:
            IOError: If the ``file_path`` is not a supported archive.
        """
        if file_path.endswith('.rar') and rarfile is not None:
            return RarExtractor(file_path)
        elif file_path.endswith('.zip'):
            return ZipExtractor(file_path)
        elif any(file_path.endswith(ext) for ext in TAR_FILE_EXTENSIONS):
            return TarExtractor(file_path)
        else:
            raise IOError('File is not a supported archive file: {!r}'.
                          format(file_path))


class TarExtractor(Extractor):
    """
    Extractor for ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tbz2",
    ".tb2", ".tar.xz", ".txz" files.
    """

    def __init__(self, fpath):
        super(TarExtractor, self).__init__(tarfile.open(fpath, 'r'))

    def iter_extract(self):
        for mi in self._archive_file:
            if not mi.isdir():
                yield (
                    normalize_archive_entry_name(mi.name),
                    self._archive_file.extractfile(mi)
                )


class ZipExtractor(Extractor):
    """Extractor for ".zip" files."""

    def __init__(self, fpath):
        super(ZipExtractor, self).__init__(zipfile.ZipFile(fpath, 'r'))

    def iter_extract(self):
        for mi in self._archive_file.infolist():
            # ignore directory entries
            if mi.filename[-1] == '/':
                continue
            yield (
                normalize_archive_entry_name(mi.filename),
                self._archive_file.open(mi)
            )


class RarExtractor(Extractor):
    """Extractor for ".rar" files."""

    def __init__(self, fpath):
        if rarfile is None:  # pragma: no cover
            raise RuntimeError('Required package not installed: rarfile.')
        super(RarExtractor, self).__init__(rarfile.RarFile(fpath, 'r'))

    def iter_extract(self):
        for mi in self._archive_file.infolist():
            if mi.isdir():
                continue
            yield (
                normalize_archive_entry_name(mi.filename),
                self._archive_file.open(mi)
            )
