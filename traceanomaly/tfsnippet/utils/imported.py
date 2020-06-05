import os
import sys

try:
    from tempfile import TemporaryDirectory
except ImportError:
    from backports.tempfile import TemporaryDirectory

__all__ = [
    'TemporaryDirectory', 'makedirs'
]


if sys.version_info[:2] < (3, 5):
    import pathlib2

    def makedirs(name, mode=0o777, exist_ok=False):
        pathlib2.Path(name).mkdir(mode=mode, parents=True, exist_ok=exist_ok)
else:
    makedirs = os.makedirs
