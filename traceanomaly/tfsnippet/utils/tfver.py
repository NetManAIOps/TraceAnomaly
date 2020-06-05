import semver

import tensorflow as tf

__all__ = ['is_tensorflow_version_higher_or_equal']


def is_tensorflow_version_higher_or_equal(version):
    """
    Check whether the version of TensorFlow is higher than or equal to
    `version`.

    Args:
        version (str): Expected version of TensorFlow.

    Returns:
        bool: True if higher or equal to, False if not.
    """
    try:
        compare_result = semver.compare_loose(version, tf.__version__)
    except AttributeError:
        compare_result = semver.compare(version, tf.__version__)
    return compare_result <= 0
