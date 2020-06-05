import six

from tfsnippet.utils import is_integer

__all__ = [
    'validate_strides_or_kernel_size',
    'cached',
    'print_with_title',
]


def validate_strides_or_kernel_size(arg_name, arg_value):
    """
    Validate the `strides` or `filter` arg, to ensure it is a tuple of
    two integers.

    Args:
        arg_name (str): The name of the argument, for formatting error.
        arg_value: The value of the argument.

    Returns:
        (int, int): The validated argument.
    """

    if not is_integer(arg_value) and (not isinstance(arg_value, tuple) or
                                      len(arg_value) != 2 or
                                      not is_integer(arg_value[0]) or
                                      not is_integer(arg_value[1])):
        raise TypeError('`{}` must be a int or a tuple (int, int).'.
                        format(arg_name))
    if not isinstance(arg_value, tuple):
        arg_value = (arg_value, arg_value)
    arg_value = tuple(int(v) for v in arg_value)
    return arg_value


def cached(method):
    """
    Decorate `method`, to cache its result.

    Args:
        method: The method whose result should be cached.

    Returns:
        The decorated method.
    """
    results = {}

    @six.wraps(method)
    def wrapper(*args, **kwargs):
        cache_key = (args, tuple((k, kwargs[k]) for k, v in sorted(kwargs)))
        if cache_key not in results:
            results[cache_key] = method(*args, **kwargs)
        return results[cache_key]

    return wrapper


def print_with_title(title, content, before='', after='', hl='='):
    """
    Print a content section with title.

    Args:
        title (str): The title of the section.
        content (str): The multi-line content.
        before (str): String to print before the title.
        after (str): String to print after the content.
        hl (str): The character for horizon line.
    """
    cont_maxlen = max(len(s) for s in content.split('\n'))
    hl_len = max(cont_maxlen, len(title))
    print('{}{}\n{}\n{}{}'.format(before, title, hl * hl_len, content, after))
