from .scheduled_var import ScheduledVariable

__all__ = ['resolve_feed_dict', 'merge_feed_dict']


def resolve_feed_dict(feed_dict, inplace=False):
    """
    Resolve all dynamic values in `feed_dict` into fixed values.

    The supported dynamic value types and corresponding resolving method
    is listed as follows:

    1. :class:`ScheduledVariable`: :meth:`get()` will be called.
    2. callable object: Will be called to get the value.

    Args:
        feed_dict (dict[tf.Tensor, any]): The feed dict to be resolved.
        inplace (bool): Whether or not to fill resolved values in
            the input `feed_dict` directly, instead of copying a new one?
            (default :obj:`False`)

    Returns:
        The resolved feed dict.
    """
    if not inplace:
        feed_dict = dict(feed_dict)
    for k in feed_dict:
        v = feed_dict[k]
        if isinstance(v, ScheduledVariable):
            feed_dict[k] = v.get()
        elif callable(v):
            feed_dict[k] = v()
    return feed_dict


def merge_feed_dict(*feed_dicts):
    """
    Merge all feed dicts into one.

    Args:
        \**feed_dicts: List of feed dicts.  The later ones will override
            values specified in the previous ones.  If a :obj:`None` is
            specified, it will be simply ignored.

    Returns:
        The merged feed dict.
    """
    ret = {}
    for feed_dict in feed_dicts:
        if feed_dict is not None:
            ret.update(feed_dict)
    return ret
