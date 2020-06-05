import functools

__all__ = []


def validate_weight_norm_arg(weight_norm, axis, use_scale):
    """
    Validate the specified `weight_norm` argument.

    Args:
        weight_norm (bool or (tf.Tensor) -> tf.Tensor)):
            If :obj:`True`, wraps :func:`~tfsnippet.layers.weight_norm`
            with `axis` and `use_scale` argument.  If a callable function,
            it will be returned directly.
        axis (int): The axis argument for `weight_norm`.
        use_scale (bool): The `use_scale` argument for `weight_norm`.

    Returns:
        None or (tf.Tensor) -> tf.Tensor: The weight normalization function,
            or None if weight normalization is not enabled.
    """
    from .normalization import weight_norm as weight_norm_fn
    if callable(weight_norm):
        return weight_norm
    elif weight_norm is True:
        return functools.partial(weight_norm_fn, axis=axis, use_scale=use_scale)
    elif weight_norm in (False, None):
        return None
    else:
        raise TypeError('Invalid value for argument `weight_norm`: expected '
                        'a bool or a callable function, got {!r}.'.
                        format(weight_norm))
