
def _require_multi_samples(axis, name):
    if axis is None:
        raise ValueError('{} requires multi-samples of latent variables, '
                         'thus the `axis` argument must be specified'.
                         format(name))
