from .base import BaseFlow

__all__ = ['InvertFlow']


class InvertFlow(BaseFlow):
    """
    Turn a :class:`BaseFlow` into its inverted flow.

    This class is particularly useful when the flow is (theoretically) defined
    in the opposite direction to the direction of network initialization.
    For example, define `z -> x`, but initialized by feeding `x`.
    """

    def __init__(self, flow, name=None, scope=None):
        """
        Construct a new :class:`InvertFlow`.

        Args:
            flow (BaseFlow): The underlying flow.
        """
        if not isinstance(flow, BaseFlow) or not flow.explicitly_invertible:
            raise ValueError('`flow` must be an explicitly invertible flow: '
                             'got {!r}'.format(flow))
        self._flow = flow

        super(InvertFlow, self).__init__(
            x_value_ndims=flow.y_value_ndims,
            y_value_ndims=flow.x_value_ndims,
            require_batch_dims=flow.require_batch_dims,
            name=name,
            scope=scope
        )

    def invert(self):
        """
        Get the original flow, inverted by this :class:`InvertFlow`.

        Returns:
            BaseFlow: The original flow.
        """
        return self._flow

    @property
    def explicitly_invertible(self):
        return True

    def build(self, input=None):  # pragma: no cover
        # since `flow` should be inverted, we should build `flow` in
        # `inverse_transform` rather than in `transform` or `build`
        pass

    def transform(self, x, compute_y=True, compute_log_det=True, name=None):
        return self._flow.inverse_transform(
            y=x, compute_x=compute_y, compute_log_det=compute_log_det,
            name=name
        )

    def inverse_transform(self, y, compute_x=True, compute_log_det=True,
                          name=None):
        return self._flow.transform(
            x=y, compute_y=compute_x, compute_log_det=compute_log_det,
            name=name
        )

    def _build(self, input=None):
        raise RuntimeError('Should never be called.')  # pragma: no cover

    def _transform(self, x, compute_y, compute_log_det):
        raise RuntimeError('Should never be called.')  # pragma: no cover

    def _inverse_transform(self, y, compute_x, compute_log_det):
        raise RuntimeError('Should never be called.')  # pragma: no cover
