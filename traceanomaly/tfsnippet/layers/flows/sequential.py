from tfsnippet.utils import add_name_and_scope_arg_doc
from .base import BaseFlow, MultiLayerFlow

__all__ = ['SequentialFlow']


class SequentialFlow(MultiLayerFlow):
    """
    Compose a large flow from a sequential of :class:`BaseFlow`.
    """

    @add_name_and_scope_arg_doc
    def __init__(self, flows, name=None, scope=None):
        """
        Construct a new :class:`SequentialFlow`.

        Args:
            flows (Iterable[BaseFlow]): The flow list.
        """
        flows = tuple(flows)  # type: tuple[BaseFlow]
        if not flows:
            raise TypeError('`flows` must not be empty.')

        for i, flow in enumerate(flows):
            if not isinstance(flow, BaseFlow):
                raise TypeError('The {}-th flow in `flows` is not an instance '
                                'of `BaseFlow`: {!r}'.format(i, flow))

        for i, (flow1, flow2) in enumerate(zip(flows[:-1], flows[1:])):
            if flow2.x_value_ndims != flow1.y_value_ndims:
                raise TypeError(
                    '`x_value_ndims` of the {}-th flow != `y_value_ndims` '
                    'of the {}-th flow: {} vs {}.'.
                    format(i + 1, i, flow2.x_value_ndims, flow1.y_value_ndims)
                )

        super(SequentialFlow, self).__init__(
            n_layers=len(flows), x_value_ndims=flows[0].x_value_ndims,
            y_value_ndims=flows[-1].y_value_ndims, name=name, scope=scope
        )
        self._flows = flows
        self._explicitly_invertible = \
            all(flow.explicitly_invertible for flow in flows)

    def _build(self, input=None):
        # do nothing, the building procedure of every flows are automatically
        # called within their `apply` methods.
        pass

    @property
    def flows(self):
        """
        Get the immutable flow list.

        Returns:
            tuple[BaseFlow]: The immutable flow list.
        """
        return self._flows

    @property
    def explicitly_invertible(self):
        return self._explicitly_invertible

    def _transform_layer(self, layer_id, x, compute_y, compute_log_det):
        flow = self._flows[layer_id]
        return flow.transform(x, compute_y, compute_log_det)

    def _inverse_transform_layer(self, layer_id, y, compute_x, compute_log_det):
        flow = self._flows[layer_id]
        return flow.inverse_transform(y, compute_x, compute_log_det)
