from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Callable

import distrax
import haiku as hk
import jax
import jax.numpy as jnp


Array = jnp.ndarray
PRNGKey = Array
BijectorParams = Any


class ActNormParamMaker(hk.Module):
    """Create a haiku module for the act norm scale and shift parameters.
    These are initialised to 0 to give the identity transformation."""
    def __init__(self, event_shape: Sequence[int], name=None):
        super().__init__(name=name)
        self.event_shape = event_shape

    def __call__(self) -> Tuple[Array, Array]:
        shift = hk.get_parameter("shift", shape=self.event_shape, init=jnp.zeros)
        log_scale = hk.get_parameter("log_scale", shape=self.event_shape, init=jnp.zeros)
        return shift, log_scale


class ActNormBijector(distrax.Bijector):
    def __init__(self, event_shape: Sequence[int]):
        event_n_dims = len(event_shape)
        assert event_n_dims == 1  # Currently we only seek to work with this case.
        super(ActNormBijector, self).__init__(event_ndims_in=event_n_dims)
        self._bijector_param_maker = ActNormParamMaker(event_shape)

    @property
    def bijector_param_maker(self) -> Callable[[Array], BijectorParams]:
        return self._bijector_param_maker

    @property
    def bijector(self) -> distrax.Bijector:
        shift, log_scale = self._bijector_param_maker()
        bijector = distrax.ScalarAffine(shift=shift, log_scale=log_scale)
        bijector = distrax.Block(bijector, 1)  # final dimension is event dimension
        return bijector

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        y, log_det = self.bijector.forward_and_log_det(x)
        return y, log_det

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        x, log_det = self.bijector.inverse_and_log_det(y)
        return x, log_det


