# see https://github.com/deepmind/distrax/blob/master/examples/flow.py

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union, Callable

import jax.nn

import distrax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from fab.types import LogProbs, HaikuDistribution
from fab.utils.networks import LayerNormMLP



PRNGKey = chex.PRNGKey
ZPoints = chex.Array  # points in latent dimension
XPoints = chex.Array  # points in x dimension (e.g. in image space)
HConditioning = chex.Array  # conditioning provided to



class RealNVP(hk.Module):
    def __init__(self,
                x_ndim: int,
                flow_num_layers: int = 8,
                mlp_hidden_size_per_x_dim: int = 2,
                mlp_num_layers: int = 2,
                use_exp: bool = True,
                layer_norm: bool = False):
        """Here we create a flow defined over q(z | x). We create a representation $h$ which is a
        function of x, which gets passed to each layer of the flow"""
        super(RealNVP, self).__init__()
        event_shape = (x_ndim,)  # is more general in jax example but here assume x is vector
        n_hidden_units = np.prod(event_shape) * mlp_hidden_size_per_x_dim
        self.make_flow_transform = lambda h: make_flow_transform(
            h=h, event_shape=event_shape, hidden_sizes=[n_hidden_units] * mlp_num_layers,
            use_exp=use_exp, num_layers=flow_num_layers, layer_norm=layer_norm
        )

    def __call__(self, h: HConditioning) -> distrax.Bijector:
        return self.make_flow_transform(h)







def make_conditioner(
        h: chex.Array,
        event_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        num_bijector_params: int = 2,
        layer_norm: bool = False) -> Callable[[chex.Array], chex.Array]:
    """
    Creates an MLP conditioner for each layer of the flow.
    Args:
        event_shape: Input dimension to the conditioner
        hidden_sizes: MLP hidden layer spec
        num_bijector_params: Number of bijector params (typically 2, one for scale, one for shift)
        layer_norm: Whether to use a layer norm MLP instead of a vanilla MLP

    Returns: Haiku moduler used for getting the scale and shift conditioned on half of the
        RealNVP coupling.

    """
    mlp = LayerNormMLP if layer_norm else hk.nets.MLP
    def conditioner(x: chex.Array) -> chex.Array:
        if len(x.shape) == len(h.shape):
            h_ = h
        else:
            assert len(h.shape) == (len(x.shape) - 1)
            # only x batched
            h_ = jnp.repeat(h[None, ...], x.shape[0], axis=0)
        input = jnp.concatenate((x, h_), axis=-1)
        output = hk.Sequential([
            hk.Flatten(preserve_dims=-len(event_shape)),
            mlp(hidden_sizes, activate_final=True),
            # We initialize this linear layer to zero so that the flow is initialized
            # to the identity function.
            hk.Linear(
                np.prod(event_shape) * num_bijector_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros),
            hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
        ])(input)
        return output
    return conditioner


def make_scalar_affine_bijector(use_exp: bool = True):
    def bijector_fn(params: jnp.ndarray):
      if use_exp:
          shift, log_scale = jnp.split(params, indices_or_sections=2, axis=-1)
          shift = jnp.squeeze(shift, axis=-1)
          log_scale = jnp.squeeze(log_scale, axis=-1)
          return distrax.ScalarAffine(shift=shift, log_scale=log_scale)
      else:
          shift, pre_activate_scale = jnp.split(params, indices_or_sections=2, axis=-1)
          shift = jnp.squeeze(shift, axis=-1)
          # for identity init
          pre_activate_scale = tfp.math.softplus_inverse(1.0) + pre_activate_scale
          scale = jnp.squeeze(jax.nn.softplus(pre_activate_scale), axis=-1)
          return distrax.ScalarAffine(shift=shift, scale=scale)
    return bijector_fn



def make_flow_transform(
        h: chex.Array,
        event_shape: Sequence[int],
        num_layers: int,
        hidden_sizes: Sequence[int],
        use_exp: bool,
        layer_norm: bool) -> distrax.Bijector:
    """Creates the flow model."""
    event_ndims = len(event_shape)
    assert event_ndims == 1  # currently only focusing on this case (all elements in 1 dim).
    layers = []
    n_params = np.prod(event_shape)
    split_index = n_params // 2
    bijector_fn = make_scalar_affine_bijector(use_exp)
    for i in range(num_layers):
        flip = i % 2 == 0
        if flip:
            num_bijector_params = n_params // 2
            num_dependent_params = n_params - num_bijector_params
        else:
            num_dependent_params = n_params // 2
            num_bijector_params = n_params - num_dependent_params

        conditioner = make_conditioner(
                h=h, event_shape=(num_bijector_params,),
                                         hidden_sizes=hidden_sizes,
                                         layer_norm=layer_norm)
        layer = distrax.SplitCoupling(
            split_index=split_index,
            bijector=bijector_fn,
            conditioner=conditioner,
            event_ndims=event_ndims,
            swap=flip)
        layers.append(layer)

    flow = distrax.Chain(layers)
    return flow

