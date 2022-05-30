# see https://github.com/deepmind/distrax/blob/master/examples/flow.py

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import jax.nn

import distrax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from fab.types import XPoints, LogProbs, HaikuDistribution
from fab.utils.networks import LayerNormMLP
from fab.learnt_distributions.act_norm import ActNormBijector



PRNGKey = chex.PRNGKey




def make_realnvp_dist_funcs(
        x_ndim: int,
        flow_num_layers: int = 8,
        mlp_hidden_size_per_x_dim: int = 2,
        mlp_num_layers: int = 2,
        use_exp: bool = True,
        layer_norm: bool = False,
        act_norm: bool = True,
        lu_layer: bool = True,
):

        event_shape = (x_ndim,)  # is more general in jax example but here assume x is vector
        n_hidden_units = np.prod(event_shape) * mlp_hidden_size_per_x_dim

        get_model = lambda: make_flow_model(
                event_shape=event_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[n_hidden_units] * mlp_num_layers,
                use_exp=use_exp,
                layer_norm=layer_norm,
                act_norm=act_norm,
                lu_layer=lu_layer
        )

        @hk.without_apply_rng
        @hk.transform
        def log_prob(data: XPoints) -> LogProbs:
            model = get_model()
            return model.log_prob(data)

        @hk.without_apply_rng
        @hk.transform
        def base_z_log_prob_and_log_det(data: XPoints) -> Tuple[chex.Array, LogProbs, LogProbs]:
            model = get_model()
            z, log_det = model.bijector.inverse_and_log_det(data)
            log_prob_base = model.distribution.log_prob(z)
            return z, log_prob_base, log_det

        @hk.without_apply_rng
        @hk.transform
        def log_det_forward(z: chex.Array) -> Tuple[XPoints, chex.Array]:
            model = get_model()
            x, log_det = model.bijector.forward_and_log_det(z)
            return x, log_det


        @hk.transform
        def sample_and_log_prob(sample_shape: Tuple = ()) \
                -> Tuple[XPoints, LogProbs]:
            model = get_model()
            return model.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


        @hk.transform
        def sample(sample_shape: Tuple = ()) -> XPoints:
            model = get_model()
            return model.sample(seed=hk.next_rng_key(), sample_shape=sample_shape)

        return HaikuDistribution(x_ndim, log_prob, sample_and_log_prob, sample,
                                 base_z_log_prob_and_log_det, log_det_forward)




def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int = 2,
                     layer_norm: bool = False) -> hk.Sequential:
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
    return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      mlp(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
    ])

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

def make_gaussian_base_dist(event_shape: Sequence[int], dtype: jax.lax.Precision):
    loc = hk.get_parameter("loc", shape=event_shape, init=jnp.zeros, dtype=dtype)
    log_scale = hk.get_parameter("log_scale", shape=event_shape, init=jnp.zeros, dtype=dtype)
    scale = jnp.exp(log_scale)
    base_dist = distrax.Independent(
        distrax.Normal(
            loc=loc,
            scale=scale),
        reinterpreted_batch_ndims=len(event_shape))
    return base_dist


def make_flow_model(event_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int],
                    use_exp: bool,
                    layer_norm: bool,
                    act_norm: bool,
                    lu_layer: bool) -> distrax.Transformed:
    """Creates the flow model."""
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    event_ndims = len(event_shape)
    assert event_ndims == 1  # currently only focusing on this case (all elements in 1 dim).
    dim = event_shape[0]
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

        layer = distrax.SplitCoupling(
            split_index=split_index,
            bijector=bijector_fn,
            conditioner=make_conditioner(event_shape=(num_bijector_params,),
                                         hidden_sizes=hidden_sizes,
                                         layer_norm=layer_norm),
            event_ndims=event_ndims,
            swap=flip)
        layers.append(layer)
        if act_norm:
            act_norm_layer = ActNormBijector(event_shape=event_shape, dtype=dtype)
            layers.append(act_norm_layer)

        if lu_layer:
            matrix = hk.get_parameter("matrix_LU_layer", shape=(dim, dim), init=jnp.zeros, dtype=dtype) + \
                     jnp.eye(dim)
            bias = hk.get_parameter("bias_LU_layer", shape=(dim,), init=jnp.zeros, dtype=dtype)
            lu_layer = distrax.LowerUpperTriangularAffine(
                matrix=matrix, bias=bias)
            layers.append(lu_layer)

    flow = distrax.Chain(layers)
    base_distribution = make_gaussian_base_dist(event_shape, dtype)
    return distrax.Transformed(base_distribution, flow)

