# see https://github.com/deepmind/distrax/blob/master/examples/flow.py

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union
from fab.types import XPoints, LogProbs, HaikuDistribution

import distrax
import chex
import haiku as hk
import jax.numpy as jnp
import numpy as np


PRNGKey = chex.PRNGKey



def make_rational_quadratic_spline_dist_funcs(
        x_ndim: int, flow_num_layers: int = 8,
                 mlp_hidden_size_per_x_dim: int = 2,  mlp_num_layers: int = 2, spline_num_bins:
            int = 4):
        event_shape = (x_ndim,)  # is more general in jax example but here assume x is vector
        n_hidden_units = np.prod(event_shape) * mlp_hidden_size_per_x_dim

        get_model = lambda: make_flow_model(
                event_shape=event_shape,
                num_layers=flow_num_layers,
                hidden_sizes=[n_hidden_units] * mlp_num_layers,
                num_bins=spline_num_bins)

        # TODO: may want to add nan checks to sample, and sample_and_log_prob (as this seems to
        #  sometimes occur with
        #  flow models)
        @hk.without_apply_rng
        @hk.transform
        def log_prob(data: XPoints) -> LogProbs:
            model = get_model()
            return model.log_prob(data)

        @hk.without_apply_rng
        @hk.transform
        def sample_and_log_prob(seed: PRNGKey, sample_shape: Tuple = ()) \
                -> Tuple[XPoints, LogProbs]:
            model = get_model()
            return model.sample_and_log_prob(seed=seed, sample_shape=sample_shape)


        @hk.without_apply_rng
        @hk.transform
        def sample(seed: PRNGKey, sample_shape: Tuple = ()) -> XPoints:
            model = get_model()
            return model.sample(seed=seed, sample_shape=sample_shape)

        return HaikuDistribution(x_ndim, log_prob, sample_and_log_prob, sample)





def make_conditioner(event_shape: Sequence[int],
                     hidden_sizes: Sequence[int],
                     num_bijector_params: int) -> hk.Sequential:
  """Creates an MLP conditioner for each layer of the flow."""
  return hk.Sequential([
      hk.Flatten(preserve_dims=-len(event_shape)),
      hk.nets.MLP(hidden_sizes, activate_final=True),
      # We initialize this linear layer to zero so that the flow is initialized
      # to the identity function.
      hk.Linear(
          np.prod(event_shape) * num_bijector_params,
          w_init=jnp.zeros,
          b_init=jnp.zeros),
      hk.Reshape(tuple(event_shape) + (num_bijector_params,), preserve_dims=-1),
  ])


def make_flow_model(event_shape: Sequence[int],
                    num_layers: int,
                    hidden_sizes: Sequence[int],
                    num_bins: int) -> distrax.Transformed:
  """Creates the flow model."""
  # Alternating binary mask.
  mask = jnp.arange(0, np.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  def bijector_fn(params: jnp.ndarray):
    return distrax.RationalQuadraticSpline(
        params, range_min=0., range_max=1.)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.
  num_bijector_params = 3 * num_bins + 1

  layers = []
  for _ in range(num_layers):
    layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=make_conditioner(event_shape, hidden_sizes,
                                     num_bijector_params))
    layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # We invert the flow so that the `forward` method is called with `log_prob`.
  flow = distrax.Inverse(distrax.Chain(layers))
  base_distribution = distrax.Independent(
      distrax.Normal(
          loc=jnp.zeros(event_shape),
          scale=jnp.ones(event_shape)),
      reinterpreted_batch_ndims=len(event_shape))

  return distrax.Transformed(base_distribution, flow)

