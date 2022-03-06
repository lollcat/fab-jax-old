from typing import Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import chex

from fab.learnt_distributions.act_norm import ActNormBijector

Array = jnp.ndarray

@hk.without_apply_rng
@hk.transform
def forward(x):
    act_norm = ActNormBijector(x[0].shape)
    return act_norm.forward_and_log_det(x)

def make_flow_model(event_shape):
    flow = ActNormBijector(event_shape)
    base_distribution = distrax.MultivariateNormalDiag(jnp.zeros(event_shape), jnp.ones(event_shape))
    return distrax.Transformed(base_distribution, flow)

@hk.transform
def sample_and_log_prob(dim, seed, sample_shape=(1,)) -> Tuple[Array, Array]:
  model = make_flow_model(event_shape=[dim])
  return model.sample_and_log_prob(seed=seed, sample_shape=sample_shape)


def test_act_norm():
    batch_size = 5
    event_size = 8
    x = jnp.ones((batch_size, event_size)) + 0.1*jax.random.normal(jax.random.PRNGKey(42),
                                                                   shape=(batch_size, event_size))
    # test bijector forward function
    params = forward.init(x=x, rng=jax.random.PRNGKey(42))
    x_norm, log_det = forward.apply(params, x)
    print(x_norm.shape, log_det.shape)
    print(jnp.mean(x_norm, axis=0), jnp.std(x_norm, axis=0))


    # test act norm
    data = jax.random.normal(jax.random.PRNGKey(42), shape=(batch_size, event_size))
    params = sample_and_log_prob.init(dim=data.shape[-1], rng=jax.random.PRNGKey(42), seed=jax.random.PRNGKey(42),
                                      sample_shape=(10,))
    x, log_prob_data = sample_and_log_prob.apply(params=params, seed=jax.random.PRNGKey(42), dim=data.shape[-1],
                                                 rng=jax.random.PRNGKey(42), sample_shape=(batch_size,))
    print(x, log_prob_data)
    print(x.shape, log_prob_data.shape)
    chex.assert_shape(x, (batch_size, event_size))
    chex.assert_shape(log_prob_data, (batch_size,))


if __name__ == '__main__':
    test_act_norm()