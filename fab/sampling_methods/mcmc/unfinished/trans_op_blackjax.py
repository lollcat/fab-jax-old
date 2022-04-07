"""Doesn't allow use to grab the step size. """


from typing import Optional, NamedTuple, Tuple

import blackjax
import blackjax.mcmc.integrators as integrators
import jax.numpy as jnp
import jax
import chex

from fab.sampling_methods.mcmc.base import TransitionOperator
from fab.types import LogProbFunc


class BlackJaxHMC(TransitionOperator):
    def __init__(self,
                 dim: int,
                 init_target_log_prob: LogProbFunc,
                 init_seed: int = 0,
                 init_run_length: int = 1000):
        self.init_target_log_pob = init_target_log_prob
        self.dim = dim
        self.num_integration_steps = 5
        self.kernel = blackjax.hmc.kernel(integrators.mclachlan)
        self.init_key = jax.random.PRNGKey(init_seed)
        self.init_run_length = init_run_length
        self.initial_position = jnp.zeros((dim,))

    def get_init_state(self) -> chex.ArrayTree:
        warmup = blackjax.window_adaptation(
            blackjax.hmc,
            self.init_target_log_pob,
            self.init_run_length,
            num_integration_steps=5
        )
        state, kernel, _ = warmup.run(
            self.init_key,
            self.initial_position,
        )
        self.kernel = kernel
        return jnp.array(0)  # no transition operator state

    def run(self,
            key: chex.PRNGKey,
            transition_operator_state: chex.ArrayTree,
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[chex.Array, chex.ArrayTree, chex.ArrayTree]:
        del(i)
        key_batch = jax.random.split(key, x.shape[0])
        state = jax.vmap(blackjax.hmc.init, in_axes=(0, None))(x, transition_target_log_prob)
        state, info = jax.vmap(self.kernel, in_axes=(0, 0, None))(key_batch, state,
                                                                  transition_target_log_prob)
        return state, transition_operator_state, info






if __name__ == '__main__':
    dim = 3
    key = jax.random.PRNGKey(0)
    target = lambda x: jnp.mean(- x**2, axis=-1)
    transition_oeprator = BlackJaxHMC(dim, target)
    transition_operator_state = transition_oeprator.get_init_state()
    x_batch = jax.random.normal(key, shape=(32, dim))
    state, transition_operator_state, info = \
        transition_oeprator.run(key, transition_operator_state, x_batch,
                                jnp.array(0), target)
