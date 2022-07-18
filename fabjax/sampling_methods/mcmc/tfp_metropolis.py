import chex
from typing import Tuple, NamedTuple

import jax.random
import jax
import jax.numpy as jnp

from fabjax.sampling_methods.mcmc.base import TransitionOperator
from fabjax.types import LogProbFunc

import tensorflow_probability.substrates.jax as tfp


class MetropolisState(NamedTuple):
    step_size: chex.Array


class MetropolisTFP(TransitionOperator):
    def __init__(self,
                 n_intermediate_distributions: int,
                 init_step_size: float = 1.0,
                 n_inner_steps: int = 5,
                 tune: bool = False,
                 target_accept_prob: float = 0.65,
                 adaption_rate: float = 0.05,
                 min_step_size: float = 0.01,
                 ):
        """Simple metropolis mcmc. n_inner_steps is the number of times we run the mcmc
        transition kernel."""
        self.n_intermediate_distributions = n_intermediate_distributions
        self.init_step_size = jnp.ones(n_intermediate_distributions) * init_step_size
        self.tune = tune
        self.target_accept_prob = target_accept_prob
        self.adaption_rate = adaption_rate
        self.min_step_size = min_step_size
        self.n_inner_steps = n_inner_steps
        if tune:
            raise NotImplementedError


    def get_init_state(self) -> chex.ArrayTree:
        return MetropolisState(self.init_step_size)

    def run(self,
            key: chex.PRNGKey,
            transition_operator_state: MetropolisState,
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[chex.Array, chex.ArrayTree, chex.ArrayTree]:
        """Currently does not support step size adjustment."""
        step_size = transition_operator_state.step_size[i]
        transition_kernel = tfp.mcmc.RandomWalkMetropolis(
            transition_target_log_prob, new_state_fn=tfp.mcmc.random_walk_normal_fn(step_size))

        x_new = x
        for _ in range(self.n_inner_steps):
            key, subkey = jax.random.split(key)
            bootstrap_results = transition_kernel.bootstrap_results(x_new)
            x_new, result = transition_kernel.one_step(x_new, bootstrap_results, subkey)
        info = {}
        return x_new, transition_operator_state, info


if __name__ == '__main__':
    import time
    key = jax.random.PRNGKey(0)
    target = lambda x: jnp.mean(- x**2, axis=-1)
    x = jnp.ones((4, 5))
    i = jnp.array(1)
    metropolis = MetropolisTFP(n_intermediate_distributions=2, tune=False)
    transition_operator_state = metropolis.get_init_state()
    # run = jax.jit(metropolis.run, static_argnums=4)
    run = metropolis.run

    for _ in range(5):
        start_time = time.time()
        x_new, transition_operator_state, _ = run(key, transition_operator_state, x, i, target)
        print(time.time() - start_time)
        print(transition_operator_state)
