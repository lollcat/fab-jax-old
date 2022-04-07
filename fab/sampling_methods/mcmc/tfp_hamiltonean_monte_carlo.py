import chex
from typing import Tuple, NamedTuple

import jax.random
import jax
import jax.numpy as jnp

from fab.sampling_methods.mcmc.base import TransitionOperator
from fab.types import LogProbFunc

import tensorflow_probability.substrates.jax as tfp


class HMCState(NamedTuple):
    step_size: chex.Array


class HamiltoneanMonteCarloTFP(TransitionOperator):
    def __init__(self,
                 n_intermediate_distributions: int,
                 n_leapfrog_steps: int = 5,
                 init_step_size: float = 1.0,
                 tune: bool = True,
                 target_accept_prob: float = 0.75,
                 adaption_rate: float = 0.05,
                 ):
        self.n_leapfrog_steps = n_leapfrog_steps
        self.n_intermediate_distributions = n_intermediate_distributions
        self.init_step_size = jnp.ones(n_intermediate_distributions) * init_step_size
        self.tune = tune
        self.target_accept_prob = target_accept_prob
        self.adaption_rate = adaption_rate


    def get_init_state(self) -> chex.ArrayTree:
        return HMCState(self.init_step_size)

    def run(self,
            key: chex.PRNGKey,
            transition_operator_state: HMCState,
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[chex.Array, chex.ArrayTree, chex.ArrayTree]:
        step_size = transition_operator_state.step_size[i]
        transition_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            transition_target_log_prob, step_size,
            self.n_leapfrog_steps)
        if self.tune:
            transition_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                transition_kernel, num_adaptation_steps=1,
                target_accept_prob=self.target_accept_prob,
                adaptation_rate=self.adaption_rate,
            )
        bootstrap_results = transition_kernel.bootstrap_results(x)
        x_new, result = transition_kernel.one_step(x, bootstrap_results, key)
        if self.tune:
            step_size = transition_operator_state.step_size.at[i].set(result.new_step_size)
            transition_operator_state = HMCState(step_size=step_size)
        return x_new, transition_operator_state, {}


if __name__ == '__main__':
    import time
    key = jax.random.PRNGKey(0)

    target = lambda x: jnp.mean(- x**2, axis=-1)
    x = jnp.ones((4, 5))
    i = jnp.array(1)
    hmc = HamiltoneanMonteCarloTFP(n_intermediate_distributions=2, tune=True)
    transition_operator_state = hmc.get_init_state()
    # run = jax.jit(hmc.run, static_argnums=4)
    run = hmc.run

    for _ in range(5):
        start_time = time.time()
        x_new, transition_operator_state, _ = run(key, transition_operator_state, x, i, target)
        print(time.time() - start_time)
        print(transition_operator_state)
