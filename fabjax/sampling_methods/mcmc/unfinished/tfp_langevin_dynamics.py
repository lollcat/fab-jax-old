import chex
from typing import Tuple, NamedTuple

import jax.random
import jax
import jax.numpy as jnp

from fabjax.sampling_methods.mcmc.base import TransitionOperator
from fabjax.types import LogProbFunc

import tensorflow_probability.substrates.jax as tfp


class LDState(NamedTuple):
    step_size: chex.Array


class LevanginDynamicsTFP(TransitionOperator):
    def __init__(self,
                 n_intermediate_distributions: int,
                 max_tree_depth: int = 10,
                 init_step_size: float = 1.0,
                 tune: bool = True,
                 target_accept_prob: float = 0.65,
                 adaption_rate: float = 0.05,
                 ):
        self.max_tree_depth = max_tree_depth
        self.n_intermediate_distributions = n_intermediate_distributions
        self.init_step_size = jnp.ones(n_intermediate_distributions) * init_step_size
        self.tune = tune
        self.target_accept_prob = target_accept_prob
        self.adaption_rate = adaption_rate


    def get_init_state(self) -> LDState:
        return LDState(self.init_step_size)

    def run(self,
            key: chex.PRNGKey,
            transition_operator_state: LDState,
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[chex.Array, chex.ArrayTree, chex.ArrayTree]:
        info = {}
        step_size = transition_operator_state.step_size[i]
        transition_kernel = tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn=transition_target_log_prob,
            step_size=step_size,
        ))
        if self.tune:
            transition_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                transition_kernel,
                num_adaptation_steps=1,
                target_accept_prob=self.target_accept_prob,
                adaptation_rate=self.adaption_rate
            )
        # there is an error here where the step size adaption is unable to grab the step size.
        bootstrap_results = transition_kernel.bootstrap_results(x)
        x_new, result = transition_kernel.one_step(x, bootstrap_results, key)
        if self.tune:
            step_size = transition_operator_state.step_size.at[i].set(result.new_step_size)
            transition_operator_state = LDState(step_size=step_size)
            info.update({f"step_{i}": step_size[i] for
                         i in range(self.n_intermediate_distributions)})
        return x_new, transition_operator_state, info


if __name__ == '__main__':
    import time
    key = jax.random.PRNGKey(0)

    target = lambda x: jnp.mean(- x**2, axis=-1)
    x = jnp.ones((4, 5))
    i = jnp.array(1)
    nuts = LevanginDynamicsTFP(n_intermediate_distributions=2, tune=True)
    transition_operator_state = nuts.get_init_state()
    run = jax.jit(nuts.run, static_argnums=4)
    # run = nuts.run

    for _ in range(5):
        start_time = time.time()
        x_new, transition_operator_state, _ = run(key, transition_operator_state, x, i, target)
        print(time.time() - start_time)
        print(transition_operator_state)
