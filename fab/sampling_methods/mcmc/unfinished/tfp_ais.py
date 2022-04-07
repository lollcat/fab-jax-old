from typing import NamedTuple

import chex
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp

from fab.sampling_methods.base import AnnealedImportanceSamplerBase

class TransitionOperatorState(NamedTuple):
    step_size: chex.Array


class AnnealedImportanceSamplerTfp(AnnealedImportanceSamplerBase):
    def __init__(self,
                 dim: int,
                 n_intermediate_distributions: int = 1,
                 *args,
                 **kwargs,
                 ):
        self.dim = dim
        self.n_intermediate_distribution = n_intermediate_distributions
        self.tune = True


    def get_init_state(self):
        return TransitionOperatorState(step_size=jnp.array(1.0))

    def run(self, x_base, log_prob_p0, key, transition_operator_state,
            base_log_prob, target_log_prob):
        def make_kernel_fn(target_log_prob):
            step_size = transition_operator_state.step_size
            transition_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=step_size,
                num_leapfrog_steps=2)
            if self.tune:
                transition_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                    transition_kernel, num_adaptation_steps=1,
                    target_accept_prob=0.75,
                    adaptation_rate=0.01,
                )
            return transition_kernel

        x_ais, log_w_ais, kernels_results = (
            tfp.mcmc.sample_annealed_importance_chain(
                num_steps=self.n_intermediate_distribution,
                proposal_log_prob_fn=base_log_prob,
                target_log_prob_fn=target_log_prob,
                current_state=x_base,
                make_kernel_fn=make_kernel_fn,
                seed=key))
        info = {}
        if self.tune:
            transition_operator_state = TransitionOperatorState(
                kernels_results.inner_results.new_step_size)
            info.update(step_size=transition_operator_state.step_size)
        return x_ais, log_w_ais, transition_operator_state, info







