import jax.numpy as jnp
import jax
from functools import partial
from typing import Dict

from fab.types_ import TargetLogProbFunc, HaikuDistribution
from fab.sampling_methods.mcmc.hamiltonean_monte_carlo import HamiltoneanMonteCarlo
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights


class AnnealedImportanceSampler:
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: TargetLogProbFunc,
                 n_parallel_runs: int,
                 n_intermediate_distributions: int = 1,
                 transition_operator_type="HMC",
                 additional_transition_operator_kwargs={},
                 distribution_spacing_type: str = "linear"):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        if transition_operator_type == "HMC":
            dim = learnt_distribution.dim
            self.transition_operator_manager = HamiltoneanMonteCarlo(
                dim, n_intermediate_distributions, self.intermediate_unnormalised_log_prob,
                n_parallel_runs, **additional_transition_operator_kwargs)
        else:
            raise NotImplementedError
        self.n_parallel_runs = n_parallel_runs
        self.n_intermediate_distributions = n_intermediate_distributions
        self.distribution_spacing_type = distribution_spacing_type
        self.setup_n_distributions()



    # @partial(jax.jit, static_argnums=(0,)) # we instead jit everything together in the agent
    def run(self, key, learnt_distribution_params, transition_operator_state):
        """Run annealed importance sampling procedure."""
        key, subkey = jax.random.split(key, 2)
        log_w = jnp.zeros(self.n_parallel_runs)  # log importance weight
        x_base, log_prob_p0 = self.learnt_distribution.sample_and_log_prob.apply(
            learnt_distribution_params, rng=subkey, sample_shape=(self.n_parallel_runs,))
        x = x_base
        log_w = log_w + self.intermediate_unnormalised_log_prob(learnt_distribution_params, 
                                                                x, 1) - log_prob_p0
        j_s = jnp.arange(1, self.n_intermediate_distributions+1)
        keys = jax.random.split(key, self.n_intermediate_distributions)
        xs = (keys, j_s)
        inner_loop_func = partial(self.inner_loop_func, learnt_distribution_params)
        (x, log_w, transition_operator_state), aux_info = \
            jax.lax.scan(inner_loop_func, init=(x, log_w, transition_operator_state), xs=xs)

        # get more info, such as effective sample size
        log_w_base = self.target_log_prob(x_base) - log_prob_p0
        aux_info.update(self.get_info(log_w_ais=log_w, log_w_base=log_w_base))
        return x, log_w, transition_operator_state, aux_info

    def inner_loop_func(self, learnt_distribution_params, carry, xs):
        x_new, log_w, transition_operator_state = carry
        key, j = xs
        x_new, log_w, transition_operator_state, aux_transition_info = \
            self.perform_transition(key,
                                   learnt_distribution_params,
                                   transition_operator_state,
                                   x_new,
                                   log_w,
                                   j)
        return (x_new, log_w, transition_operator_state), aux_transition_info


    def perform_transition(self, key, learnt_distribution_params,
                           transition_operator_state, x, log_w, j):
        x, transition_operator_state, aux_transition_info = \
            self.transition_operator_manager.run(key, learnt_distribution_params,
                  transition_operator_state, x, j - 1)
        log_w = log_w + self.intermediate_unnormalised_log_prob(learnt_distribution_params, x, j + 1) - \
                self.intermediate_unnormalised_log_prob(learnt_distribution_params, x, j)
        return x, log_w, transition_operator_state, aux_transition_info



    def intermediate_unnormalised_log_prob(self, learnt_distribution_params, x, j):
        # j is the step of the algorithm, and corresponds which intermediate distribution that we are sampling from
        # j = 0 is the sampling distribution, j=N is the target distribution
        beta = self.B_space[j]
        return (1-beta) * self.learnt_distribution.log_prob.apply(learnt_distribution_params, x) + beta * \
               self.target_log_prob(x)


    def setup_n_distributions(self):
        assert self.n_intermediate_distributions > 0
        if self.n_intermediate_distributions < 3:
            print(f"using linear spacing as there is {self.n_intermediate_distributions}"
                  f"intermediate distribution")
            self.distribution_spacing_type = "linear"
        if self.distribution_spacing_type == "geometric":
            # rough heuristic, copying ratio used in example in AIS paper
            n_linspace_points = max(int(self.n_intermediate_distributions / 5), 2)
            n_geomspace_points = self.n_intermediate_distributions - n_linspace_points
            self.B_space = jnp.concatenate([jnp.linspace(0, 0.1, n_linspace_points + 1)[:-1],
                                   jnp.geomspace(0.1, 1, n_geomspace_points)])
        elif self.distribution_spacing_type == "linear":
            self.B_space = jnp.linspace(0.0, 1.0, self.n_intermediate_distributions+2)
        else:
            raise Exception(f"distribution spacing incorrectly specified:"
                            f" '{self.distribution_spacing_type}',"
                            f"options are 'geometric' or 'linear'")

    def get_info(self, log_w_ais: jnp.ndarray, log_w_base: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        info = {}
        ess_base = effective_sample_size_from_unnormalised_log_weights(log_w_base)
        ess_ais = effective_sample_size_from_unnormalised_log_weights(log_w_ais)
        info.update(ess_ais=ess_ais,
                    ess_base=ess_base)
        return info
