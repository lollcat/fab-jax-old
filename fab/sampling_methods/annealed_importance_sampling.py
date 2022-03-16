import chex
import jax.numpy as jnp
import jax
from functools import partial
from typing import Dict, Optional

from fab.types_ import TargetLogProbFunc, HaikuDistribution
from fab.sampling_methods.mcmc.hamiltonean_monte_carlo import HamiltoneanMonteCarlo
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights


class AnnealedImportanceSampler:
    """Annealed importance sampling, designed for use in the fab agent."""
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: TargetLogProbFunc,
                 n_intermediate_distributions: int = 1,
                 transition_operator_type: str = "HMC",
                 additional_transition_operator_kwargs: Dict = {},
                 distribution_spacing_type: str = "linear"):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        if transition_operator_type == "HMC":
            dim = learnt_distribution.dim
            self.transition_operator_manager = HamiltoneanMonteCarlo(
                dim, n_intermediate_distributions, self.intermediate_unnormalised_log_prob,
                **additional_transition_operator_kwargs)
        else:
            raise NotImplementedError
        self.n_intermediate_distributions = n_intermediate_distributions
        self.distribution_spacing_type = distribution_spacing_type
        self.beta_space = self.setup_n_distributions()


    def run(self, batch_size, key, learnt_distribution_params, transition_operator_state):
        """Run annealed importance sampling procedure. Note that we jit everything inside the
        agent so this function will be slow if used elsewhere without jitting."""
        key, subkey = jax.random.split(key, 2)
        log_w = jnp.zeros(batch_size)  # log importance weight
        x_base, log_prob_p0 = self.learnt_distribution.sample_and_log_prob.apply(
            learnt_distribution_params, rng=subkey, sample_shape=(batch_size,))
        x = x_base
        log_w = log_w + self.intermediate_unnormalised_log_prob(learnt_distribution_params, 
                                                                x, 1) - log_prob_p0
        j_s = jnp.arange(1, self.n_intermediate_distributions+1)
        keys = jax.random.split(key, self.n_intermediate_distributions)
        xs = (keys, j_s)
        inner_loop_func = partial(self.inner_loop_func, learnt_distribution_params)
        (x, log_w, transition_operator_state), aux_info = \
            jax.lax.scan(inner_loop_func, init=(x, log_w, transition_operator_state), xs=xs)

        aux_info = self.manage_transition_operator_info(aux_info)
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
        log_w = log_w + \
                self.intermediate_unnormalised_log_prob(learnt_distribution_params, x, j + 1) - \
                self.intermediate_unnormalised_log_prob(learnt_distribution_params, x, j)
        return x, log_w, transition_operator_state, aux_transition_info


    def intermediate_unnormalised_log_prob(self, learnt_distribution_params, x, j):
        """Calculate the intermediate log prob function.
        j is the step of the algorithm, and corresponds which intermediate distribution that
        # we are sampling from. j = 0 is the base (flow) distribution,
        j=N is the target distribution. """
        beta = self.beta_space[j]
        return (1-beta) * self.learnt_distribution.log_prob.apply(learnt_distribution_params, x) + beta * \
               self.target_log_prob(x)


    def setup_n_distributions(self) -> chex.Array:
        """Setup beta_space, which determines how we interpolate between the base and target
        distribution."""
        assert self.n_intermediate_distributions > 0
        if self.n_intermediate_distributions < 3:
            print(f"using linear spacing as there is {self.n_intermediate_distributions}"
                  f"intermediate distribution")
            self.distribution_spacing_type = "linear"
        if self.distribution_spacing_type == "geometric":
            # rough heuristic, copying ratio used in example in AIS paper
            n_linspace_points = max(int(self.n_intermediate_distributions / 5), 2)
            n_geomspace_points = self.n_intermediate_distributions - n_linspace_points
            beta_space = jnp.concatenate([jnp.linspace(0, 0.1, n_linspace_points + 1)[:-1],
                                   jnp.geomspace(0.1, 1, n_geomspace_points)])
        elif self.distribution_spacing_type == "linear":
            beta_space = jnp.linspace(0.0, 1.0, self.n_intermediate_distributions+2)
        else:
            raise Exception(f"distribution spacing incorrectly specified:"
                            f" '{self.distribution_spacing_type}',"
                            f"options are 'geometric' or 'linear'")
        return beta_space


    def get_info(self, log_w_ais: jnp.ndarray, log_w_base: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Get info for logging."""
        info = {}
        ess_base = effective_sample_size_from_unnormalised_log_weights(log_w_base)
        ess_ais = effective_sample_size_from_unnormalised_log_weights(log_w_ais)
        info.update(ess_ais=ess_ais,
                    ess_base=ess_base)
        return info


    def manage_transition_operator_info(self, info: Dict) -> Dict[str, jnp.ndarray]:
        """The transition operator info has an entry for each distribution, here we split this
        info up so we only have one entry per key in the dict."""
        new_info = {}
        for key, val in info.items():
            dist_indxs = list(range(val.shape[0]))
            new_info.update({key + f"_dist{i}": sub_val for i, sub_val in
                              zip(dist_indxs, val)})
        return new_info
