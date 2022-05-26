import chex
import jax.numpy as jnp
import jax
from functools import partial
from typing import Dict, Callable

from fab.sampling_methods.mcmc.base import TransitionOperator
from fab.sampling_methods.base import AnnealedImportanceSamplerBase
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights
from fab.types import LogProbFunc


IntermediateLogProb = Callable[[chex.Array, int], chex.Array]

class AnnealedImportanceSampler(AnnealedImportanceSamplerBase):
    """Annealed importance sampling, designed for use in the fab agent."""
    def __init__(self,
                 dim: int,
                 n_intermediate_distributions: int = 1,
                 transition_operator_type: str = "hmc_tfp",
                 additional_transition_operator_kwargs: Dict = {},
                 distribution_spacing_type: str = "linear"
                 ):
        self.transition_operator_manager: TransitionOperator
        if transition_operator_type == "hmc":
            from fab.sampling_methods.mcmc.hamiltonean_monte_carlo import HamiltoneanMonteCarlo
            self.transition_operator_manager = HamiltoneanMonteCarlo(
                dim, n_intermediate_distributions,
                **additional_transition_operator_kwargs)
        elif transition_operator_type == "hmc_tfp":
            from fab.sampling_methods.mcmc.tfp_hamiltonean_monte_carlo import HamiltoneanMonteCarloTFP
            self.transition_operator_manager = HamiltoneanMonteCarloTFP(
                n_intermediate_distributions,
                **additional_transition_operator_kwargs)
        elif transition_operator_type == "nuts_tfp":
            from fab.sampling_methods.mcmc.tfp_nuts import NoUTurnSamplerTFP
            self.transition_operator_manager = NoUTurnSamplerTFP(
                n_intermediate_distributions,
                **additional_transition_operator_kwargs)
        else:
            raise NotImplementedError
        self.n_intermediate_distributions = n_intermediate_distributions
        self.distribution_spacing_type = distribution_spacing_type
        self.beta_space = self.setup_n_distributions()


    def get_init_state(self):
        return self.transition_operator_manager.get_init_state()

    def run(self, x_base, log_prob_p0, key, transition_operator_state,
            base_log_prob, target_log_prob):
        """Run annealed importance sampling procedure. Note that we jit everything inside the
        agent so this function will be slow if used elsewhere without jitting."""
        intermediate_log_prob: IntermediateLogProb = partial(self.intermediate_unnormalised_log_prob,
                                      base_log_prob=base_log_prob, target_log_prob=target_log_prob)
        key, subkey = jax.random.split(key, 2)
        log_w = jnp.zeros(x_base.shape[0])  # log importance weight
        x = x_base
        log_w = log_w + intermediate_log_prob(x, 1) - log_prob_p0
        j_s = jnp.arange(1, self.n_intermediate_distributions+1)
        keys = jax.random.split(subkey, self.n_intermediate_distributions)
        xs = (keys, j_s)
        inner_loop_func = partial(self.inner_loop_func,
                                  intermediate_log_prob=intermediate_log_prob)
        (x, log_w, transition_operator_state), aux_info = \
            jax.lax.scan(inner_loop_func, init=(x, log_w, transition_operator_state), xs=xs)
        aux_info = self.manage_transition_operator_info(aux_info)
        # get more info, such as effective sample size
        log_w_base = target_log_prob(x_base) - log_prob_p0
        aux_info.update(self.get_info(log_w_ais=log_w, log_w_base=log_w_base))
        return x, log_w, transition_operator_state, aux_info

    def inner_loop_func(self, carry, xs, intermediate_log_prob: IntermediateLogProb):
        x_new, log_w, transition_operator_state = carry
        key, j = xs
        x_new, log_w, transition_operator_state, aux_transition_info = \
            self.perform_transition(key,
                                   transition_operator_state,
                                   x_new,
                                   log_w,
                                   j,
                                   intermediate_log_prob)
        return (x_new, log_w, transition_operator_state), aux_transition_info


    def perform_transition(self, key, transition_operator_state, x, log_w, j,
                           intermediate_log_prob: IntermediateLogProb):
        transition_target_log_prob: LogProbFunc = partial(intermediate_log_prob, j=j)
        x, transition_operator_state, aux_transition_info = \
            self.transition_operator_manager.run(key, transition_operator_state, x,
                                                 j - 1, transition_target_log_prob)
        log_w = log_w + intermediate_log_prob(x, j + 1) - intermediate_log_prob(x, j)
        return x, log_w, transition_operator_state, aux_transition_info


    def intermediate_unnormalised_log_prob(self, x, j, base_log_prob, target_log_prob) -> \
            chex.Array:
        """Calculate the intermediate log prob function.
        j is the step of the algorithm, and corresponds which intermediate distribution that
        # we are sampling from. j = 0 is the base (flow) distribution,
        j=N is the target distribution. """
        beta = self.beta_space[j]
        return (1-beta) * base_log_prob(x) + beta * target_log_prob(x)


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
            # rough heuristic, copying ratio used in example in AIS paper
            n_linspace_points = max(int(self.n_intermediate_distributions / 4), 2)
            n_geomspace_points = self.n_intermediate_distributions - n_linspace_points
            beta_space = jnp.concatenate([jnp.linspace(0, 0.01, n_linspace_points + 3)[:-1],
                                      jnp.geomspace(0.01, 1, n_geomspace_points)])
            beta_space = jnp.flip(1 - beta_space)
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
