from typing import Tuple
import chex
import jax.numpy as jnp
import jax.random

from fabjax.types import LogProbFunc
from fabjax.sampling_methods.mcmc.base import TransitionOperator
from blackjax.mcmc.hmc import init, kernel
from blackjax.mcmc.integrators import velocity_verlet
from functools import partial

class HamiltoneanMonteCarloBlackJax(TransitionOperator):
    def __init__(self,
                 dim,
                 n_intermediate_distributions,
                 n_outer_steps=1, n_inner_steps=5,
                 init_step_size: float = 0.1):
        self.dim = dim
        self.n_intermediate_distributions = n_intermediate_distributions
        self.n_outer_steps = n_outer_steps
        self.n_inner_steps = n_inner_steps
        self.init_step_size = init_step_size
        self.inverse_mass_matrix = jnp.ones(dim)
        self.integrator = velocity_verlet
        self._step = partial(
            kernel(self.integrator, divergence_threshold=1000),
            step_size=self.init_step_size,
            inverse_mass_matrix=self.inverse_mass_matrix,
            num_integration_steps=self.n_inner_steps)


    def get_init_state(self) -> chex.ArrayTree:
        return jnp.array(float("nan"))

    def run(self,
            key: chex.PRNGKey,
            transition_operator_state, # for now blank
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[chex.Array, chex.ArrayTree, dict]:
        init_state = jax.vmap(init, in_axes=(0, None))(x, transition_target_log_prob)
        hmc_state = init_state
        step_fn = partial(self._step, logprob_fn=transition_target_log_prob)

        for i in range(self.n_outer_steps):
            key, subkey = jax.random.split(key)
            key_batch = jax.random.split(subkey, x.shape[0])
            hmc_state, info = jax.vmap(step_fn)(
                key_batch,
                hmc_state,
                )
        info = {f"mean_p_accept": jnp.mean(info.acceptance_probability)}
        return hmc_state.position, transition_operator_state, info



if __name__ == '__main__':
    import distrax
    import matplotlib.pyplot as plt

    dim = 2
    n_intermediate_distributions = 2
    batch_size = 500
    loc = 1
    n_outer_steps = 5
    target_distribution = distrax.MultivariateNormalDiag(loc=jnp.zeros((dim,)) + loc,
                                                         scale_diag=jnp.ones((dim,)))
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(shape=(batch_size, dim), key=key) * 0.1
    x_init = x

    hmc = HamiltoneanMonteCarloBlackJax(dim, 2, n_outer_steps=n_outer_steps)
    transition_operator_state = hmc.get_init_state()
    key, subkey = jax.random.split(key, 2)
    x, transition_operator_state, info = hmc.run(subkey, transition_operator_state,
                                                 x, 0, target_distribution.log_prob)

    x_linspace = jnp.linspace(-5, 5, 100)
    x_linspace = jnp.stack((x_linspace, jnp.zeros_like(x_linspace)), axis=-1)
    plt.plot(x_linspace[:, 0], jnp.exp(target_distribution.log_prob(x_linspace)))
    plt.hist(x[:, 0], density=True, bins=200)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()

    plt.plot(x_linspace[:, 0], jnp.exp(target_distribution.log_prob(x_linspace)))
    plt.hist(x_init[:, 0], density=True, bins=200)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()


