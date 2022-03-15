from typing import Optional, Callable, NamedTuple, Tuple, Dict, Any

import chex
import jax.numpy as jnp
import haiku as hk
from functools import partial
import numpy as np
import jax
import optax
from tqdm import tqdm

from fab.types_ import TargetLogProbFunc, HaikuDistribution
from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab.utils.logging import Logger, ListLogger, to_numpy


Info = Dict[str, Any]


class State(NamedTuple):
    key: chex.PRNGKey
    learnt_distribution_params: hk.Params
    optimizer_state: optax.OptState
    transition_operator_state: chex.ArrayTree


class AgentFAB:
    """Flow Annealed Importance Sampling Bootstrap Agent"""
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: TargetLogProbFunc,
                 batch_size: int,
                 n_intermediate_distributions: int = 2,
                 loss_type: str = "alpha_2_div",
                 AIS_kwargs = None,
                 seed: int = 0,
                 optimizer: optax.GradientTransformation = optax.adam(1e-4),
                 plotter: Optional[Callable] = None,
                 logger: Logger = ListLogger(save=False)
                 ):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        self.batch_size = batch_size
        assert loss_type in ["alpha_2_div", "forward_kl"]
        self.loss_type = loss_type
        self.plotter = plotter
        self.logger = logger
        self.annealed_importance_sampler = AnnealedImportanceSampler(learnt_distribution,
                                                                     target_log_prob,
                                                                     batch_size,
                                                                     n_intermediate_distributions,
                                                                     **AIS_kwargs)
        self.optimizer = optimizer
        self.state = self.init_state(seed)

    def init_state(self, seed) -> State:
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        dummy_x = jnp.zeros((self.batch_size, self.learnt_distribution.dim))
        learnt_distribution_params = self.learnt_distribution.log_prob.init(subkey,
                                                                            dummy_x)

        optimizer_state = self.optimizer.init(learnt_distribution_params)
        transition_operator_state = self.annealed_importance_sampler.\
            transition_operator_manager.get_init_state()
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      transition_operator_state=transition_operator_state,
                      optimizer_state=optimizer_state)
        return state


    @partial(jax.jit, static_argnums=0)
    def step(self, state: State) -> Tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        x_ais, log_w_ais, transition_operator_state, ais_info = self.annealed_importance_sampler.run(
            subkey, state.learnt_distribution_params, state.transition_operator_state)
        learnt_distribution_params, optimizer_state, info = \
            self.update(x_ais, log_w_ais, state.learnt_distribution_params,
                        state.optimizer_state)
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      optimizer_state=optimizer_state,
                      transition_operator_state=transition_operator_state)
        info.update(ais_info)
        return state, info


    def run(self, n_iter, n_plots: int = None):
        if n_plots is not None:
            plot_iter = list(np.linspace(0, n_iter - 1, n_plots, dtype="int"))

        pbar = tqdm(range(n_iter))
        for i in pbar:
            self.state, info = self.step(self.state)
            info = to_numpy(info)
            self.logger.write(info)
            if i % 500:
                pbar.set_description(f"ess_ais: {info['ess_ais']}, ess_base: {info['ess_base']}")
            if n_plots is not None:
                if i in plot_iter:
                    self.plotter(self)



    def update(self, x_ais, log_w_ais, learnt_distribution_params, opt_state):
        (alpha_2_loss, (log_w, log_q_x, log_p_x)), grads = jax.value_and_grad(
            self.loss, argnums=2, has_aux=True)(
            x_ais, log_w_ais, learnt_distribution_params
        )
        updates, opt_state = self.optimizer.update(grads, opt_state,
                                                       params=learnt_distribution_params)
        learnt_distribution_params = optax.apply_updates(learnt_distribution_params, updates)
        info = self.get_info(x_ais, log_w_ais, log_w, log_q_x, log_p_x, alpha_2_loss)
        return learnt_distribution_params, opt_state, info


    def loss(self, x_samples, log_w_ais, learnt_distribution_params):
        if self.loss_type == "alpha_2_div":
            return self.alpha_2_loss(x_samples, log_w_ais, learnt_distribution_params)
        else:
            return self.forward_kl_loss(x_samples, log_w_ais, learnt_distribution_params)

    def alpha_2_loss(self, x_samples, log_w_ais, learnt_distribution_params):
        """Minimise upper bound of $\alpha$-divergence with $\alpha=2$."""
        valid_samples = jnp.isfinite(log_w_ais) & jnp.all(jnp.isfinite(x_samples), axis=-1)
        # remove invalid x_samples so we don't get NaN gradients.
        x_samples = jnp.where(valid_samples[:, None].repeat(x_samples.shape[-1], axis=-1),
                              x_samples, jnp.zeros_like(x_samples))
        log_q_x = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x_samples)
        log_p_x = self.target_log_prob(x_samples)
        log_w = log_p_x - log_q_x
        inner_term = log_w_ais + log_w
        # give invalid x_sample terms 0 importance weight.
        inner_term = jnp.where(valid_samples, inner_term, -jnp.ones_like(inner_term) * float("inf"))
        alpha_2_loss = jax.nn.logsumexp(inner_term, axis=0)
        return alpha_2_loss, (log_w, log_q_x, log_p_x)

    def forward_kl_loss(self, x_samples, log_w_ais, learnt_distribution_params):
        w_ais = jax.nn.softmax(log_w_ais, axis=0)
        log_q_x = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x_samples)
        log_p_x = self.target_log_prob(x_samples)
        log_w = log_p_x - log_q_x
        return - jnp.mean(w_ais * log_q_x), (log_w, log_q_x, log_p_x)

    @staticmethod
    def get_info(x_ais, log_w_ais, log_w, log_q_x, log_p_x, alpha_2_loss):
        info = {}
        mean_log_p_x = jnp.mean(log_p_x)
        info.update(loss=alpha_2_loss,
                    mean_log_p_x=mean_log_p_x,
                    n_non_finite_ais_log_w=jnp.sum(~jnp.isfinite(log_w_ais)),
                    n_non_finite_ais_x_samples=jnp.sum(~jnp.isfinite(x_ais[:, 0])))
        return info