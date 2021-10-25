from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
import jax.numpy as jnp
from functools import partial
from fab.types import TargetLogProbFunc, HaikuDistribution
import haiku as hk
import jax
import optax
from tqdm import tqdm
jax.config.update("jax_enable_x64", True)


class AgentFAB:
    """Flow Annealed Importance Sampling Bootstrap Agent"""
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: TargetLogProbFunc,
                 batch_size: int,
                 n_iter : int,
                 n_intermediate_distributions: int = 3,
                 AIS_kwargs = None,
                 seed = 0,
                 lr = 1e-3,
                 ):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.annealed_importance_sampler = AnnealedImportanceSampler(learnt_distribution,
                                                                     target_log_prob,
                                                                     batch_size,
                                                                     n_intermediate_distributions,
                                                                     **AIS_kwargs)
        self.rng = hk.PRNGSequence(key_or_seed=seed)
        dummy_x = jnp.zeros((batch_size, learnt_distribution.dim))
        self.learnt_distribution_params = self.learnt_distribution.log_prob.init(next(self.rng),
                                                                                 dummy_x)
        self.optimizer = optax.chain(
            optax.clip(1.0),
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale(-lr))
        self.optimizer_state = self.optimizer.init(self.learnt_distribution_params)
        self.update = jax.jit(self._update)



    def run(self):
        pbar = tqdm(range(self.n_iter))
        for i in pbar:
            x_AIS, log_w_AIS = self.annealed_importance_sampler.run(next(self.rng),
                                                                    self.learnt_distribution_params)
            learnt_distribution_params, opt_state, (log_w, log_q_x) = \
                self.update(x_AIS, log_w_AIS, self.learnt_distribution_params,
                            self.optimizer_state)
            self.learnt_distribution_params, self.optimizer_state = learnt_distribution_params, \
                                                                    opt_state



    def _update(self, x_AIS, log_w_AIS, learnt_distribution_params, opt_state):
        (alpha_2_loss, (log_w, log_q_x)), grads = jax.value_and_grad(self._alpha_2_fab_loss,
                                                                  argnums=2, has_aux=True)(
            x_AIS, log_w_AIS, self.learnt_distribution_params
        )
        updates, new_opt_state = self.optimizer.update(grads, opt_state,
                                                       params=learnt_distribution_params)
        learnt_distribution_params = optax.apply_updates(learnt_distribution_params, updates)
        return learnt_distribution_params, opt_state, (log_w, log_q_x)


    def _alpha_2_fab_loss(self, x_samples, log_w_AIS, learnt_distribution_params):
        # alpha divergence, alpha = 2
        alpha = 2.0
        x_samples = jax.lax.stop_gradient(x_samples)
        log_w_AIS = jax.lax.stop_gradient(log_w_AIS)
        log_q_x = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x_samples)
        log_p_x = self.target_log_prob(x_samples)
        log_w = log_p_x - log_q_x
        # remove nans and infs by making them carry very little wait inside the exp
        # https://jax.readthedocs.io/en/latest/_modules/jax/_src/numpy/lax_numpy.html#nan_to_num
        neg_inf = -100.0
        b = jnp.where((log_w == neg_inf) | (log_w_AIS == neg_inf), 0.0, 1.0)
        log_w_AIS = jnp.nan_to_num(log_w_AIS, nan=neg_inf, neginf=neg_inf)
        log_w_AIS_normed = log_w_AIS - jax.nn.logsumexp(log_w_AIS)
        log_w = jnp.nan_to_num(log_w, nan=neg_inf, neginf=neg_inf)
        alpha_2_loss = jax.nn.logsumexp((alpha - 1) * log_w + log_w_AIS_normed, b=b)
        alpha_2_loss = jnp.clip(alpha_2_loss, a_max=1000, a_min=-1000)
        return alpha_2_loss, (log_w, log_q_x)
