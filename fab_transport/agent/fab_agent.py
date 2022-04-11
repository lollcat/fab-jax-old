from typing import Optional, Callable, NamedTuple, Tuple, Dict, Any, Iterable

import chex
import jax.numpy as jnp
import haiku as hk
from functools import partial
import numpy as np
import jax
import optax
import pickle
import os
from tqdm import tqdm
import pathlib
import matplotlib.pyplot as plt

from fab_transport.types import LogProbFunc, HaikuDistribution
from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab.utils.logging import Logger, ListLogger, to_numpy
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights


class State(NamedTuple):
    key: chex.PRNGKey
    learnt_distribution_params: hk.Params
    optimizer_state: optax.OptState
    transition_operator_state: chex.ArrayTree


Info = Dict[str, Any]
Agent = Any
BatchSize = int
Plotter = Callable[[Agent], Iterable[plt.Figure]]
Evaluator = Callable[[BatchSize, BatchSize, State], Dict[str, chex.Array]]


class AgentFAB:
    """Flow Annealed Importance Sampling Bootstrap Agent"""
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: LogProbFunc,
                 n_intermediate_distributions: int = 2,
                 loss_type: str = "alpha_2_div",
                 soften_ais_weights: bool = False,
                 style: str = "vanilla",
                 add_reverse_kl_loss: bool = False,
                 reverse_kl_loss_coeff: float = 0.001,
                 AIS_kwargs: Dict = {"transition_operator_type": "hmc_tfp"},
                 seed: int = 0,
                 optimizer: optax.GradientTransformation = optax.adam(1e-4),
                 plotter: Optional[Plotter] = None,
                 logger: Optional[Logger] = None,
                 evaluator: Optional[Evaluator] = None,
                 ):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        assert loss_type in ["alpha_2_div", "forward_kl", "sample_prob"]
        assert style in ["vanilla", "proptoloss"]
        self.loss_type = loss_type
        self.style = style
        self.plotter = plotter
        self.evaluator = evaluator
        if logger is None:
            self.logger = ListLogger(save=False)
        else:
            self.logger = logger
        self.annealed_importance_sampler = AnnealedImportanceSampler(dim=self.learnt_distribution.dim,
                n_intermediate_distributions=n_intermediate_distributions, **AIS_kwargs)
        self.optimizer = optimizer
        self.state = self.init_state(seed)
        self.reverse_kl_loss_coeff = reverse_kl_loss_coeff
        self.add_reverse_kl_loss = add_reverse_kl_loss
        self.soften_ais_weights = soften_ais_weights
        self.batch_size: int

    def init_state(self, seed) -> State:
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)

        dummy_x = jnp.zeros((1, self.learnt_distribution.dim))
        learnt_distribution_params = self.learnt_distribution.log_prob.init(subkey,
                                                                            dummy_x)

        optimizer_state = self.optimizer.init(learnt_distribution_params)
        transition_operator_state = self.annealed_importance_sampler.get_init_state()
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      transition_operator_state=transition_operator_state,
                      optimizer_state=optimizer_state)
        return state

    def get_base_log_prob(self, params):
        """Base distribution in latent space log prob. """
        def base_log_prob(z):
            return self.learnt_distribution.z_log_prob.apply(
                params, z)
        return base_log_prob


    def get_target_log_prob(self, params):
        if self.style == "vanilla":
            def target_log_prob(z):
                x, log_det = self.learnt_distribution.z_x_and_log_det_forward.apply(params, z)
                log_p_x = self.target_log_prob(x)
                chex.assert_equal_shape((log_p_x, log_det))
                return log_p_x + log_det
        elif self.style == "proptoloss":
            if self.loss_type == "alpha_2_div" or "sample_prob":
                def target_log_prob(z):
                    x, log_det = self.learnt_distribution.z_x_and_log_det_forward.apply(params, z)
                    log_q_z = self.learnt_distribution.z_log_prob.apply(params, z)
                    log_p_z = self.target_log_prob(x) + log_det
                    return 2 * log_p_z - log_q_z
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return target_log_prob


    def loss(self, z_samples, log_w_ais, learnt_distribution_params, rng_key):
        if self.style == "vanilla":
            if self.loss_type == "alpha_2_div":
                ais_loss, (log_w, log_q_x, log_p_x) = self.alpha_2_loss(z_samples, log_w_ais, learnt_distribution_params)
            else:
                ais_loss, (log_w, log_q_x, log_p_x) = self.forward_kl_loss(z_samples, log_w_ais, learnt_distribution_params)
        elif self.style == "proptoloss":
            # sample in proportion to the loss
            if self.loss_type == "alpha_2_div":
                ais_loss, (log_w, log_q_x, log_p_x) = self.alpha_2_div_proptoloss_target(z_samples, log_w_ais, learnt_distribution_params)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        loss = ais_loss
        if self.add_reverse_kl_loss:
            raise NotImplementedError
        return loss, (log_w, log_q_x, log_p_x)


    def reverse_kl_loss(self, batch_size, learnt_distribution_params, rng_key):
        raise NotImplementedError

    def alpha_2_div_proptoloss_target(self, z_ais, log_w_ais, learnt_distribution_params):
        """
        Loss when our ais chain targets p^2/q.
        """
        raise NotImplementedError
        valid_samples = jnp.isfinite(log_w_ais) & jnp.all(jnp.isfinite(z_ais), axis=-1)
        # remove invalid z_ais so we don't get NaN gradients.
        z_ais = jnp.where(valid_samples[:, None].repeat(z_ais.shape[-1], axis=-1),
                          z_ais, jnp.zeros_like(z_ais))
        log_q_x, log_p_x = self.get_log_q_and_log_p(learnt_distribution_params, z_ais)
        w_ais = jax.nn.softmax(log_w_ais, axis=0)
        return jnp.mean(w_ais * (2*log_p_x - log_q_x)), (log_p_x - log_q_x, log_q_x, log_p_x)

    def get_log_q_and_log_p(self, learnt_distribution_params, z):
        x_samples, log_det_forward = self.learnt_distribution.z_x_and_log_det_forward.apply(
            learnt_distribution_params, z)
        log_q_x = self.learnt_distribution.z_log_prob.apply(
            learnt_distribution_params, z) - log_det_forward
        log_p_x = self.target_log_prob(x_samples)
        return log_q_x, log_p_x


    def alpha_2_loss(self, z_ais, log_w_ais, learnt_distribution_params):
        """Minimise upper bound of $\alpha$-divergence with $\alpha=2$."""
        valid_samples = jnp.isfinite(log_w_ais) & jnp.all(jnp.isfinite(z_ais), axis=-1)
        # remove invalid x_samples so we don't get NaN gradients.
        z_ais = jnp.where(valid_samples[:, None].repeat(z_ais.shape[-1], axis=-1),
                          z_ais, jnp.zeros_like(z_ais))
        log_q_x, log_p_x = self.get_log_q_and_log_p(learnt_distribution_params, z_ais)
        log_w = log_p_x - log_q_x
        inner_term = log_w_ais + log_w
        # give invalid x_sample terms 0 importance weight.
        inner_term = jnp.where(valid_samples, inner_term, -jnp.ones_like(inner_term) * float("inf"))
        alpha_2_loss = jax.nn.logsumexp(inner_term, axis=0)
        return alpha_2_loss, (log_w, log_q_x, log_p_x)

    def forward_kl_loss(self, z_ais, log_w_ais, learnt_distribution_params):
        w_ais = jax.nn.softmax(log_w_ais, axis=0)
        log_q_x, log_p_x = self.get_log_q_and_log_p(learnt_distribution_params, z_ais)
        log_w = log_p_x - log_q_x
        return - jnp.mean(w_ais * log_q_x), (log_w, log_q_x, log_p_x)


    def update(self, x_ais, log_w_ais, learnt_distribution_params, opt_state, rng_key):
        (alpha_2_loss, (log_w, log_q_x, log_p_x)), grads = jax.value_and_grad(
            self.loss, argnums=2, has_aux=True)(
            x_ais, log_w_ais, learnt_distribution_params, rng_key
        )
        updates, opt_state = self.optimizer.update(grads, opt_state,
                                                       params=learnt_distribution_params)
        learnt_distribution_params = optax.apply_updates(learnt_distribution_params, updates)
        info = self.get_info(x_ais, log_w_ais, log_w, log_q_x, log_p_x, alpha_2_loss)
        return learnt_distribution_params, opt_state, info


    @partial(jax.jit, static_argnums=(0,1))
    def step(self, batch_size: int, state: State) -> Tuple[State, Info]:
        key, subkey1, subkey2, subkey3 = jax.random.split(state.key, 4)
        # get base and target log prob
        base_log_prob = self.get_base_log_prob(state.learnt_distribution_params)
        target_log_prob = self.get_target_log_prob(state.learnt_distribution_params)
        z_base, log_q_z_base = self.learnt_distribution.z_sample_and_log_prob.apply(
                state.learnt_distribution_params, rng=subkey1,
            sample_shape=(batch_size,))
        x_ais, log_w_ais, transition_operator_state, ais_info = \
            self.annealed_importance_sampler.run(
                z_base, log_q_z_base, subkey2,
                state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=target_log_prob
            )
        if self.soften_ais_weights:
            log_w_ais = self.clip_log_w_ais(log_w_ais)
        learnt_distribution_params, optimizer_state, info = \
            self.update(x_ais, log_w_ais, state.learnt_distribution_params,
                        state.optimizer_state, subkey3)
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      optimizer_state=optimizer_state,
                      transition_operator_state=transition_operator_state)
        info.update(ais_info)
        return state, info

    @partial(jax.jit, static_argnums=0)
    def clip_log_w_ais(self, log_w_ais):
        top_log_w = jnp.stack(self.top_k_func(log_w_ais))
        max_clip_low_w = jnp.min(top_log_w)  # get medium sized log_w
        log_w_ais = jnp.clip(log_w_ais, a_max=max_clip_low_w)
        return log_w_ais

    @staticmethod
    def get_info(x_ais, log_w_ais, log_w, log_q_x, log_p_x, alpha_2_loss) -> Info:
        """Get info for logging during training."""
        info = {}
        mean_log_p_x = jnp.mean(log_p_x)
        info.update(loss=alpha_2_loss,
                    mean_log_p_x=mean_log_p_x,
                    n_non_finite_ais_log_w=jnp.sum(~jnp.isfinite(log_w_ais)),
                    n_non_finite_ais_x_samples=jnp.sum(~jnp.isfinite(x_ais[:, 0])))
        return info

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def get_eval_info(self, outer_batch_size: int, inner_batch_size: int, state: State) -> Info:
        """Evaluate the model. We split outer_batch_size into chunks of size inner_batch_size
        to prevent overloading the GPU.
        """
        n_inner_batch = outer_batch_size // inner_batch_size
        base_log_prob = self.get_base_log_prob(state.learnt_distribution_params)
        target_log_prob = self.target_log_prob
        def scan_func(carry, x):
            key = carry
            key, subkey = jax.random.split(key)
            z_base, log_q_z_base = self.learnt_distribution.z_sample_and_log_prob.apply(
                state.learnt_distribution_params, rng=state.key,
                sample_shape=(inner_batch_size,))
            log_w = target_log_prob(z_base) - log_q_z_base
            x_ais, log_w_ais, transition_operator_state, ais_info = \
                self.annealed_importance_sampler.run(
                    z_base, log_q_z_base, subkey,
                    state.transition_operator_state,
                    base_log_prob=base_log_prob,
                    target_log_prob=target_log_prob
                )
            return key, (log_w_ais, log_w)

        _, (log_w_ais, log_w) = jax.lax.scan(scan_func, state.key, jnp.arange(n_inner_batch))
        eval_info = \
            {"eval_ess_ais": effective_sample_size_from_unnormalised_log_weights(
                log_w_ais.flatten()),
            "eval_ess_flow": effective_sample_size_from_unnormalised_log_weights(log_w.flatten())
            }
        if self.evaluator is not None:
            eval_info.update(self.evaluator(outer_batch_size, inner_batch_size, state))
        return eval_info


    def run(self,
            n_iter: int,
            batch_size: int,
            eval_batch_size: Optional[int] = None,
            n_evals: Optional[int] = None,
            n_plots: Optional[int] = None,
            n_checkpoints: Optional[int] = None,
            save: bool = False,
            plots_dir: str = "tmp/plots",
            checkpoints_dir: str = "tmp/chkpts",
            logging_freq: int = 1) -> None:
        """Train the fab model."""
        self.batch_size = batch_size
        max_frac = 0.05
        k = int(max_frac * self.batch_size)
        @jax.jit
        def top_k_func(a):
            return jax.lax.approx_max_k(a, k)
        self.top_k_func = top_k_func

        if save:
            pathlib.Path(plots_dir).mkdir(exist_ok=True, parents=True)
            pathlib.Path(checkpoints_dir).mkdir(exist_ok=True, parents=True)
        if n_checkpoints:
            checkpoint_iter = list(np.linspace(0, n_iter - 1, n_checkpoints, dtype="int"))
        if n_evals is not None:
            eval_iter = list(np.linspace(0, n_iter - 1, n_evals, dtype="int"))
            assert eval_batch_size is not None
            assert eval_batch_size % batch_size == 0
        if n_plots is not None:
            plot_iter = list(np.linspace(0, n_iter - 1, n_plots, dtype="int"))

        pbar = tqdm(range(n_iter))
        for i in pbar:
            self.state, info = self.step(batch_size, self.state)
            if i % logging_freq == 0:
                info = to_numpy(info)
                info.update(step=i)
                self.logger.write(info)
                if i % max(10*logging_freq, 100):
                    pbar.set_description(f"ess_ais: {info['ess_ais']}, ess_base: {info['ess_base']}")
            if n_evals is not None:
                if i in eval_iter:
                    eval_info = self.get_eval_info(
                        outer_batch_size=eval_batch_size,
                        inner_batch_size=batch_size,
                        state=self.state)
                    eval_info.update(step=i)
                    self.logger.write(eval_info)

            if n_plots is not None:
                if i in plot_iter:
                    figures = self.plotter(self)
                    if save:
                        for j, figure in enumerate(figures):
                            figure.savefig(os.path.join(plots_dir, f"{j}_iter_{i}.png"))

            if n_checkpoints is not None:
                if i in checkpoint_iter:
                    checkpoint_path = os.path.join(checkpoints_dir, f"iter_{i}/")
                    pathlib.Path(checkpoint_path).mkdir(exist_ok=False)
                    self.save(checkpoint_path)

        self.logger.close()


    def save(self, path: str):
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            pickle.dump(self.state, f)

    def load(self, path: str):
        self.state = pickle.load(open(os.path.join(path, "state.pkl"), "rb"))


