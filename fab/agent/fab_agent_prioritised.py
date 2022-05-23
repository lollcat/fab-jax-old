from typing import Optional, Callable, NamedTuple, Tuple, Dict, Any, Iterable, Union

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

from fab.types import LogProbFunc, HaikuDistribution
from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab.utils.logging import Logger, ListLogger, to_numpy
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights
from fab.utils.prioritised_replay_buffer import PrioritisedBufferState, PrioritisedReplayBuffer


class State(NamedTuple):
    key: chex.PRNGKey
    learnt_distribution_params: hk.Params
    optimizer_state: optax.OptState
    transition_operator_state: chex.ArrayTree
    buffer_state: Optional[PrioritisedBufferState] = ()


Info = Dict[str, Any]
Agent = Any
BatchSize = int
Plotter = Callable[[Agent], Iterable[plt.Figure]]
Evaluator = Callable[[BatchSize, BatchSize, State], Dict[str, chex.Array]]


class PrioritisedAgentFAB:
    """Flow Annealed Importance Sampling Bootstrap Agent"""
    def __init__(self,
                 learnt_distribution: HaikuDistribution,
                 target_log_prob: LogProbFunc,
                 replay_buffer: PrioritisedReplayBuffer,
                 n_intermediate_distributions: int = 2,
                 n_buffer_updates_per_forward: int = 4,
                 max_w_adjust: float = 10.0,
                 AIS_kwargs: Dict = {"transition_operator_type": "hmc_tfp"},
                 seed: int = 0,
                 optimizer: optax.GradientTransformation = optax.adam(1e-4),
                 plotter: Optional[Plotter] = None,
                 logger: Optional[Logger] = None,
                 evaluator: Optional[Evaluator] = None,
                 ):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
        self.max_w_adjust = max_w_adjust
        self.plotter = plotter
        self.evaluator = evaluator
        if logger is None:
            self.logger = ListLogger(save=False)
        else:
            self.logger = logger
        self.annealed_importance_sampler = AnnealedImportanceSampler(dim=self.learnt_distribution.dim,
                n_intermediate_distributions=n_intermediate_distributions, **AIS_kwargs)
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.n_buffer_updates_per_forward = n_buffer_updates_per_forward
        self.batch_size: int
        self.state = self.init_state(seed)

    def init_state(self, seed, batch_size: int = 100) -> State:
        """Initialise the state of the fab agent."""
        key = jax.random.PRNGKey(seed)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        dummy_x = jnp.zeros((1, self.learnt_distribution.dim))
        learnt_distribution_params = self.learnt_distribution.log_prob.init(subkey1,
                                                                            dummy_x)

        optimizer_state = self.optimizer.init(learnt_distribution_params)
        transition_operator_state = self.annealed_importance_sampler.get_init_state()
        # dummy state
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      transition_operator_state=transition_operator_state,
                      optimizer_state=optimizer_state,
                      buffer_state=None)
        # init prioritised replay state
        def sampler(rng_key):
            # get samples to init buffer
            x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, \
            ais_info = self.forward(batch_size, state, rng_key)
            return x_ais, log_w_ais, log_q_x_base
        buffer_state = self.replay_buffer.init(subkey2, sampler)
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      transition_operator_state=transition_operator_state,
                      optimizer_state=optimizer_state,
                      buffer_state=buffer_state)

        return state

    def get_base_log_prob(self, params):
        """Currently the base log prob is always the learnt distribution."""
        def base_log_prob(x):
            return self.learnt_distribution.log_prob.apply(
                params, x)
        return base_log_prob


    def get_ais_target_log_prob(self, params):
        """Get the target log prob function for AIS, which is p^2/q."""
        def target_log_prob(x):
            return 2 * self.target_log_prob(x) - self.learnt_distribution.log_prob.apply(
                params, x)
        return target_log_prob

    def forward(self, batch_size: int, state: State, key,
                train: bool = True):
        """Note eval we always target p, for training we sometimes experiment with
        other targets for AIS."""
        subkey1, subkey2 = jax.random.split(key, 2)
        # get base and target log prob
        base_log_prob = self.get_base_log_prob(state.learnt_distribution_params)
        if train:
            target_log_prob = self.get_ais_target_log_prob(state.learnt_distribution_params)
        else:  # used for eval
            target_log_prob = self.target_log_prob
        x_base, log_q_x_base = self.learnt_distribution.sample_and_log_prob.apply(
                state.learnt_distribution_params, rng=subkey1,
            sample_shape=(batch_size,))
        x_ais, log_w_ais, transition_operator_state, ais_info = \
            self.annealed_importance_sampler.run(
                x_base, log_q_x_base, subkey2,
                state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=target_log_prob
            )
        return x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, ais_info

    def loss(self, x: chex.Array, log_q_old: chex.Array,
             learnt_distribution_params: chex.ArrayTree) -> \
            chex.Array:
        log_q = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x)
        w_adjust = jax.lax.stop_gradient(jnp.clip(jnp.exp(log_q_old - log_q),
                                                  a_max=self.max_w_adjust))
        return -jnp.mean(w_adjust*log_q)


    def sgd_step(self, x: chex.Array, log_q_old: chex.Array,
                 learnt_distribution_params: chex.ArrayTree, opt_state: chex.ArrayTree):
        loss, grads = jax.value_and_grad(
            self.loss, argnums=2)(
            x, log_q_old, learnt_distribution_params)
        updates, opt_state = self.optimizer.update(grads, opt_state,
                                                       params=learnt_distribution_params)
        learnt_distribution_params = optax.apply_updates(learnt_distribution_params, updates)
        info = {}
        info.update(grad_norm=optax.global_norm(grads), update_norm=optax.global_norm(updates))
        return learnt_distribution_params, opt_state, info


    @partial(jax.jit, static_argnums=(0, 1))
    def step(self, batch_size: int, state: State) -> Tuple[State, Info]:
        """Perform 1 AIS forward pass, and then multiple sgd steps sampling from the prioritised
        replay buffer."""
        info = {}
        # perform ais forward pass
        key, subkey1, subkey2, buffer_key = jax.random.split(state.key, 4)
        x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, \
        ais_info = self.forward(batch_size, state, subkey1)
        info.update(ais_info)

        buffer_state = self.replay_buffer.add(x_ais, log_w_ais, log_q_x_base, state.buffer_state)

        # now do replay sampling
        buffer_key, subkey = jax.random.split(buffer_key)
        minibatches = self.replay_buffer.sample_n_batches(
                buffer_state=state.buffer_state,
                n_batches=self.n_buffer_updates_per_forward,
                key=subkey,
                batch_size=batch_size)

        # gradient steps on minibatches
        def scan_sgd_step_fn(carry, xs):
            """Perform sgd on minibatches."""
            learnt_distribution_params, opt_state = carry
            x, log_w, log_q_old, indices = xs
            learnt_distribution_params, opt_state, info = self.sgd_step(x, log_q_old,
                                                                        learnt_distribution_params,
                                                                        opt_state)
            return (learnt_distribution_params, opt_state), info

        (learnt_distribution_params, opt_state), sgd_step_info = jax.lax.scan(
            scan_sgd_step_fn,
            init=(state.learnt_distribution_params, state.optimizer_state),
            xs=minibatches
        )
        info.update(jax.tree_map(jnp.mean, sgd_step_info))

        # adjust weights in the buffer
        def scan_w_adjust_fn(carry, xs):
            """Adjust weights on minibatch according to new flow params."""
            buffer_state = carry
            x, log_w, log_q_old, indices = xs
            log_q = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x)
            log_w_adjust = log_q_old - log_q
            buffer_state = self.replay_buffer.adjust(log_w_adjust, log_q, indices, buffer_state)
            return buffer_state, None

        buffer_state, _ = jax.lax.scan(
            scan_w_adjust_fn,
            init=buffer_state,
            xs=minibatches
        )

        state = State(key=key,
                      learnt_distribution_params=learnt_distribution_params,
                      optimizer_state=opt_state,
                      transition_operator_state=transition_operator_state,
                      buffer_state=buffer_state)

        return state, info

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def get_eval_info(self, outer_batch_size: int, inner_batch_size: int, state: State) -> Info:
        """Evaluate the model. We split outer_batch_size into chunks of size inner_batch_size
        to prevent overloading the GPU.
        """
        n_inner_batch = outer_batch_size // inner_batch_size

        def scan_func(carry, x):
            key = carry
            key, subkey = jax.random.split(key)
            # w.r.t target
            x_base, log_q_x_base, _, log_w_ais_target, _, _ = \
                self.forward(inner_batch_size, state, subkey, train=False)
            log_w_target = self.target_log_prob(x_base) - log_q_x_base

            # w.r.t p^2/q
            _, _, _, log_w_ais_p2_over_q, _, _ = \
                self.forward(inner_batch_size, state, subkey, train=True)
            log_w_p2_over_q = self.get_ais_target_log_prob(state.learnt_distribution_params)(x_base) \
                              - log_q_x_base
            return key, (log_w_ais_target, log_w_target, log_w_ais_p2_over_q, log_w_p2_over_q)

        _, (log_w_ais_target, log_w_target, log_w_ais_p2_over_q, log_w_p2_over_q) = \
            jax.lax.scan(scan_func, state.key, jnp.arange(n_inner_batch))
        eval_info = {}
        eval_info.update(
            eval_ess_ais_p=effective_sample_size_from_unnormalised_log_weights(
                log_w_ais_target.flatten()),
            eval_ess_flow=effective_sample_size_from_unnormalised_log_weights(log_w_target.flatten()),
            eval_ess_ais_p2_over_q=effective_sample_size_from_unnormalised_log_weights(
                log_w_ais_p2_over_q.flatten()),
            eval_ess_flow_p2_over_q=
            effective_sample_size_from_unnormalised_log_weights(log_w_p2_over_q.flatten()))
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


