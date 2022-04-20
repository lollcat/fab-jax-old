from typing import Optional, Callable, NamedTuple, Tuple, Dict, Any, Iterable, Iterator

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
import time
import tensorflow_probability.substrates.jax as tfp

from fab.agent.fab_agent import AgentFAB, State, Info, Agent, BatchSize, Evaluator, Plotter
from fab.utils.logging import Logger, ListLogger, to_numpy
from fab.utils.numerical_utils import effective_sample_size_from_unnormalised_log_weights
from fab.utils.training import DatasetIterator


class AgentTargetSamples(AgentFAB):
    """Agent trained on maximum likeihood using samples from target agent"""
    dataset: DatasetIterator
    def __init__(self, num_results=int(1e4), num_burnin_steps=int(1e3),
                 *args, **kwargs):
        super(AgentTargetSamples, self).__init__(*args, **kwargs)
        self.target_samples = self.setup_dataset(num_results, num_burnin_steps)

    def setup_dataset(self, num_results=int(1e4), num_burnin_steps=int(1e3), shuffle=True):

        # see https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX
        init_key, sample_key, shuffle_key = jax.random.split(jax.random.PRNGKey(0), 3)
        init_params = jnp.zeros(self.learnt_distribution.dim)
        @jax.jit
        def run_chain(key, state):
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.target_log_prob,
                    num_leapfrog_steps=3,
                    step_size=1.),
                num_adaptation_steps=int(num_burnin_steps * 0.8))
            return tfp.mcmc.sample_chain(
                      num_results=num_results,
                      num_burnin_steps=num_burnin_steps,
                      num_steps_between_results=10,
                      current_state=state,
                      kernel=kernel,
                      trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                      seed=key,
                      parallel_iterations=100,
            )
        start_time = time.time()
        states, is_accepted = run_chain(sample_key, init_params)
        print(f"time to generate dataset: {(time.time() - start_time) / 60}  min")
        if shuffle:
            states = jax.random.shuffle(x=states, key=shuffle_key, axis=0)
        return states



    def loss(self, x_samples, learnt_distribution_params, rng_key):
        log_q_x = self.learnt_distribution.log_prob.apply(learnt_distribution_params, x_samples)
        loss = - jnp.mean(log_q_x)
        return loss


    def update(self, x_samples, learnt_distribution_params, opt_state, rng_key):
        loss, grads = jax.value_and_grad(self.loss, argnums=1)(x_samples,
                                                               learnt_distribution_params,
                                                               rng_key)
        updates, opt_state = self.optimizer.update(grads, opt_state,
                                                       params=learnt_distribution_params)
        learnt_distribution_params = optax.apply_updates(learnt_distribution_params, updates)
        info = {"loss": loss,
                "grad_norm": optax.global_norm(grads)}
        return learnt_distribution_params, opt_state, info

    @partial(jax.jit, static_argnums=(0,))
    def step(self, x_samples: chex.Array, state: State) -> Tuple[State, Info]:
        key, subkey = jax.random.split(state.key)
        x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, \
        ais_info = self.forward(self.batch_size, state, subkey)
        key, subkey = jax.random.split(key)
        learnt_distribution_params, optimizer_state, info = \
            self.update(x_samples, state.learnt_distribution_params,
                        state.optimizer_state, subkey)
        state = State(key=key, learnt_distribution_params=learnt_distribution_params,
                      optimizer_state=optimizer_state,
                      transition_operator_state=transition_operator_state)
        info.update(ais_info)
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
            x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, ais_info = \
                self.forward(inner_batch_size, state, subkey)
            log_w = self.target_log_prob(x_base) - log_q_x_base
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
        self.dataset = DatasetIterator(self.batch_size, self.target_samples)

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
            for mini_batch in self.dataset:
                self.state, info = self.step(mini_batch, self.state)
            if i % logging_freq == 0:
                info = to_numpy(info)
                info.update(step=i)
                self.logger.write(info)
                if i % max(10*logging_freq, 100) == 0:
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


