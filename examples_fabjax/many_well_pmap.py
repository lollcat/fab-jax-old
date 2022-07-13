"""Setup full training procedure for the many well problem."""
from typing import Union

import os
import pathlib
import chex
import hydra
import wandb
import jax.numpy as jnp
from omegaconf import DictConfig
from datetime import datetime
import jax
import optax
import matplotlib.pyplot as plt

from fabjax.utils.logging import PandasLogger, WandbLogger, Logger
from fabjax.types import HaikuDistribution
from fabjax.utils.plotting import plot_marginal_pair, plot_contours_2D
from fabjax.agent.fab_agent import AgentFAB, Evaluator, State
from fabjax.agent.fab_agent_prioritised_pmap import PrioritisedAgentFAB
from fabjax.target_distributions.many_well import ManyWellEnergy, setup_manywell_evaluator
from fabjax.sampling_methods.mcmc.tfp_hamiltonean_monte_carlo import HamiltoneanMonteCarloTFP, HMCState
from fabjax.utils.replay_buffer import ReplayBuffer
from fabjax.utils.prioritised_replay_buffer import PrioritisedReplayBuffer


def setup_logger(cfg: DictConfig, save_path: str) -> Logger:
    if hasattr(cfg.logger, "pandas_logger"):
        logger = PandasLogger(save=True,
                              save_path=save_path + "logging_hist.csv",
                              save_period=cfg.logger.pandas_logger.save_period)
    elif hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


def setup_flow(cfg: DictConfig) -> HaikuDistribution:
    assert cfg.flow.type == "rnvp"
    from fabjax.learnt_distributions.real_nvp import make_realnvp_dist_funcs
    flow = make_realnvp_dist_funcs(
        x_ndim=cfg.target.dim,
        flow_num_layers=cfg.flow.n_layers,
        mlp_hidden_size_per_x_dim=cfg.flow.layer_nodes_per_dim,
        use_exp=cfg.flow.use_exp,
        layer_norm=cfg.flow.layer_norm,
        act_norm=cfg.flow.act_norm,
        lu_layer=cfg.flow.lu_layer
    )
    return flow


def setup_plotter(cfg, batch_size, dim, target):
    """For the many well problem."""
    def plot(fab_agent, dim=dim, key=jax.random.PRNGKey(0)):
        plotting_bounds = 3
        n_rows = dim // 2

        @jax.jit
        def get_info(state):
            base_log_prob = fab_agent.get_base_log_prob(state.learnt_distribution_params)
            target_log_prob = fab_agent.get_ais_target_log_prob(state.learnt_distribution_params)
            x_base, log_q_x_base = fab_agent.learnt_distribution.sample_and_log_prob.apply(
                state.learnt_distribution_params, rng=state.key,
                sample_shape=(batch_size,))
            x_ais_target, _, _, _ = \
                fab_agent.annealed_importance_sampler.run(
                    x_base, log_q_x_base, state.key,
                    state.transition_operator_state,
                    base_log_prob=base_log_prob,
                    target_log_prob=target_log_prob
                )

            if cfg.buffer.prioritised is True:
                buffer_samples = fab_agent.replay_buffer.sample(state.buffer_state, state.key,
                                                                batch_size)
                return x_base, x_ais_target, buffer_samples[0]
            else:
                return x_base, x_ais_target

        if cfg.buffer.prioritised is True:
            fig, axs = plt.subplots(dim // 2, 3, sharex=True, sharey=True, figsize=(10, n_rows * 3))
            x_base, x_ais_target, x_buffer = get_info(fab_agent.state_first_device)
        else:
            fig, axs = plt.subplots(dim // 2, 2, sharex=True, sharey=True, figsize=(15, n_rows * 3))
            x_base, x_ais_target = get_info(fab_agent.state_first_device)


        for i in range(n_rows):
            plot_contours_2D(target.log_prob_2D, bound=plotting_bounds, ax=axs[i, 0])
            plot_contours_2D(target.log_prob_2D, bound=plotting_bounds, ax=axs[i, 1])

            # plot flow samples
            plot_marginal_pair(x_base, ax=axs[i, 0], bounds=(-plotting_bounds, plotting_bounds),
                               marginal_dims=(i*2, i*2+1))
            axs[i, 0].set_xlabel(f"dim {i*2}")
            axs[i, 0].set_ylabel(f"dim {i*2 + 1}")



            # plot ais samples
            plot_marginal_pair(x_ais_target, ax=axs[i, 1],
                               bounds=(-plotting_bounds, plotting_bounds), marginal_dims=(i*2,i*2+1))
            axs[i, 1].set_xlabel(f"dim {i*2}")
            axs[i, 1].set_ylabel(f"dim {i*2+1}")

            # plot buffer samples
            if cfg.buffer.prioritised is True:
                plot_contours_2D(target.log_prob_2D, bound=plotting_bounds, ax=axs[i, 2])
                plot_marginal_pair(x_buffer, ax=axs[i, 2],
                                   bounds=(-plotting_bounds, plotting_bounds),
                                   marginal_dims=(i * 2, i * 2 + 1))
                axs[i, 2].set_xlabel(f"dim {i * 2}")
                axs[i, 2].set_ylabel(f"dim {i * 2 + 1}")


        axs[0, 1].set_title("ais samples")
        axs[0, 0].set_title("flow samples")
        if cfg.buffer.prioritised is True:
            axs[0, 2].set_title("buffer samples")
        plt.tight_layout()
        plt.show()
        return [fig]
    return plot


def _run(cfg: DictConfig):
    dim = cfg.target.dim  # applies to flow and target
    if cfg.training.use_64_bit:
        jax.config.update("jax_enable_x64", True)
    current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = cfg.evaluation.save_path + current_time + "/"
    if not hasattr(cfg.logger, "wandb"):
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=False)
    logger = setup_logger(cfg, save_path)
    if hasattr(cfg.logger, "wandb"):
        # if using wandb then save to wandb path
        save_path = os.path.join(wandb.run.dir, save_path)
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=False)
    with open(save_path + "config.txt", "w") as file:
        file.write(str(cfg))

    target = ManyWellEnergy(dim=dim)
    flow = setup_flow(cfg)
    base_optimizer = getattr(optax, cfg.training.optimizer_type)(cfg.training.lr)
    if cfg.training.max_grad_norm is not None:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(cfg.training.max_grad),
                                optax.clip_by_global_norm(cfg.training.max_grad_norm),
                                base_optimizer)
    else:
        optimizer = optax.chain(optax.zero_nans(),
                                base_optimizer)
    AIS_kwargs = {"transition_operator_type": cfg.fab.transition_operator.type,
        "additional_transition_operator_kwargs":
                      {
                       "n_inner_steps": cfg.fab.transition_operator.n_inner_steps,
                       "init_step_size": cfg.fab.transition_operator.init_step_size}
                  }
    plotter = setup_plotter(cfg, batch_size=512, dim=dim, target=target)

    evaluator = setup_manywell_evaluator(many_well=target, flow=flow)

    if cfg.buffer.use:
        if cfg.buffer.prioritised:
            buffer = PrioritisedReplayBuffer(
                dim=cfg.target.dim,
                max_length=cfg.buffer.maximum_buffer_length,
                min_sample_length=cfg.buffer.min_buffer_length)
        else:
            buffer = ReplayBuffer(dim=cfg.target.dim,
                                  max_length=cfg.buffer.maximum_buffer_length,
                                  min_sample_length=cfg.buffer.min_buffer_length)
    else:
        buffer = None
    if not cfg.buffer.use or not cfg.buffer.prioritised:
        agent = AgentFAB(learnt_distribution=flow,
                                    target_log_prob=target.log_prob,
                                    n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                                    replay_buffer=buffer,
                                    n_buffer_updates_per_forward=cfg.buffer.n_batches_buffer_sampling,
                                    AIS_kwargs=AIS_kwargs,
                                    seed=cfg.training.seed,
                                    optimizer=optimizer,
                                    loss_type=cfg.fab.loss_type,
                                    plotter=plotter,
                                    logger=logger,
                                    evaluator=evaluator
                         )
    else:
        agent = PrioritisedAgentFAB(learnt_distribution=flow,
                                    target_log_prob=target.log_prob,
                                    n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                                    replay_buffer=buffer,
                                    max_w_adjust=cfg.buffer.max_w_adjust,
                                    n_buffer_updates_per_forward=cfg.buffer.n_batches_buffer_sampling,
                                    AIS_kwargs=AIS_kwargs,
                                    seed=cfg.training.seed,
                                    optimizer=optimizer,
                                    plotter=plotter,
                                    logger=logger,
                                    evaluator=evaluator)

    # now we can run the agent
    agent.run(n_iter=cfg.training.n_iterations,
              batch_size=cfg.training.batch_size,
              eval_batch_size=cfg.evaluation.batch_size,
              n_evals=cfg.evaluation.n_evals,
              n_plots=cfg.evaluation.n_plots,
              n_checkpoints=cfg.evaluation.n_checkpoints,
              save=True,
              plots_dir=os.path.join(save_path, "plots"),
              checkpoints_dir=os.path.join(save_path, "checkpoints"))



@hydra.main(config_path="./config", config_name="many_well.yaml")
def run(cfg: DictConfig):
    _run(cfg)


if __name__ == '__main__':
    run()
