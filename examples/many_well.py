"""Setup full training procedure for the many well problem."""
import os
import pathlib
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
import jax
import optax
import matplotlib.pyplot as plt

from fab.utils.logging import PandasLogger, WandbLogger, Logger
from fab.types_ import HaikuDistribution
from fab.utils.plotting import plot_marginal_pair, plot_contours_2D
from fab.agent.fab_agent import AgentFAB
from fab.target_distributions.many_well import ManyWellEnergy


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
    from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
    flow = make_realnvp_dist_funcs(
        x_ndim=cfg.target.dim,
        flow_num_layers=cfg.flow.n_layers,
        mlp_hidden_size_per_x_dim=cfg.flow.layer_nodes_per_dim,
        use_exp=True,
        layer_norm=cfg.flow.layer_norm,
        act_norm=cfg.flow.act_norm)
    return flow


def setup_plotter(batch_size, dim, target):
    """For the many well problem."""
    def plot(fab_agent, dim=dim, key=jax.random.PRNGKey(0)):
        plotting_bounds = 3
        n_rows = dim // 2
        fig, axs = plt.subplots(dim // 2, 2,  sharex=True, sharey=True, figsize=(10, n_rows*3))


        samples_flow = fab_agent.learnt_distribution.sample.apply(
            fab_agent.state.learnt_distribution_params, key, (batch_size,))
        samples_ais = fab_agent.annealed_importance_sampler.run(
            batch_size, key, fab_agent.state.learnt_distribution_params,
        fab_agent.state.transition_operator_state)[0]

        for i in range(n_rows):
            plot_contours_2D(target.log_prob_2D, bound=plotting_bounds, ax=axs[i, 0])
            plot_contours_2D(target.log_prob_2D, bound=plotting_bounds, ax=axs[i, 1])

            # plot flow samples
            plot_marginal_pair(samples_flow, ax=axs[i, 0], bounds=(-plotting_bounds, plotting_bounds), marginal_dims=(i*2,i*2+1))
            axs[i, 0].set_xlabel(f"dim {i*2}")
            axs[i, 0].set_ylabel(f"dim {i*2 + 1}")



            # plot ais samples
            plot_marginal_pair(samples_ais, ax=axs[i, 1], bounds=(-plotting_bounds, plotting_bounds), marginal_dims=(i*2,i*2+1))
            axs[i, 1].set_xlabel(f"dim {i*2}")
            axs[i, 1].set_ylabel(f"dim {i*2+1}")
            plt.tight_layout()
        axs[0, 1].set_title("ais samples")
        axs[0, 0].set_title("flow samples")
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
    optimizer = optax.chain(optax.zero_nans(),
                            optax.clip_by_global_norm(cfg.training.max_grad_norm),
                            optax.adamw(cfg.training.lr))
    assert cfg.fab.transition_operator.type == "HMC"
    AIS_kwargs = {"transition_operator_type": cfg.fab.transition_operator.type,
        "additional_transition_operator_kwargs":
                      {
                          "step_tuning_method": cfg.fab.transition_operator.step_tuning_method,
                       "n_inner_steps": cfg.fab.transition_operator.n_inner_steps}
                  }
    plotter = setup_plotter(batch_size=512, dim=dim, target=target)
    agent = AgentFAB(learnt_distribution=flow,
                     target_log_prob=target.log_prob,
                     n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                     AIS_kwargs=AIS_kwargs,
                     seed=cfg.training.seed,
                     optimizer=optimizer,
                     loss_type=cfg.fab.loss_type,
                     plotter=plotter,
                     logger=logger)

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