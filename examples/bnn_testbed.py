from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fab.target_distributions.bnn_testbed import BNNEnergyFunction
from fab.agent.fab_agent import AgentFAB
from fab.utils.plotting import plot_history, plot_marginal_pair, plot_contours_2D
import matplotlib.pyplot as plt
import optax
import jax
import copy
from functools import partial
import jax.numpy as jnp

from neural_testbed import generative
import os
import pathlib
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
import jax
import optax
import matplotlib.pyplot as plt
import plotnine as gg

from fab.utils.logging import PandasLogger, WandbLogger, Logger
from fab.types import HaikuDistribution
from fab.agent.fab_agent import AgentFAB



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

def setup_flow(cfg: DictConfig, dim) -> HaikuDistribution:
    assert cfg.flow.type == "rnvp"
    from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
    flow = make_realnvp_dist_funcs(
        x_ndim=dim,
        flow_num_layers=cfg.flow.n_layers,
        mlp_hidden_size_per_x_dim=cfg.flow.layer_nodes_per_dim,
        use_exp=True,
        layer_norm=cfg.flow.layer_norm,
        act_norm=cfg.flow.act_norm)
    return flow


def setup_target(cfg: DictConfig):
    bnn_problem = BNNEnergyFunction(bnn_mlp_units=cfg.bnn.mlp_units)
    dim = bnn_problem.dim
    print(f"running bnn with {dim} parameters")
    return bnn_problem, dim


def make_enn(fab_agent: AgentFAB, bnn_problem, state):
    batch_size = 10
    # @jax.jit
    def enn_sampler(x, key):
        key1, key2, key3 = jax.random.split(key, 3)
        base_log_prob = fab_agent.get_base_log_prob(state.learnt_distribution_params)
        target_log_prob = fab_agent.get_target_log_prob(state.learnt_distribution_params)
        x_base, log_q_x_base = fab_agent.learnt_distribution.sample_and_log_prob.apply(
            state.learnt_distribution_params, rng=key1,
            sample_shape=(batch_size,))
        x_ais_loss, log_w_ais, _, _ = \
            fab_agent.annealed_importance_sampler.run(
                x_base, log_q_x_base, key2,
                state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=target_log_prob
            )
        index = jax.random.choice(jax.random.PRNGKey(0), log_w_ais.shape[0],
                                  p=jax.nn.softmax(log_w_ais), shape=(),
                                  replace=True)
        theta_tree = bnn_problem.array_to_tree(x_ais_loss[index])
        dist_y = bnn_problem.bnn.apply(theta_tree, x)
        return jnp.squeeze(dist_y.distribution.logits, axis=-2)  # dist_y.sample(seed=key3)

    return enn_sampler


def make_evaluator(agent, target):
    # TODO: make jittable (i.e. seperate things which are a function of agent.state)
    def evaluator(outer_batch_size, inner_batch_size, state):
        key1, key2 = jax.random.split(state.key)
        key_batch = jax.random.split(key1, inner_batch_size)
        enn_sampler = make_enn(agent, target, state)
        test_x, test_y = jax.vmap(target.problem.data_sampler.test_data)(key_batch)[0]
        test_x, test_y = jnp.squeeze(test_x, axis=1), jnp.squeeze(test_y, axis=1)
        logits = enn_sampler(test_x, key2)
        test_log_prob = jnp.mean(logits[test_y])
        # quality = target.problem.evaluate_quality(enn_sampler)
        # info = {"kl_estimate": quality.kl_estimate}
        # info.update(quality.extra)
        info = {"test_log_prob": test_log_prob}
        return info
    return evaluator


def make_wrapped_gg_plot(plot: gg.ggplot):
    """So we can save using the savefig method in our training loop."""
    class wrapped_gg_plot:
        def __init__(self, plot):
            self.plot = plot
        def savefig(self, path):
            self.plot.save(filename=path[:-4], path="")
    return wrapped_gg_plot(plot)


def setup_plotter(bnn_problem, show=False):
    def plotter(fab_agent):
        enn_sampler = jax.jit(make_enn(fab_agent, bnn_problem, fab_agent.state))
        plots = generative.sanity_plots(bnn_problem.problem, enn_sampler)
        p1 = plots['more_enn']
        p2 = plots['sample_enn']
        if show:
            _ = p1.draw(show=True, return_ggplot=True)
            _ = p2.draw(show=True, return_ggplot=True)
        return [make_wrapped_gg_plot(p1), make_wrapped_gg_plot(p2)]
    return plotter


def _run(cfg: DictConfig):
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

    target, dim = setup_target(cfg)
    flow = setup_flow(cfg, dim)
    if cfg.training.max_grad_norm is not None:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip_by_global_norm(cfg.training.max_grad_norm),
                                optax.adam(cfg.training.lr))
    else:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.adam(cfg.training.lr))
    assert cfg.fab.transition_operator.type == "hmc_tfp"
    AIS_kwargs = {"transition_operator_type": cfg.fab.transition_operator.type,
        "additional_transition_operator_kwargs":
                      {
                       "n_inner_steps": cfg.fab.transition_operator.n_inner_steps}
                  }
    plotter = setup_plotter(target)
    agent = AgentFAB(learnt_distribution=flow,
                     target_log_prob=target.log_prob,
                     n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                     AIS_kwargs=AIS_kwargs,
                     seed=cfg.training.seed,
                     optimizer=optimizer,
                     loss_type=cfg.fab.loss_type,
                     plotter=plotter,
                     logger=logger,
                     evaluator=None)
    evaluator = make_evaluator(agent, target)  # we need the agent to make the evaluator
    agent.evaluator = evaluator
    evaluated = evaluator(0, 2, agent.state)
    print(evaluated)
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



@hydra.main(config_path="./config", config_name="bnn.yaml")
def run(cfg: DictConfig):
    _run(cfg)


if __name__ == '__main__':
    run()
