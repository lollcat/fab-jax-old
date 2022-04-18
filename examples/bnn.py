import os
import pathlib
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
import jax
import jax.numpy as jnp
import chex
import optax
import matplotlib.pyplot as plt

from fab.utils.logging import PandasLogger, WandbLogger, Logger
from fab.types import HaikuDistribution
from fab.agent.fab_agent import AgentFAB
from fab.target_distributions.bnn import BNNEnergyFunction



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


def make_posterior_log_prob(fab_agent: AgentFAB, bnn_problem, state):
    batch_size = 5
    def joint_log_prob(theta, x, y):
        # sum over sequence of x and y datapoints
        return jnp.sum(bnn_problem.bnn.apply(theta, x).log_prob(y))

    def get_posterior_log_prob(key, x, y):
        x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, ais_info = \
            fab_agent.forward(batch_size,  state, key, train=False)
        log_w = fab_agent.target_log_prob(x_base) - log_q_x_base
        theta_tree_base = jax.vmap(bnn_problem.array_to_tree)(x_base)
        theta_tree_ais = jax.vmap(bnn_problem.array_to_tree)(x_ais)
        log_q_y_base = jax.vmap(joint_log_prob, in_axes=(0, None, None))(theta_tree_base, x, y)
        log_q_y_ais = jax.vmap(joint_log_prob, in_axes=(0, None, None))(theta_tree_ais, x, y)
        log_q_y_base = jnp.sum(jax.nn.softmax(log_w, axis=0) * log_q_y_base, axis=0)
        log_q_y_ais = jnp.sum(jax.nn.softmax(log_w_ais, axis=0) * log_q_y_ais, axis=0)
        return log_q_y_base, log_q_y_ais
    return get_posterior_log_prob


def make_evaluator(agent, target: BNNEnergyFunction):
    tau = 10
    def evaluator(outer_batch_size, inner_batch_size, state):
        enn_sampler = make_posterior_log_prob(agent, target, state)
        def evaluate(rng_key):
            key1, key2 = jax.random.split(rng_key)
            test_x, test_y = target.generate_data(key1, tau)
            log_q_y_base, log_q_y_ais = enn_sampler(key2, test_x, test_y)
            test_log_p = jnp.sum(target.target_prob(test_x, test_y), axis=0)
            chex.assert_equal_shape((log_q_y_ais, log_q_y_base, test_log_p))
            return log_q_y_base, log_q_y_ais, test_log_p
        
        key_batch = jax.random.split(state.key, inner_batch_size)
        log_q_y_base, log_q_y_ais, test_log_p = jax.vmap(evaluate)(key_batch)
        expected_kl_base = jnp.mean(test_log_p - log_q_y_base)
        expected_kl_ais = jnp.mean(test_log_p - log_q_y_ais)
        info = {"test_log_q_ais": jnp.mean(log_q_y_ais),
                "test_log_q_base": jnp.mean(log_q_y_base),
                "exp_kl_div_ais": expected_kl_ais,
                "exp_kl_div_base": expected_kl_base}

        return info
    return evaluator



def setup_plotter(bnn_problem: BNNEnergyFunction, show=True):
    n_plots = 5
    def plotter(fab_agent: AgentFAB):
        fig, axs = plt.subplots(n_plots, 2, figsize=(10, 4*n_plots))
        keys = jax.random.split(fab_agent.state.key, n_plots)
        for i in range(n_plots):
            key = keys[i]
            x_base, _, x_ais, _, _, _ = \
                fab_agent.forward(1, fab_agent.state, key, train=False)
            x_ais = jnp.squeeze(x_ais, axis=0)
            x_base = jnp.squeeze(x_base, axis=0)
            bnn_problem.plot(bnn_problem.array_to_tree(x_base), axs[i, 0])
            bnn_problem.plot(bnn_problem.array_to_tree(x_ais), axs[i, 1])
        fig.tight_layout()
        if show:
            fig.show()
        return [fig]
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
    plotter(agent)
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
