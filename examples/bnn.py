import os
import pathlib
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
import time

from fab.utils.logging import PandasLogger, WandbLogger, Logger
from fab.types import HaikuDistribution
from fab.agent.fab_agent import AgentFAB
from fab.agent.bbb_agent import AgentBBB
from fab.agent.target_samples_agent import AgentTargetSamples
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
    bnn_problem = BNNEnergyFunction(bnn_mlp_units=cfg.bnn.mlp_units,
                                    train_n_points=cfg.bnn.n_train_datapoints,
                                    seed=cfg.bnn.seed)
    dim = bnn_problem.dim
    print(f"running bnn with {dim} parameters")
    return bnn_problem, dim


def make_posterior_log_prob(fab_agent: AgentFAB, bnn_problem, state):
    batch_size = 50
    def joint_log_prob(theta, x, y):
        # sum over sequence of x and y datapoints
        dist = bnn_problem.bnn.apply(theta, x)
        return jnp.sum(dist.log_prob(y))

    def get_posterior_log_prob(key, x, y):
        x_base, log_q_x_base, x_ais, log_w_ais, transition_operator_state, ais_info = \
            fab_agent.forward(batch_size,  state, key, train=False)
        log_w = fab_agent.target_log_prob(x_base) - log_q_x_base
        log_w = log_w - jax.nn.logsumexp(log_w, axis=0)  # normalise weights
        log_w_ais = log_w_ais - jax.nn.logsumexp(log_w_ais, axis=0)  # normalise weights
        theta_tree_base = jax.vmap(bnn_problem.array_to_tree)(x_base)
        theta_tree_ais = jax.vmap(bnn_problem.array_to_tree)(x_ais)
        log_q_y_base = jax.vmap(joint_log_prob, in_axes=(0, None, None))(theta_tree_base, x, y)
        log_q_y_ais = jax.vmap(joint_log_prob, in_axes=(0, None, None))(theta_tree_ais, x, y)
        log_q_y_base = jax.nn.logsumexp(log_w + log_q_y_base, axis=0)
        log_q_y_ais = jax.nn.logsumexp(log_w_ais + log_q_y_ais, axis=0)
        return log_q_y_base, log_q_y_ais
    return get_posterior_log_prob

def setup_dataset(target: BNNEnergyFunction, total_size, batch_size = 100):
    num_burnin_steps = min(int(2e3), total_size // batch_size)
    # see https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX
    init_key, sample_key = jax.random.split(jax.random.PRNGKey(0))
    init_params = jnp.zeros(target.dim)
    @jax.jit
    def run_chain(key, state):
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target.log_prob,
                num_leapfrog_steps=3,
                step_size=1.),
            num_adaptation_steps=int(num_burnin_steps * 0.8))
        states, acceptance_probs = tfp.mcmc.sample_chain(
                      num_results=total_size,
                      num_burnin_steps=num_burnin_steps,
                      num_steps_between_results=10,
                      current_state=jnp.zeros((target.dim, )),
                      kernel=kernel,
                      trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                      seed=key,
                      parallel_iterations=batch_size
        )
        return states
    start_time = time.time()
    states = run_chain(sample_key, init_params)
    print(f"time to generate dataset: {(time.time() - start_time) / 60}  min")
    return states


def make_evaluator(agent, target: BNNEnergyFunction, test_set_size):
    test_set = setup_dataset(target, test_set_size)
    def evaluator(outer_batch_size, inner_batch_size, state):
        enn_sampler = make_posterior_log_prob(agent, target, state)
        def evaluate_single(rng_key, tau):
            key1, key2 = jax.random.split(rng_key)
            test_x, test_y = target.generate_data(key1, tau)
            log_q_y_base, log_q_y_ais = enn_sampler(key2, test_x, test_y)
            test_log_p = jnp.sum(target.target_prob(test_x, test_y), axis=0)
            # chex.assert_equal_shape((log_q_y_ais, log_q_y_base, test_log_p))
            return log_q_y_base, log_q_y_ais, test_log_p

        def evaluate(tau):
            key_batch = jax.random.split(state.key, 10)
            log_q_y_base, log_q_y_ais, test_log_p = jax.vmap(evaluate_single,
                                                             in_axes=(0, None))(key_batch, tau)
            expected_kl_base = jnp.mean(test_log_p - log_q_y_base)
            expected_kl_ais = jnp.mean(test_log_p - log_q_y_ais)
            return expected_kl_base, expected_kl_ais

        expected_kl_base_tau1, expected_kl_ais_tau1 = evaluate(1)
        expected_kl_base_tau10, expected_kl_ais_tau10 = evaluate(10)
        expected_kl_base_tau100, expected_kl_ais_tau100 = evaluate(100)


        test_set_log_prob = jnp.mean(agent.learnt_distribution.log_prob.apply(
            state.learnt_distribution_params, test_set))
        info = {
                "exp_kl_div_y_given_x_ais_tau1": expected_kl_ais_tau1,
                "exp_kl_div_y_given_x_base_tau1": expected_kl_base_tau1,
                "exp_kl_div_y_given_x_ais_tau10": expected_kl_ais_tau10,
                "exp_kl_div_y_given_x_base_tau10": expected_kl_base_tau10,
                "exp_kl_div_y_given_x_ais_tau100": expected_kl_ais_tau100,
                "exp_kl_div_y_given_x_base_tau100": expected_kl_base_tau100,
                "test_set_log_prob": test_set_log_prob}
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
    assert cfg.agent.transition_operator.type == "hmc_tfp"
    AIS_kwargs = {"transition_operator_type": cfg.agent.transition_operator.type,
        "additional_transition_operator_kwargs":
                      {
                       "n_inner_steps": cfg.agent.transition_operator.n_inner_steps}
                  }
    plotter = setup_plotter(target)
    if cfg.agent.agent_type == "fab":
        agent = AgentFAB(learnt_distribution=flow,
                         target_log_prob=target.log_prob,
                         n_intermediate_distributions=cfg.agent.n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         seed=cfg.training.seed,
                         optimizer=optimizer,
                         loss_type=cfg.agent.loss_type,
                         plotter=plotter,
                         logger=logger,
                         evaluator=None,
                         soften_ais_weights=cfg.agent.soften_ais_weights,
                         max_clip_frac=cfg.agent.max_clip_frac)
    elif cfg.agent.agent_type == "bbb":
        agent = AgentBBB(learnt_distribution=flow,
                         target_log_prob=target.log_prob,
                         n_intermediate_distributions=cfg.agent.n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         seed=cfg.training.seed,
                         optimizer=optimizer,
                         loss_type=cfg.agent.loss_type,
                         plotter=plotter,
                         logger=logger,
                         evaluator=None)
    elif cfg.agent.agent_type == "target_samples":
        agent = AgentTargetSamples(
            learnt_distribution=flow,
                         target_log_prob=target.log_prob,
                         n_intermediate_distributions=cfg.agent.n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         seed=cfg.training.seed,
                         optimizer=optimizer,
                         loss_type=cfg.agent.loss_type,
                         plotter=plotter,
                         logger=logger,
                         evaluator=None)
    else:
        raise NotImplementedError
    plotter(agent)
    evaluator = make_evaluator(agent, target, cfg.evaluation.test_set_size)  # we need the agent to make the evaluator
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
    return agent



@hydra.main(config_path="./config", config_name="bnn.yaml")
def run(cfg: DictConfig):
    agent = _run(cfg)



if __name__ == '__main__':
    run()
