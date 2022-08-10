from functools import partial

import distrax
import tensorflow_probability.substrates.jax as tfp
import numpy as np

from flows_for_atomic_solids.experiments.train import _num_particles
from flows_for_atomic_solids.experiments import lennard_jones_config
from flows_for_atomic_solids.experiments import monatomic_water_config


from fabjax.learnt_distributions.model_to_haiku_dist import model_to_haiku_dist
from fabjax.utils.plotting import plot_history

import os
import pathlib
import chex
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
import jax
import optax
import matplotlib.pyplot as plt

from fabjax.utils.logging import PandasLogger, WandbLogger, Logger, ListLogger
from fabjax.agent.fab_agent_prioritised import PrioritisedAgentFAB
from fabjax.utils.prioritised_replay_buffer import PrioritisedReplayBuffer

SYSTEMS = ['mw_cubic_8', 'mw_cubic_64', 'mw_cubic_216', 'mw_cubic_512',
                   'mw_hex_64', 'mw_hex_216', 'mw_hex_512',
                   'lj_32', 'lj_256', 'lj_500',
                   ]


def setup_logger(cfg: DictConfig, save_path: str) -> Logger:
    if hasattr(cfg.logger, "pandas_logger"):
        logger = PandasLogger(save=True,
                              save_path=save_path + "logging_hist.csv",
                              save_period=cfg.logger.pandas_logger.save_period)
    elif hasattr(cfg.logger, "wandb"):
        logger = WandbLogger(**cfg.logger.wandb, config=dict(cfg))
    elif hasattr(cfg.logger, "list_logger"):
        logger = ListLogger(save=False)
    else:
        raise Exception("No logger specified, try adding the wandb or "
                        "pandas logger to the config file.")
    return logger


def create_model(config, cfg: DictConfig) -> distrax.Distribution:
    state = config.state
    if not cfg.flow.use_default_params:
        # Overwrite params from atomic solids paper.
        config.model["kwargs"]["bijector"]["kwargs"].update(num_bins=cfg.flow.num_bins,
                                                            num_layers=cfg.flow.num_layers)
        config.model["kwargs"]["bijector"]["kwargs"]["conditioner"]["kwargs"].update(
            embedding_size=cfg.flow.embedding_size)
    model = config.model['constructor'](
        num_particles=state.num_particles,
        lower=state.lower,
        upper=state.upper,
        **config.model['kwargs'])
    event_shape = model.event_shape
    reshape_bijector = tfp.bijectors.Reshape(event_shape_out=(np.prod(event_shape),),
                                             event_shape_in=event_shape)
    model = distrax.Transformed(model, reshape_bijector)
    return model


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


    system = 'mw_cubic_8'  # SYSTEMS[0]


    if system.startswith('lj'):
        config = lennard_jones_config.get_config(_num_particles(system))
    elif system.startswith('mw_cubic'):
        config = monatomic_water_config.get_config(_num_particles(system), 'cubic')
    elif system.startswith('mw_hex'):
        config = monatomic_water_config.get_config(_num_particles(system), 'hex')
    else:
        raise KeyError(system)

    energy_fn_train = config.train_energy.constructor(
        **config.train_energy.kwargs)
    energy_fn_test = config.test_energy.constructor(**config.test_energy.kwargs)

    event_shape_before_reshape = (config.state.num_particles, 3)

    def energy_fn(x: chex.Array) -> chex.Array:
        if len(x.shape) == 2:
            x = x.reshape((-1, *event_shape_before_reshape))
        elif len(x.shape) == 2:
            x = x.reshape(event_shape_before_reshape)
        else:
            raise Exception
        return energy_fn_train(x)

    dim = int(np.prod(event_shape_before_reshape))
    flow = model_to_haiku_dist(partial(create_model, config, cfg), dim)
    # if True:
        # params = flow.sample.init(jax.random.PRNGKey(0))
        # x, log_prob = flow.sample_and_log_prob.apply(params, jax.random.PRNGKey(0), (3,))
        # # data = jax.random.normal(key=jax.random.PRNGKey(0), shape=(2, dim))
        # # flow.log_prob.apply(params, jnp.zeros(2, dim))
        # Below code give nan's as samples are outside of the base distribution.
        # log_prob = flow.log_prob.apply(params, x + jax.random.normal(jax.random.PRNGKey(0),
        #                                                              x.shape)*1.0)
    optimizer = optax.adam(cfg.training.lr)

    # AIS_kwargs = {"transition_operator_type": "hmc_tfp",
    #               "additional_transition_operator_kwargs":
    #                   {"init_step_size": 0.2}}

    AIS_kwargs = {"transition_operator_type": "metropolis_tfp",
                  "additional_transition_operator_kwargs":
    {"init_step_size": 0.2}}

    buffer = PrioritisedReplayBuffer(
        dim=dim,
        max_length=cfg.buffer.maximum_buffer_length,
        min_sample_length=cfg.buffer.min_buffer_length)

    agent = PrioritisedAgentFAB(learnt_distribution=flow,
                                target_log_prob=energy_fn,
                                n_intermediate_distributions=cfg.fab.n_intermediate_distributions,
                                replay_buffer=buffer,
                                max_w_adjust=cfg.buffer.max_w_adjust,
                                n_buffer_updates_per_forward=cfg.buffer.n_batches_buffer_sampling,
                                AIS_kwargs=AIS_kwargs,
                                seed=cfg.training.seed,
                                optimizer=optimizer,
                                plotter=None,
                                logger=logger,
                                evaluator=None)

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
    if hasattr(cfg.logger, "list_logger"):
        plot_history(agent.logger.history)
        plt.show()


@hydra.main(config_path="./config", config_name="atomic_solids.yaml")
def run(cfg: DictConfig):
    _run(cfg)


if __name__ == '__main__':
    run()





