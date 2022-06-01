from functools import partial

import chex
import distrax
import optax
import tensorflow_probability.substrates.jax as tfp
import numpy as np
import matplotlib.pyplot as plt

from flows_for_atomic_solids.experiments.train import _num_particles
from flows_for_atomic_solids.experiments import lennard_jones_config
from flows_for_atomic_solids.experiments import monatomic_water_config
from flows_for_atomic_solids.experiments import utils
from flows_for_atomic_solids.utils import observable_utils as obs_utils


from fab.agent.fab_agent_prioritised import PrioritisedAgentFAB
from fab.utils.prioritised_replay_buffer import PrioritisedReplayBuffer
from fab.learnt_distributions.model_to_haiku_dist import model_to_haiku_dist
from fab.utils.plotting import plot_history

SYSTEMS = ['mw_cubic_8', 'mw_cubic_64', 'mw_cubic_216', 'mw_cubic_512',
                   'mw_hex_64', 'mw_hex_216', 'mw_hex_512',
                   'lj_32', 'lj_256', 'lj_500',
                   ]


def create_model(state) -> distrax.Distribution:
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

if __name__ == '__main__':
    system = 'mw_cubic_8'  # SYSTEMS[0]
    batch_size = 2
    n_iter = int(1e3)
    loss_type = "alpha_2_div"  # "forward_kl"  "alpha_2_div"
    style = "vanilla"  # "vanilla"  "proptoloss"
    n_intermediate_distributions: int = 1
    max_grad_norm = 10.0
    lr = 1e-4
    n_plots = 10
    n_evals = 4
    n_buffer_updates_per_forward = 8
    eval_batch_size = batch_size

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
    flow = model_to_haiku_dist(partial(create_model, config.state), dim)

    optimizer = optax.adam(1e-4)

    AIS_kwargs = {"transition_operator_type": "hmc_tfp",
                  "additional_transition_operator_kwargs":
                      {"init_step_size": 0.2}}


    buffer = PrioritisedReplayBuffer(dim=dim,
                                     max_length=batch_size * n_buffer_updates_per_forward*10,
                                     min_sample_length=batch_size * n_buffer_updates_per_forward)

    fab_agent = PrioritisedAgentFAB(learnt_distribution=flow,
                                    target_log_prob=energy_fn,
                                    n_intermediate_distributions=n_intermediate_distributions,
                                    replay_buffer=buffer,
                                    n_buffer_updates_per_forward=n_buffer_updates_per_forward,
                                    AIS_kwargs=AIS_kwargs,
                                    optimizer=optimizer,
                                    max_w_adjust=10.0,
                                    )

    fab_agent.run(n_iter=n_iter, batch_size=batch_size, save=False)
    plot_history(fab_agent.logger.history)
    plt.show()



