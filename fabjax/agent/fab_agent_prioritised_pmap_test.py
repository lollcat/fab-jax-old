import jax.random
from absl.testing import absltest
import optax
import matplotlib.pyplot as plt
from functools import partial

from fabjax.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fabjax.target_distributions.many_well import ManyWellEnergy, setup_manywell_evaluator
from fabjax.agent.fab_agent_prioritised_pmap import PrioritisedAgentFAB, State
from fabjax.utils.plotting import plot_history, plot_marginal_pair, plot_contours_2D
from fabjax.utils.prioritised_replay_buffer import PrioritisedReplayBuffer


def plotter(fab_agent: PrioritisedAgentFAB, log_prob_2D):
    batch_size = 100

    @jax.jit
    def get_info(state: State):
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
        buffer_samples = fab_agent.replay_buffer.sample(state.buffer_state, state.key, batch_size)
        return x_base, x_ais_target, buffer_samples[0]

    x_base, x_ais_target, x_buffer = get_info(fab_agent.state_first_device)
    n_plots = 3
    fig, axs = plt.subplots(1, 3, figsize=(5*n_plots, 4))
    plot_contours_2D(log_prob_2D, ax=axs[0], bound=3, levels=20)
    plot_marginal_pair(x_base, ax=axs[0])
    axs[0].set_title("base samples")
    plot_contours_2D(log_prob_2D, ax=axs[1], bound=3, levels=20)
    plot_marginal_pair(x_ais_target, ax=axs[1])
    axs[1].set_title("ais target samples")
    plot_contours_2D(log_prob_2D, ax=axs[2], bound=3, levels=20)
    plot_marginal_pair(x_buffer, ax=axs[2])
    axs[2].set_title("buffer samples")
    plt.show()
    return [fig]


class Test_AgentFAB(absltest.TestCase):
    jax.config.update("jax_enable_x64", True)
    dim = 4
    flow_num_layers = 5
    mlp_hidden_size_per_x_dim = 10
    real_nvp_flo = make_realnvp_dist_funcs(dim, flow_num_layers,
                                           mlp_hidden_size_per_x_dim=mlp_hidden_size_per_x_dim,
                                           act_norm=False,
                                           layer_norm=False,
                                           lu_layer=False)
    target = ManyWellEnergy(dim=dim)
    evaluator = setup_manywell_evaluator(target, real_nvp_flo)
    target_log_prob = target.log_prob
    log_prob_2D = target.log_prob_2D
    batch_size = 64
    n_iter = int(1e3)
    loss_type = "alpha_2_div"  # "forward_kl"  "alpha_2_div"
    style = "vanilla"  # "vanilla"  "proptoloss"
    n_intermediate_distributions: int = 4
    max_grad_norm = 10.0
    lr = 1e-4
    n_plots = 10
    n_evals = 4
    n_buffer_updates_per_forward = 8
    eval_batch_size = batch_size

    base_optax_transform = optax.adam(lr)  # optax.adabelief(lr)
    # base_optax_transform = optax.sgd(lr)

    buffer = PrioritisedReplayBuffer(dim=dim,
                          max_length=batch_size*n_buffer_updates_per_forward*100,
                          min_sample_length=batch_size*n_buffer_updates_per_forward*10)

    # buffer = None
    # AIS_kwargs = {"additional_transition_operator_kwargs": {"step_tuning_method": "p_accept"}}
    AIS_kwargs = {"transition_operator_type": "hmc_tfp",
                  "additional_transition_operator_kwargs":
                      {"init_step_size": 0.2}}

    if max_grad_norm is None:
        optimizer = optax.chain(optax.zero_nans(), base_optax_transform)
    else:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip(100.0),
                                optax.clip_by_global_norm(max_grad_norm),
                                base_optax_transform)
    plotter = partial(plotter, log_prob_2D=log_prob_2D)

    fab_agent = PrioritisedAgentFAB(learnt_distribution=real_nvp_flo,
                                    target_log_prob=target_log_prob,
                                    n_intermediate_distributions=n_intermediate_distributions,
                                    replay_buffer=buffer,
                                    n_buffer_updates_per_forward=n_buffer_updates_per_forward,
                                    AIS_kwargs=AIS_kwargs,
                                    optimizer=optimizer,
                                    plotter=plotter,
                                    max_w_adjust=10.0,
                                    evaluator=evaluator
                                    )

    def test_fab_agent(self):
        # self.plotter(self.fab_agent)
        self.fab_agent.run(n_iter=self.n_iter, batch_size=self.batch_size, n_plots=self.n_plots,
                           save=False, n_checkpoints=None, n_evals=self.n_evals,
                           eval_batch_size=self.eval_batch_size)
        plot_history(self.fab_agent.logger.history)
        plt.show()
