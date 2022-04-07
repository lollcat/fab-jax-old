import jax.random
from absl.testing import absltest
import optax
import matplotlib.pyplot as plt
from functools import partial

from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fab.target_distributions.many_well import ManyWellEnergy
from fab.agent.fab_agent import AgentFAB
from fab.utils.plotting import plot_history, plot_marginal_pair, plot_contours_2D


def plotter(fab_agent, log_prob_2D):
    batch_size = 100

    @jax.jit
    def get_info(state):
        base_log_prob = fab_agent.get_base_log_prob(state.learnt_distribution_params)
        target_log_prob = fab_agent.get_target_log_prob(state.learnt_distribution_params)
        x_base, log_q_x_base = fab_agent.learnt_distribution.sample_and_log_prob.apply(
            state.learnt_distribution_params, rng=state.key,
            sample_shape=(batch_size,))
        x_ais_loss, _, _, _ = \
            fab_agent.annealed_importance_sampler.run(
                x_base, log_q_x_base, state.key,
                state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=target_log_prob
            )
        x_ais_target, _, _, _ = \
            fab_agent.annealed_importance_sampler.run(
                x_base, log_q_x_base, state.key,
                state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=fab_agent.target_log_prob
            )
        return x_base, x_ais_loss, x_ais_target

    x_base, x_ais_loss, x_ais_target = get_info(fab_agent.state)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    plot_contours_2D(log_prob_2D, ax=axs[0], bound=3, levels=20)
    plot_marginal_pair(x_base, ax=axs[0])
    axs[0].set_title("base samples")
    plot_contours_2D(log_prob_2D, ax=axs[1], bound=3, levels=20)
    plot_marginal_pair(x_ais_loss, ax=axs[1])
    axs[1].set_title("p^2 / q samples")
    plot_contours_2D(log_prob_2D, ax=axs[2], bound=3, levels=20)
    plot_marginal_pair(x_ais_target, ax=axs[2])
    axs[2].set_title("ais target samples")
    plt.show()
    return [fig]


class Test_AgentFAB(absltest.TestCase):
    # jax.config.update("jax_enable_x64", True)
    dim = 4
    flow_num_layers = 5
    mlp_hidden_size_per_x_dim = 1
    real_nvp_flo = make_realnvp_dist_funcs(dim, flow_num_layers,
                                           mlp_hidden_size_per_x_dim=mlp_hidden_size_per_x_dim)
    target = ManyWellEnergy(dim=dim)
    target_log_prob = target.log_prob
    log_prob_2D = target.log_prob_2D
    batch_size = 64
    n_iter = int(5e3)
    loss_type = "alpha_2_div"  # "forward_kl"  "alpha_2_div"
    style = "proptoloss"  # "vanilla"  "proptoloss"
    n_intermediate_distributions: int = 4
    soften_ais_weights = False
    use_reparam_loss = False
    max_grad_norm = None
    lr = 1e-3
    n_plots = 5
    n_evals = 10
    eval_batch_size = batch_size

    # AIS_kwargs = {"additional_transition_operator_kwargs": {"step_tuning_method": "p_accept"}}
    AIS_kwargs = {"transition_operator_type": "hmc_tfp"}  #  "hmc_tfp", "nuts_tfp"

    if max_grad_norm is None:
        optimizer = optax.chain(optax.zero_nans(), optax.adam(lr))
    else:
        optimizer = optax.chain(optax.zero_nans(),
                                optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    plotter = partial(plotter, log_prob_2D=log_prob_2D)

    fab_agent = AgentFAB(learnt_distribution=real_nvp_flo,
                         target_log_prob=target_log_prob,
                         n_intermediate_distributions=n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         optimizer=optimizer,
                         loss_type=loss_type,
                         style=style,
                         plotter=plotter,
                         add_reverse_kl_loss=use_reparam_loss,
                         soften_ais_weights=soften_ais_weights,
                         )

    def test_fab_agent(self):
        # self.plotter(self.fab_agent)
        self.fab_agent.run(n_iter=self.n_iter, batch_size=self.batch_size, n_plots=self.n_plots,
                           save=False, n_checkpoints=None, n_evals=self.n_evals,
                           eval_batch_size=self.eval_batch_size)
        plot_history(self.fab_agent.logger.history)
        plt.show()


