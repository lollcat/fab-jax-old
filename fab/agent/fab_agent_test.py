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
    fig, ax = plt.subplots()
    plot_contours_2D(log_prob_2D, ax=ax, bound=3, levels=20)
    samples = fab_agent.learnt_distribution.sample.apply(
        fab_agent.learnt_distribution_params,
        jax.random.PRNGKey(0), (500,))
    plot_marginal_pair(samples, ax=ax)
    plt.show()


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
    batch_size = 128
    n_iter = int(1e4)
    loss_type = "alpha_2_div"  # "forward_kl"  "alpha_2_div"
    n_intermediate_distributions: int = 2
    max_grad_norm = 1.0
    lr = 5e-4
    n_plots = 10

    AIS_kwargs = {"additional_transition_operator_kwargs": {"step_tuning_method": "p_accept"}}

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))
    plotter = partial(plotter, log_prob_2D=log_prob_2D)

    fab_agent = AgentFAB(learnt_distribution=real_nvp_flo,
                         target_log_prob=target_log_prob,
                         batch_size=batch_size,
                         n_intermediate_distributions=n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         optimizer=optimizer,
                         loss_type=loss_type,
                         plotter=plotter)

    def test_fab_agent(self):
        self.fab_agent.run(n_iter=self.n_iter, n_plots=self.n_plots)
        plot_history(self.fab_agent.logger.history)
        plt.show()


