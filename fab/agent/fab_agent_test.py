import jax.random
from absl.testing import absltest
import optax
import matplotlib.pyplot as plt

from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fab.target_distributions.many_well import DoubleWellEnergy
from fab.agent.fab_agent import AgentFAB
from fab.utils.plotting import plot_history, plot_marginal_pair, plot_contours_2D



class Test_AgentFAB(absltest.TestCase):
    jax.config.update("jax_enable_x64", True)
    dim = 2
    flow_num_layers = 5
    mlp_hidden_size_per_x_dim = 5
    real_nvp_flo = make_realnvp_dist_funcs(dim, flow_num_layers,
                                           mlp_hidden_size_per_x_dim=mlp_hidden_size_per_x_dim)
    target_log_prob = DoubleWellEnergy(dim=dim).log_prob
    batch_size = 64
    n_iter = int(2e3)
    n_intermediate_distributions: int = 2
    max_grad_norm = 1.0
    lr = 1e-3

    AIS_kwargs = {"additional_transition_operator_kwargs": {"step_tuning_method": "p_accept"}}

    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))

    fab_agent = AgentFAB(learnt_distribution=real_nvp_flo,
                         target_log_prob=target_log_prob,
                         batch_size=batch_size, n_iter=n_iter,
                         n_intermediate_distributions=n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs,
                         optimizer=optimizer)

    def test_fab_agent(self):
        self.fab_agent.run()
        plot_history(self.fab_agent.history)
        plt.show()

        fig, ax = plt.subplots()
        plot_contours_2D(self.fab_agent.target_log_prob, ax=ax, bound=3, levels=20)
        samples = self.fab_agent.learnt_distribution.sample.apply(
            self.fab_agent.learnt_distribution_params,
            jax.random.PRNGKey(0), (500,))
        plot_marginal_pair(samples, ax=ax)
        plt.show()
