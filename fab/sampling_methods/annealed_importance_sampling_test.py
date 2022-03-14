from absl.testing import absltest
from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fab.target_distributions.many_well import ManyWellEnergy
import distrax
import jax.numpy as jnp
import haiku as hk
import chex



class Test_HMC(absltest.TestCase):
    x_ndim = 4
    flow_num_layers = 2

    # flow_haiku_dist = make_rational_quadratic_spline_dist_funcs(
    #     x_ndim=x_ndim, flow_num_layers=flow_num_layers)
    flow_haiku_dist = make_realnvp_dist_funcs(x_ndim=x_ndim, flow_num_layers=flow_num_layers)
    target_log_prob = ManyWellEnergy(dim=x_ndim).log_prob
    n_parallel_runs = 12
    n_intermediate_distributions = 2
    AIS = AnnealedImportanceSampler(
                 learnt_distribution=flow_haiku_dist,
                 target_log_prob=target_log_prob,
                 n_parallel_runs=n_parallel_runs,
                 n_intermediate_distributions=n_intermediate_distributions)
    rng = hk.PRNGSequence(0)
    x = jnp.zeros((n_parallel_runs, x_ndim))
    init_learnt_distribution_params = flow_haiku_dist.log_prob.init(next(rng), x)


    def test__run(self):
        x_new, log_w = self.AIS.run(next(self.rng), self.init_learnt_distribution_params)
        chex.assert_shape(x_new, (self.n_parallel_runs, self.x_ndim))
        chex.assert_shape(log_w, (self.n_parallel_runs,))


    def test__run_and_plot(self):
        n_parallel_runs = 1000
        n_intermediate_distributions = 100
        self.AIS = AnnealedImportanceSampler(
            learnt_distribution=self.flow_haiku_dist,
            target_log_prob=self.target_log_prob,
            n_parallel_runs=n_parallel_runs,
            n_intermediate_distributions=n_intermediate_distributions)
        x_new, log_w = self.AIS.run(next(self.rng), self.init_learnt_distribution_params)
        import matplotlib.pyplot as plt
        plt.plot(x_new[:, 0], x_new[:, 1], "o", alpha=0.3)
        plt.title("points from AIS")
        plt.show()

        # TODO: plot with resampling

    def test__multi_step_run(self):
        pass
        # TODO: test p_accept tuning, and gradient based tuning with nice visualisations




