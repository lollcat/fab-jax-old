from absl.testing import absltest
import distrax
from fab.target_distributions.many_well import ManyWellEnergy
import jax.numpy as jnp
import haiku as hk
import chex

from fab_vae.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler



class Test_HMC(absltest.TestCase):
    """See examples/visualise_ais.ipynb for further visualisation."""
    x_ndim = 4
    base_distribution = distrax.MultivariateNormalDiag(jnp.zeros((x_ndim,)),
                                                         jnp.ones((x_ndim,)))
    base_log_prob_fn = distrax.MultivariateNormalDiag(
        jnp.zeros((x_ndim, )), jnp.ones((x_ndim,))).log_prob
    target_log_prob = ManyWellEnergy(dim=x_ndim).log_prob
    n_parallel_runs = 12
    n_intermediate_distributions = 2
    AIS = AnnealedImportanceSampler(dim=x_ndim,
                                    n_intermediate_distributions=n_intermediate_distributions)
    rng = hk.PRNGSequence(0)
    x = jnp.zeros((n_parallel_runs, x_ndim))
    init_transition_operator_state = AIS.transition_operator_manager.get_init_state()


    def test__run(self):
        x_base, log_q_base = self.base_distribution.sample_and_log_prob(seed=next(self.rng),
                                                           sample_shape=(10,))
        x, log_w, transition_operator_state, aux_info = self.AIS.run(
            x_base, log_q_base,
            next(self.rng), self.init_transition_operator_state,
            self.base_log_prob_fn, self.target_log_prob
        )
        chex.assert_equal_shape((x, x_base))
        chex.assert_equal_shape((log_q_base, log_w))


    def test__run_and_plot(self):
        n_intermediate_distributions = 1000
        batch_size = 1000
        self.AIS = AnnealedImportanceSampler(dim=self.x_ndim,
                                        n_intermediate_distributions=n_intermediate_distributions)
        x_base, log_q_base = self.base_distribution.sample_and_log_prob(seed=next(self.rng),
                                                           sample_shape=(batch_size,))
        x, log_w, transition_operator_state, aux_info = self.AIS.run(
            x_base, log_q_base,
            next(self.rng), self.init_transition_operator_state,
            self.base_log_prob_fn, self.target_log_prob
        )
        import matplotlib.pyplot as plt
        plt.plot(x[:, 0], x[:, 1], "o", alpha=0.3)
        plt.title("points from AIS")
        plt.show()





