import jax.random
from absl.testing import absltest
import distrax
from functools import partial
from fabjax.target_distributions.many_well import ManyWellEnergy
import jax.numpy as jnp
import haiku as hk
import chex

from fabjax.sampling_methods.mcmc.unfinished.tfp_ais import AnnealedImportanceSamplerTfp as AnnealedImportanceSampler



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
    init_transition_operator_state = AIS.get_init_state()


    def test__run(self):
        jit = True
        x_base, log_q_base = self.base_distribution.sample_and_log_prob(seed=next(self.rng),
                                                           sample_shape=(10,))
        if jit:
            run = partial(jax.jit, static_argnums=(4, 5))(self.AIS.run)
        else:
            run = self.AIS.run
        x, log_w, transition_operator_state, aux_info = run(
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

        # test that log weights thin the distribution appropriately
        indices = jax.random.choice(jax.random.PRNGKey(0), log_w.shape[0],
                                    p=jax.nn.softmax(log_w), shape=(1000,),
                                    replace=True)
        plt.plot(x[indices, 0], x[indices, 1], "o", alpha=0.3)
        plt.title("points from AIS after resampling")
        plt.show()