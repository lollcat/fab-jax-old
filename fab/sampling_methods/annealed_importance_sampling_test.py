from absl.testing import absltest
from fab.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab.learnt_distributions.distrax_flo import make_rational_quadratic_spline_dist_funcs
import distrax
import jax.numpy as jnp
import haiku as hk
import chex

def make_target_log_prob(dim):
    dist = distrax.MultivariateNormalDiag(jnp.zeros((dim, )), jnp.ones((dim, )))
    log_prob_fn = dist.log_prob
    return log_prob_fn

class Test_HMC(absltest.TestCase):
    x_ndim = 3
    flow_num_layers = 2
    quadratic_split_haiku_dist = make_rational_quadratic_spline_dist_funcs(
        x_ndim=x_ndim, flow_num_layers=flow_num_layers)
    target_log_prob = make_target_log_prob(x_ndim)
    n_parallel_runs = 12
    n_intermediate_distributions = 3
    AIS = AnnealedImportanceSampler(
                 learnt_distribution = quadratic_split_haiku_dist,
                 target_log_prob = target_log_prob,
                 n_parallel_runs = n_parallel_runs,
                 n_intermediate_distributions = n_intermediate_distributions)
    rng = hk.PRNGSequence(0)
    x = jnp.zeros((n_parallel_runs, x_ndim))
    init_learnt_distribution_params = quadratic_split_haiku_dist.log_prob.init(next(rng), x)


    def test__run(self):
        x_new, log_w = self.AIS.run(next(self.rng), self.init_learnt_distribution_params)
        chex.assert_shape(x_new, (self.n_parallel_runs, self.x_ndim))
        chex.assert_shape(log_w, (self.n_parallel_runs,))

