import jax.random
from absl.testing import absltest
from distrax_flo import make_rational_quadratic_spline_dist_funcs
import jax.numpy as jnp
import chex
import haiku as hk




class UnitTests(absltest.TestCase):

    def test_rational_quadratic_spline(self):
        rng = hk.PRNGSequence(0)
        batch_size = 7
        x_ndim = 3
        flow_num_layers = 2
        quadratic_split_haiku_dist = make_rational_quadratic_spline_dist_funcs(
            x_ndim=x_ndim, flow_num_layers=flow_num_layers)
        x = jnp.zeros((batch_size, x_ndim))
        params = quadratic_split_haiku_dist.log_prob.init(next(rng), x)
        log_prob = quadratic_split_haiku_dist.log_prob.apply(params, x)
        chex.assert_shape(log_prob, (batch_size,))
        samples, log_probs = quadratic_split_haiku_dist.sample_and_log_prob.apply(params, next(rng),
                                                          sample_shape=(batch_size,))
        chex.assert_shape(log_probs, (batch_size,))
        chex.assert_shape(samples, (batch_size, x_ndim))
        samples = quadratic_split_haiku_dist.sample.apply(params, next(rng), sample_shape=(batch_size,))
        chex.assert_shape(samples, (batch_size, x_ndim))




if __name__ == '__main__':
  absltest.main()





