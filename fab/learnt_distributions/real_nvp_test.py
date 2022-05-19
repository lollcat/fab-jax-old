import jax.random
from absl.testing import absltest
from real_nvp import make_realnvp_dist_funcs
import jax.numpy as jnp
import chex
import haiku as hk




class UnitTests(absltest.TestCase):

    def test_real_nvp(self):
        rng = hk.PRNGSequence(0)
        batch_size = 7
        x_n_elements = 16
        flow_num_layers = 4
        realNVP_haiku_dist = make_realnvp_dist_funcs(
            x_ndim=x_n_elements, flow_num_layers=flow_num_layers, act_norm=True)
        x = jnp.zeros((batch_size, x_n_elements))
        params = realNVP_haiku_dist.log_prob.init(next(rng), x)
        log_prob = realNVP_haiku_dist.log_prob.apply(params, x)
        chex.assert_shape(log_prob, (batch_size,))
        key = next(rng)
        print(key)
        samples, log_probs = realNVP_haiku_dist.sample_and_log_prob.apply(params, key,
                                                          sample_shape=(batch_size,))
        key1 = next(rng)
        print(key1)
        samples_, _ = realNVP_haiku_dist.sample_and_log_prob.apply(params, key1,
                                                                          sample_shape=(
                                                                          batch_size,))
        print(key)
        samples__, _ = realNVP_haiku_dist.sample_and_log_prob.apply(params, key,
                                                                   sample_shape=(
                                                                       batch_size,))
        # check randomness
        assert (samples_ != samples).all()
        assert (samples__ == samples).all()
        chex.assert_shape(log_probs, (batch_size,))
        chex.assert_shape(samples, (batch_size, x_n_elements))
        samples = realNVP_haiku_dist.sample.apply(params, next(rng), sample_shape=(batch_size,))
        chex.assert_shape(samples, (batch_size, x_n_elements))




if __name__ == '__main__':
  absltest.main()





