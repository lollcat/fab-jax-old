import chex
import jax.numpy as jnp
from absl.testing import absltest
import jax
import haiku as hk
from functools import partial

from fab.SamplingMethods.MCMC.HamiltoneanMonteCarlo import HamiltoneanMonteCarlo, \
    HMCStateGradientBased, HMCStatePAccept



class Test_HMC(absltest.TestCase):
    dim = 2
    n_intermediate_distributions = 2
    intermediate_target_log_prob_fn = lambda params, x, j: jnp.exp(-jnp.sum(x**2) + jnp.sum(x))
    batch_size = 5
    n_outer_steps = 2
    n_inner_steps = 4
    HMC_p_accept = HamiltoneanMonteCarlo(dim, n_intermediate_distributions,
                                  intermediate_target_log_prob_fn,
                     batch_size, step_tuning_method="p_accept", n_outer_steps=n_outer_steps,
                                         n_inner_steps=n_inner_steps,
                     initial_step_size= 1.0, lr=1e-3)
    HMC_grad_based = HamiltoneanMonteCarlo(dim, n_intermediate_distributions,
                                  intermediate_target_log_prob_fn,
                     batch_size, step_tuning_method="gradient_based", n_outer_steps=n_outer_steps,
                                           n_inner_steps=n_inner_steps,
                     initial_step_size=1.0, lr=1e-3)

    initial_state_p_accept = HMC_p_accept.get_init_state()
    initial_state_grad_based = HMC_grad_based.get_init_state()

    rng = hk.PRNGSequence(0)
    run_plot_test = False

    def test__init_state(self):
        initial_state = self.HMC_p_accept.get_init_state()
        assert isinstance(initial_state, HMCStatePAccept)
        initial_state = self.HMC_grad_based.get_init_state()
        assert isinstance(initial_state, HMCStateGradientBased)

    def test__get_step_size_param_for_dist(self):
        for i in range(self.n_intermediate_distributions):
            # p_accept HMC
            step_size_param = self.HMC_p_accept.get_step_size_param_for_dist(
                self.initial_state_p_accept.step_size_params, i)
            chex.assert_shape(step_size_param, (self.n_outer_steps, ))
            # grad_based HMC
            step_size_param = self.HMC_grad_based.get_step_size_param_for_dist(
                self.initial_state_grad_based.step_size_params, i)
            chex.assert_shape(step_size_param, (self.n_outer_steps, self.dim))


    def test__inner_most_loop_func(self):
        L = self.n_inner_steps
        i = 0
        learnt_distribution_params = None  # dummy
        outer_loop_iter = 0
        p = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        q = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        inner_loop_func = partial(self.HMC_p_accept._inner_most_loop_func,
                                  self.HMC_p_accept.grad_U,
                                       self.n_inner_steps, learnt_distribution_params,
                                  i)
        # p_accept HMC
        epsilon = self.HMC_p_accept.get_step_size_param_for_dist(
                self.initial_state_p_accept.step_size_params, 0)[outer_loop_iter]
        xs_inner = (jnp.arange(L), jnp.repeat(epsilon[None, ...], L, axis=0))
        (p, q), _ = jax.lax.scan(inner_loop_func, init=(p, q), xs=xs_inner)
        chex.assert_shape(p, (self.dim,))
        chex.assert_shape(q, (self.dim,))
        # grad_based_HMC
        # inner loop func is the same so we can re-use it
        epsilon = self.HMC_grad_based.get_step_size_param_for_dist(
                self.initial_state_grad_based.step_size_params, 0)[outer_loop_iter]
        xs_inner = (jnp.arange(L), jnp.repeat(epsilon[None, ...], L, axis=0))
        (p, q), _ = jax.lax.scan(inner_loop_func, init=(p, q), xs=xs_inner)
        chex.assert_shape(p, (self.dim,))
        chex.assert_shape(q, (self.dim,))

    def test__outer_loop_func(self):
        L = self.n_inner_steps
        i = 0
        key = next(self.rng)
        current_q = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        learnt_distribution_params = None  # dummy
        inner_loop_func = partial(self.HMC_p_accept._inner_most_loop_func,
                                  self.HMC_p_accept.grad_U,
                                       self.n_inner_steps, learnt_distribution_params,
                                  i)
        outer_loop_func = partial(self.HMC_p_accept._outer_loop_func, self.HMC_p_accept.U,
                                  self.HMC_p_accept.grad_U,
                                  inner_loop_func,
                                  self.n_inner_steps, learnt_distribution_params, i)

        # p_accept HMC
        step_size = self.HMC_p_accept.get_step_size_param_for_dist(
            self.initial_state_p_accept.step_size_params, i)
        xs = {"rng_key": jax.random.split(key, self.n_outer_steps),
              "step_size": step_size}
        q_out, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            jax.lax.scan(outer_loop_func, init=(current_q), xs=xs)
        chex.assert_equal_shape([q_out, current_q])
        chex.assert_shape(current_q_per_outer_loop, (self.n_outer_steps, *current_q.shape))
        assert not jnp.any(q_out == current_q)
        # grad based HMC
        step_size = self.HMC_grad_based.get_step_size_param_for_dist(
            self.initial_state_grad_based.step_size_params, i)
        xs = {"rng_key": jax.random.split(key, self.n_outer_steps),
              "step_size": step_size}
        q_out, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            jax.lax.scan(outer_loop_func, init=(current_q), xs=xs)
        chex.assert_equal_shape([q_out, current_q])
        chex.assert_shape(current_q_per_outer_loop, (self.n_outer_steps, *current_q.shape))
        assert not jnp.any(q_out == current_q)


    def test__run(self):
        key = next(self.rng)
        keys = jax.random.split(key, self.batch_size)
        learnt_distribution_param = None
        x = jax.random.normal(shape=(self.batch_size, self.dim), key=next(self.rng))
        i = 0
        # p_accept
        step_size = self.HMC_p_accept.get_step_size_param_for_dist(
            self.initial_state_p_accept.step_size_params, 0)
        x_out, (x_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            self.HMC_p_accept.run(keys, learnt_distribution_param, step_size, x, i)
        chex.assert_equal_shape([x_out, x])
        chex.assert_shape(x_per_outer_loop, (self.batch_size, self.n_outer_steps, self.dim))
        chex.assert_shape(acceptance_probabilities_per_outer_loop, (self.batch_size,
                                                                    self.n_outer_steps))
        # gradient based
        step_size = self.HMC_grad_based.get_step_size_param_for_dist(
            self.initial_state_grad_based.step_size_params, i)
        x_out, (x_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
        self.HMC_grad_based.run(keys, learnt_distribution_param, step_size, x, i)
        chex.assert_equal_shape([x_out, x])
        chex.assert_shape(x_per_outer_loop, (self.batch_size, self.n_outer_steps, self.dim))
        chex.assert_shape(acceptance_probabilities_per_outer_loop, (self.batch_size,
                                                                    self.n_outer_steps))


    def test__run_and_los(self):




    def test__with_plot(self):
        if self.run_plot_test:
            pass # TODO
        else:
            pass


    # TODO: eyeball the algorithm to double check all the steps are correct






if __name__ == '__main__':
  absltest.main()
