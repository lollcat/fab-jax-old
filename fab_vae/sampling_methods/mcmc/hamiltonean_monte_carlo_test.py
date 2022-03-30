import chex
import jax.numpy as jnp
from absl.testing import absltest
import jax
import haiku as hk
import numpy as np
from functools import partial
import distrax

from fab_vae.sampling_methods.mcmc.hamiltonean_monte_carlo import HamiltoneanMonteCarlo, \
    HMCStateGradientBased, HMCStatePAccept
from fab.utils.plotting import plot_history
from fab.utils.logging import ListLogger



class Test_HMC(absltest.TestCase):
    dim = 2
    n_intermediate_distributions = 2
    target_distribution = distrax.MultivariateNormalDiag(jnp.zeros((dim,)),
                                                         jnp.ones((dim,)))
    intermediate_target_log_prob_fn = distrax.MultivariateNormalDiag(
        jnp.zeros((dim, )), jnp.ones((dim,))).log_prob
    batch_size = 30
    n_outer_steps = 3
    n_inner_steps = 4
    HMC_p_accept = HamiltoneanMonteCarlo(dim, n_intermediate_distributions,
                     step_tuning_method="p_accept", n_outer_steps=n_outer_steps,
                                         n_inner_steps=n_inner_steps,
                     initial_step_size= 1.0, lr=1e-3)
    HMC_grad_based = HamiltoneanMonteCarlo(dim, n_intermediate_distributions,
                     step_tuning_method="gradient_based", n_outer_steps=n_outer_steps,
                                           n_inner_steps=n_inner_steps,
                     initial_step_size=1.0, lr=1e-3)

    initial_state_p_accept = HMC_p_accept.get_init_state()
    initial_state_grad_based = HMC_grad_based.get_init_state()

    rng = hk.PRNGSequence(0)
    run_plot_test = False

    def U(self, q):
        return - self.intermediate_target_log_prob_fn(q)

    def grad_U(self, q):
        return jnp.clip(jax.grad(self.U)(q), a_min=-100, a_max=100)

    def test__init_state(self):
        initial_state = self.HMC_p_accept.get_init_state()
        assert isinstance(initial_state, HMCStatePAccept)
        initial_state = self.HMC_grad_based.get_init_state()
        assert isinstance(initial_state, HMCStateGradientBased)

    def test__get_step_size_param_for_dist(self) -> jnp.ndarray:
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
        outer_loop_iter = 0
        p = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        q = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        q_in = q

        inner_loop_func = partial(self.HMC_p_accept._inner_most_loop_func,
                                  self.grad_U,
                                  self.n_inner_steps)
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
        assert not jnp.any(q == q_in)

    def test__outer_loop_func(self):
        i = 0
        key = next(self.rng)
        current_q = jax.random.normal(shape=(self.dim,), key=next(self.rng))
        inner_loop_func = partial(self.HMC_p_accept._inner_most_loop_func,
                                  self.grad_U,
                                       self.n_inner_steps)
        outer_loop_func = partial(self.HMC_p_accept._outer_loop_func, self.U,
                                  self.grad_U,
                                  inner_loop_func,
                                  self.n_inner_steps)
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
        x = jax.random.normal(shape=(self.batch_size, self.dim), key=next(self.rng))
        i = 0
        # p_accept
        step_size = self.HMC_p_accept.get_step_size_param_for_dist(
            self.initial_state_p_accept.step_size_params, 0)
        x_out, (x_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            self.HMC_p_accept._run(keys, step_size, x, self.U)
        chex.assert_equal_shape([x_out, x])
        chex.assert_shape(x_per_outer_loop, (self.batch_size, self.n_outer_steps, self.dim))
        chex.assert_shape(acceptance_probabilities_per_outer_loop, (self.batch_size,
                                                                    self.n_outer_steps))
        # gradient based
        step_size = self.HMC_grad_based.get_step_size_param_for_dist(
            self.initial_state_grad_based.step_size_params, i)
        x_out, (x_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
        self.HMC_grad_based._run(keys, step_size, x, self.U)
        chex.assert_equal_shape([x_out, x])
        chex.assert_shape(x_per_outer_loop, (self.batch_size, self.n_outer_steps, self.dim))
        chex.assert_shape(acceptance_probabilities_per_outer_loop, (self.batch_size,
                                                                    self.n_outer_steps))


    def test__run_and_loss(self):
        key = next(self.rng)
        x = jax.random.normal(shape=(self.batch_size, self.dim), key=next(self.rng))
        i = 0
        # p_accept
        transition_operator_step_sizes = self.initial_state_p_accept.step_size_params
        transition_operator_additional_state_info = self.initial_state_p_accept.no_grad_params
        x_batch_final, info = self.HMC_p_accept.run_and_loss(key,
                                                             transition_operator_step_sizes,
                                                             transition_operator_additional_state_info,
                                                             x, i, self.U)
        chex.assert_equal_shape([x, x_batch_final])
        assert (x != x_batch_final).any()  # confirm at least one movement
        chex.assert_shape(info.expected_scaled_mean_distance_per_outer_loop, (self.batch_size,
                                                                              self.n_outer_steps))
        chex.assert_shape(info.average_acceptance_probabilities_per_outer_loop,
                          (self.n_outer_steps,))
        chex.assert_shape(info.mean_delta_per_element, (self.n_outer_steps, self.dim))
        chex.assert_shape(info.std_per_element, (self.dim,))

        # gradient_based
        transition_operator_step_sizes = self.initial_state_grad_based.step_size_params
        transition_operator_additional_state_info = self.initial_state_grad_based.no_grad_params
        loss, (x_batch_final, info) = self.HMC_grad_based.run_and_loss(key,
                                                             transition_operator_step_sizes,
                                                             transition_operator_additional_state_info,
                                                             x, i, self.U)
        chex.assert_shape(loss, ())
        assert jnp.isfinite(loss)
        chex.assert_equal_shape([x, x_batch_final])
        assert (x != x_batch_final).any()  # confirm at least one movement
        chex.assert_shape(info.expected_scaled_mean_distance_per_outer_loop, (self.batch_size,
                                                                              self.n_outer_steps))
        chex.assert_shape(info.average_acceptance_probabilities_per_outer_loop,
                          (self.n_outer_steps,))
        chex.assert_shape(info.mean_delta_per_element, (self.n_outer_steps, self.dim))
        chex.assert_shape(info.std_per_element, (self.dim,))


    def test__vectorised_run_with_updates(self):
        key = next(self.rng)
        learnt_distribution_param = None
        x = jax.random.normal(shape=(self.batch_size, self.dim), key=next(self.rng))
        i = 0
        # p_accept
        transition_operator_state = self.initial_state_p_accept
        x_out, new_transition_operator_state, interesting_info = \
            self.HMC_p_accept.run(
                key, transition_operator_state, x, i, self.intermediate_target_log_prob_fn)
        assert isinstance(new_transition_operator_state, HMCStatePAccept)
        assert isinstance(interesting_info, dict)
        chex.assert_equal_shape([x, x_out])
        # gradient based
        transition_operator_state = self.initial_state_grad_based
        x_out, new_transition_operator_state, interesting_info = \
            self.HMC_grad_based.run(
                key, transition_operator_state, x, i, self.intermediate_target_log_prob_fn)
        assert isinstance(new_transition_operator_state, HMCStateGradientBased)
        assert isinstance(interesting_info, dict)
        chex.assert_equal_shape([x, x_out])



    def test__with_plot(self):
        if not self.run_plot_test:
            import matplotlib.pyplot as plt
            key = next(self.rng)
            target_samples = np.array(self.target_distribution.sample(seed=key, sample_shape=(
                self.batch_size,)))
            x_init = jax.random.normal(shape=(self.batch_size, self.dim), key=next(self.rng))*0.1
            i = 0
            n_loops = 10
            for name, transition_operator_state, HMC_class in [
                ("p_accept", self.initial_state_p_accept, self.HMC_p_accept),
                ("gradient_based", self.initial_state_grad_based, self.HMC_grad_based)]:
                jitted_transition = jax.jit(HMC_class.run,
                                            static_argnums=(3,4))
                x = x_init
                logger = ListLogger(save=False)
                for _ in range(n_loops):
                    x, transition_operator_state, interesting_info = jitted_transition(
                       key, transition_operator_state, x, i, self.intermediate_target_log_prob_fn)
                    logger.write(interesting_info)
                x_init = np.array(x_init)
                x = np.array(x)
                fig, axs = plt.subplots(3)
                axs[0].plot(x_init[:, 0], x_init[:, 1], "o")
                axs[0].set_title("initial samples")
                axs[1].plot(x[:, 0], x[:, 1], "o")
                axs[1].set_title("HMC samples")
                axs[2].plot(target_samples[:, 0], target_samples[:, 1], "p")
                axs[2].set_title("target samples")
                plt.tight_layout()
                plt.show()

                plot_history(logger.history)
                plt.show()
        else:
            pass


if __name__ == '__main__':
  absltest.main()
