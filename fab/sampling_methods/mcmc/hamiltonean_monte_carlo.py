import jax.numpy as jnp
import jax
import optax
from typing import NamedTuple
from functools import partial
from fab.types import Params
import chex


# TODO: types



class NoGradParams(NamedTuple):
    std_devs: jnp.ndarray # per dist, per element


class HMCStatePAccept(NamedTuple):
    # these params are the step sizes
    no_grad_params: NoGradParams  # additional params not directly used to compute step size
    step_size_params: jnp.ndarray


class HMCStateGradientBased(NamedTuple):
    no_grad_params: NoGradParams  # additional params not directly used to compute step size
    step_size_params: Params
    optimizer_state: optax.OptState


class Info(NamedTuple):
    average_acceptance_probabilities_per_outer_loop: jnp.ndarray
    std_per_element: jnp.ndarray
    mean_delta_per_element: jnp.ndarray
    expected_scaled_mean_distance_per_outer_loop: jnp.ndarray



class HamiltoneanMonteCarlo:
    def __init__(self, dim, n_intermediate_distributions, intermediate_target_log_prob_fn,
                 batch_size, step_tuning_method="p_accept", n_outer_steps=1, n_inner_steps=5,
                 initial_step_size: float = 1.0, lr=1e-3):
        """ Everything inside init is fixed throughout training, as self is static"""
        self.dim = dim
        self.intermediate_target_log_prob_fn = intermediate_target_log_prob_fn
        self.n_intermediate_distributions = n_intermediate_distributions
        self.batch_size = batch_size
        self.step_tuning_method = step_tuning_method
        self.n_outer_steps = n_outer_steps
        self.n_inner_steps = n_inner_steps
        self._initial_step_size = initial_step_size
        if self.step_tuning_method == "gradient_based":
            self.lr = lr
        elif self.step_tuning_method == "p_accept":
            self.target_p_accept = 0.65
        else:
            raise NotImplementedError

        def U(learnt_dist_params, q, i):
            j = i + 1  # j is loop iter param in annealed_importance_sampling.py
            return - self.intermediate_target_log_prob_fn(learnt_dist_params, q, j)
        def grad_U(learnt_dist_params, q, j):
            return jnp.clip(jax.grad(U, argnums=1)(learnt_dist_params, q, j), a_min=-100.0,
                            a_max=100.0)
        self.U = U
        self.grad_U = grad_U

    def get_init_state(self):
        no_grad_params = NoGradParams(jnp.ones((
            self.n_intermediate_distributions, self.dim)))
        if self.step_tuning_method == "gradient_based":
            step_params = {}
            for i in range(self.n_intermediate_distributions):
                step_params_per_dist = jnp.log(
                        jnp.ones((self.n_outer_steps, self.dim)))
                step_params[i] = step_params_per_dist
            self.optimizer = optax.adam(self.lr)
            initial_opt_state = self.optimizer.init(step_params)
            initial_state = HMCStateGradientBased(no_grad_params, step_params, initial_opt_state)
        elif self.step_tuning_method == "p_accept":
            step_size = jnp.ones((self.n_intermediate_distributions,  self.n_outer_steps))*\
                        self._initial_step_size
            initial_state = HMCStatePAccept(no_grad_params, step_size)
        else:
            raise NotImplemented
        return initial_state

    def get_step_size_param_for_dist(self, step_size_params, i):
        if self.step_tuning_method == "p_accept":
            return step_size_params[i]
        elif self.step_tuning_method == "gradient_based":
            return jnp.exp(step_size_params[i])
        else:
            raise NotImplemented


    @staticmethod
    def _inner_most_loop_func(grad_U, L, learnt_dist_params, i, carry, xs):
        p, q = jax.lax.stop_gradient(carry)  # ensure only grad to epsilon for final iter
        l, epsilon = xs
        q = q + epsilon * p
        p = jax.lax.cond(l != (L - 1), lambda p: p - epsilon * grad_U(learnt_dist_params, q, i),
                         lambda p: p, p)
        return (p, q), None


    @staticmethod
    def _outer_loop_func(U, grad_U, inner_most_loop_func, L, learnt_dist_params, i, carry, xs):
        current_q = jax.lax.stop_gradient(carry)
        rng_key = xs["rng_key"]
        epsilon = xs["step_size"]
        q = current_q
        rng_key, subkey = jax.random.split(rng_key)
        p = jax.random.normal(key=subkey, shape=(q.shape))
        current_p = p
        p = p - epsilon * grad_U(learnt_dist_params, q, i) / 2
        xs_inner = (jnp.arange(L), jnp.repeat(epsilon[None, ...], L, axis=0))
        (p, q), _ = jax.lax.scan(inner_most_loop_func, init=(p, q), xs=xs_inner)
        p = p - epsilon * grad_U(learnt_dist_params, q, i) / 2
        p = -p

        U_current = U(learnt_dist_params, current_q, i)
        U_proposed = U(learnt_dist_params, q, i)
        K_current = jnp.sum(current_p ** 2) / 2
        K_proposed = jnp.sum(p ** 2) / 2

        acceptance_probability = jnp.clip(jnp.exp(U_current - U_proposed + K_current - K_proposed),
                                          a_max=1.0)
        # reject samples that have nan acceptance prob
        acceptance_probability_clip_low = jnp.nan_to_num(acceptance_probability, nan=0.0, posinf=0.0,
                                                         neginf=0.0)
        accept = (acceptance_probability_clip_low > jax.random.uniform(key=subkey,
                                shape=acceptance_probability_clip_low.shape))
        current_q = jax.lax.select(accept, q, current_q)
        return current_q, (current_q, acceptance_probability_clip_low)

    @partial(jax.vmap, in_axes=(None, 0, None, None, 0, None))
    def _run(self, key, learnt_distribution_params, step_size, x, i):
        current_q = x  # set current_q equal to input x from AIS
        chex.assert_shape(current_q, (self.dim,))
        xs = {"rng_key": jax.random.split(key, self.n_outer_steps),
              "step_size": step_size}
        inner_most_loop_func = partial(self._inner_most_loop_func, self.grad_U,
                                       self.n_inner_steps, learnt_distribution_params, i)
        outer_loop_func = partial(self._outer_loop_func, self.U, self.grad_U,
                                  inner_most_loop_func,
                                  self.n_inner_steps, learnt_distribution_params, i)
        current_q, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            jax.lax.scan(outer_loop_func, init=(current_q), xs=xs)
        return current_q, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop)


    def get_key_info(self, x_batch_initial, current_q_per_outer_loop,
                     acceptance_probabilities_per_outer_loop,
                     transition_operator_additional_state_info, i) -> Info:
        """Info for calculating losses, and for inspections"""
        # average over batch
        average_acceptance_probabilities_per_outer_loop = jnp.mean(
            acceptance_probabilities_per_outer_loop, axis=0)

        # calculate distance moved
        # block grad as we use this for calculating grad based loss
        starting_points = jax.lax.stop_gradient(jnp.concatenate([x_batch_initial[:, None, ...],
                                                                 current_q_per_outer_loop[:, :-1,
                                                                 ...
                                                                 ]], axis=1))
        delta_per_element = current_q_per_outer_loop - starting_points
        mean_delta_per_outer_loop_per_element = jnp.mean(jnp.abs(delta_per_element), axis=0)  # save
        # for
        # logging
        std_per_element = jnp.std(current_q_per_outer_loop[:, -1, :], axis=0)
        # distance with std scaling
        stds = transition_operator_additional_state_info.std_devs[i][None, None, :]
        chex.assert_equal_rank([delta_per_element, stds])  # check broadcasting will be fine
        squared_scaled_distance = jnp.sum((delta_per_element / stds)**2, axis=-1)
        chex.assert_equal_rank([acceptance_probabilities_per_outer_loop,
                                squared_scaled_distance]) # check safe broadcasting
        expected_scaled_mean_distance_per_outer_loop = \
            acceptance_probabilities_per_outer_loop * squared_scaled_distance

        return Info(average_acceptance_probabilities_per_outer_loop,
                    std_per_element,
                    mean_delta_per_outer_loop_per_element,
                    expected_scaled_mean_distance_per_outer_loop)


    def run_and_loss(self, key, learnt_distribution_params,
                     transition_operator_step_size_params, transition_operator_additional_state_info,
                     x_batch, i):
        seeds = jax.random.split(key, self.batch_size)
        step_size = self.get_step_size_param_for_dist(transition_operator_step_size_params, i)
        x_batch_final, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            self._run(seeds, learnt_distribution_params,
                      step_size, x_batch, i)
        info = self.get_key_info(x_batch, current_q_per_outer_loop,
                                 acceptance_probabilities_per_outer_loop,
                                 transition_operator_additional_state_info, i)
        if self.step_tuning_method == "p_accept":
            return x_batch_final, info
        elif self.step_tuning_method == "gradient_based":
            # prevent nan from 1/0.0
            expected_scaled_msd = jnp.where(
                info.expected_scaled_mean_distance_per_outer_loop == 0.0,
                jnp.ones_like(info.expected_scaled_mean_distance_per_outer_loop)*0.1,
                info.expected_scaled_mean_distance_per_outer_loop
                                            )
            loss = jnp.mean(1.0 / expected_scaled_msd  -
                            expected_scaled_msd)
            return loss, (x_batch_final, info)
        else:
            raise NotImplementedError

    def get_interesting_info(self, info) -> dict:
        return info._asdict()

    def update_no_grad_params(self, i, no_grad_params, info) -> NoGradParams:
        return no_grad_params

    def update_step_size_p_accept(self, step_size_params,
                              average_acceptance_probabilities_per_outer_loop, i):
        multiplying_factor = jnp.where(average_acceptance_probabilities_per_outer_loop >
                                  self.target_p_accept, jnp.array(1.05),
                                       jnp.array(1.0)/1.05
                                  )
        chex.assert_equal_shape([step_size_params[i], multiplying_factor])
        step_size_params = \
            step_size_params.at[i].set(step_size_params[i]*multiplying_factor)
        return step_size_params



    def run(self, key, learnt_distribution_params,
            transition_operator_state, x_batch, i):
        """Vectorised run with updates to transition operator state"""
        if self.step_tuning_method == "p_accept":
            x_out, info = self.run_and_loss(key, learnt_distribution_params,
                                            transition_operator_state.step_size_params,
            transition_operator_state.no_grad_params, x_batch, i)
            # tune step size to reach target of p_accept = 0.65
            new_step_size_params = self.update_step_size_p_accept(
                transition_operator_state.step_size_params,
                info.average_acceptance_probabilities_per_outer_loop,
            i)
            no_grad_params = self.update_no_grad_params(i,
                                                        transition_operator_state.no_grad_params,
                                                        info)
            new_transition_operator_state = HMCStatePAccept(no_grad_params, new_step_size_params)
        else:
            (loss, (x_out, info)), grads = jax.value_and_grad(self.run_and_loss,
                                                            has_aux=True, argnums=2)(
                key, learnt_distribution_params, transition_operator_state.step_size_params,
                transition_operator_state.no_grad_params, x_batch, i)
            updates, new_opt_state = self.optimizer.update(grads,
                                                           transition_operator_state.optimizer_state)
            step_size_params_new = optax.apply_updates(transition_operator_state.step_size_params,
            updates)
            no_grad_params = self.update_no_grad_params(i, transition_operator_state.no_grad_params,\
                             info)
            new_transition_operator_state = HMCStateGradientBased(no_grad_params,
                                                                  step_size_params_new,
                                                                  new_opt_state)

        interesting_info = self.get_interesting_info(info)
        return x_out, new_transition_operator_state, interesting_info
