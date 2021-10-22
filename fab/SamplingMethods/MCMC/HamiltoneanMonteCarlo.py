import jax.numpy as jnp
import jax
import optax
from dataclasses import dataclass
from functools import partial
from fab.types import Params
import chex

# TODO: ask alex

@dataclass
class HMCStatePAccept:
    # these params are the step sizes
    step_size_params: jnp.ndarray = jnp.array(1.0)

@dataclass
class HMCStateGradientBased:
    step_size_params: Params
    no_grad_params: Params
    optimizer_state: optax.OptState




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

        def U(learnt_dist_params, q, i):
            j = i + 1  # j is loop iter param in AnnealedImportanceSampling.py
            return - self.intermediate_target_log_prob_fn(learnt_dist_params, q, j)
        def grad_U(learnt_dist_params, q, j):
            return jnp.clip(jax.grad(U, argnums=1)(learnt_dist_params, q, j), a_min=-100.0,
                            a_max=100.0)
        self.U = U
        self.grad_U = grad_U

    def get_init_state(self):
        if self.step_tuning_method == "gradient_based":
            step_params = {}
            for i in range(self.n_intermediate_distributions):
                step_params_per_dist = jnp.log(
                        jnp.ones((self.n_outer_steps, self.dim)))
                step_params[i] = step_params_per_dist
            self.optimizer = optax.adam(self.lr)
            initial_opt_state = self.optimizer.init(step_params)
            no_grad_params = {"stds_per_dist_per_dim": jnp.ones((
                self.n_intermediate_distributions,  self.n_outer_steps, self.dim))}
            initial_state = HMCStateGradientBased(step_params, no_grad_params, initial_opt_state)
        elif self.step_tuning_method == "p_accept":
            step_size = jnp.ones((self.n_intermediate_distributions,  self.n_outer_steps))*\
                        self._initial_step_size
            initial_state = HMCStatePAccept(step_size)
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
        p, q = carry
        l, epsilon = xs
        q = q + epsilon * p
        p = jax.lax.cond(l != (L - 1), lambda p: p - epsilon * grad_U(learnt_dist_params, q, i),
                         lambda p: p, p)
        return (p, q), None


    @staticmethod
    def _outer_loop_func(U, grad_U, inner_most_loop_func, L, learnt_dist_params, i, carry, xs):
        current_q = carry
        rng_key = xs["rng_key"]
        epsilon = xs["step_size"]
        q = current_q
        rng_key, subkey = jax.random.split(rng_key)
        p = jax.random.normal(key=subkey, shape=(q.shape))
        current_p = p
        p = p - epsilon * grad_U(learnt_dist_params, q, i) / 2
        xs_inner = (jnp.arange(L), jnp.repeat(epsilon[None, ...], L))
        (p, q), _ = jax.lax.scan(inner_most_loop_func, init=(p, q), xs=xs_inner)
        p = p - epsilon * grad_U(learnt_dist_params, q, i) / 2
        p = -p

        U_current = U(learnt_dist_params, current_q, i)
        U_proposed = U(learnt_dist_params, q, i)
        current_K = jnp.sum(current_p ** 2) / 2
        proposed_K = jnp.sum(p ** 2) / 2

        acceptance_probability = jnp.clip(jnp.exp(U_current - U_proposed + current_K - proposed_K),
                                          a_max=1.0)
        # reject samples that have nan acceptance prob
        acceptance_probability_clip_low = jnp.nan_to_num(acceptance_probability, nan=0.0, posinf=0.0,
                                                         neginf=0.0)
        accept = (acceptance_probability_clip_low > jax.random.uniform(key=subkey,
                                shape=acceptance_probability_clip_low.shape))
        current_q = jax.lax.select(accept, q, current_q)
        return current_q, (current_q, acceptance_probability_clip_low)

    @partial(jax.vmap, in_axes=(None, 0, None, None, 0, None))
    def run(self, key, learnt_distribution_params, step_size, x, i):
        current_q = x  # set current_q equal to input x from AIS
        chex.assert_shape(current_q == (self.dim,))
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
                     acceptance_probabilities_per_outer_loop):
        """Info for calculating losses, and for inspections"""
        info = {}
        # average over batch
        info["average_acceptance_probabilities_per_outer_loop"] = jnp.mean(
            acceptance_probabilities_per_outer_loop, axis=0)

        # calculate distance moved
        # block grad as we use this for calculating grad based loss
        starting_points = jax.lax.stop_gradient(jnp.concatenate([x_batch_initial[:, None, ...],
                                                                 current_q_per_outer_loop[:, :-1,
                                                                 ...
                                                                 ]], axis=1))
        delta_per_element = current_q_per_outer_loop - starting_points
        distance_per_batch_per_outer_loop = jnp.linalg.norm(delta_per_element, ord=2, axis=-1)
        mean_distance_per_outer_loop = jnp.mean(distance_per_batch_per_outer_loop, axis=0)
        info["mean_distance_per_outer_loop"] = mean_distance_per_outer_loop
        # distance with std scaling
        # TODO
        return info


    def vectorised_run(self, key, learnt_distribution_params,
                       transition_operator_step_sizes, transition_operator_additional_state_info,
                       x_batch, i):
        seeds = jax.random.split(key, self.batch_size)
        step_size = self.get_step_size_param_for_dist(transition_operator_step_sizes, i)
        x_batch_final, (current_q_per_outer_loop, acceptance_probabilities_per_outer_loop) = \
            self.run(seeds, learnt_distribution_params,
                     step_size, x_batch, i)
        info = self.get_key_info(x_batch, current_q_per_outer_loop,
                                 acceptance_probabilities_per_outer_loop)
        if self.step_tuning_method == "p_accept":
            return x_batch_final, info
        elif self.step_tuning_method == "gradient_based":
            p_weighted_scaled_distance_squared = acceptance_probabilities_per_outer_loop *\
                                         info["mean_distance_per_outer_loop"]  / \
                              transition_operator_additional_state_info["stds_per_dist_per_dim"][i]
            loss = jnp.mean(1.0 / p_weighted_scaled_distance_squared -
                            p_weighted_scaled_distance_squared)
            return loss, (x_batch_final, info)
        else:
            raise NotImplementedError

    def get_interesting_info(self, info):
        return info

    def update_no_grad_params(self, i, no_grad_params, info):
        return no_grad_params

    def vectorised_run_with_update(self, key, learnt_distribution_params,
                       transition_operator_state, x_batch, i):
        if self.step_tuning_method == "p_accept":
            x_out, info = self.vectorised_run(key, learnt_distribution_params,
                                        transition_operator_state, x_batch, i)
            # tune step size to reach target of p_accept = 0.65
            new_transition_operator_state = transition_operator_state
        else:
            (loss, x_out, info), grads = jax.value_and_grad(self.vectorised_run,
                                                     has_aux=True, argnums=2)(
                key, learnt_distribution_params, transition_operator_state.step_size_params,
                transition_operator_state.no_grad_params, x_batch, i)
            updates, new_opt_state = self.optimizer.update(grads,
                                                           transition_operator_state.opt_state)
            step_size_params_new = optax.apply_updates(transition_operator_state.step_size_params,
            updates)
            no_grad_params = self.update_no_grad_params(i, transition_operator_state.no_grad_params,\
                             info)
            new_transition_operator_state = HMCStateGradientBased(step_size_params_new,
                                                                  no_grad_params, new_opt_state)

        interesting_info = self.get_interesting_info(info)
        return x_out, new_transition_operator_state, interesting_info

