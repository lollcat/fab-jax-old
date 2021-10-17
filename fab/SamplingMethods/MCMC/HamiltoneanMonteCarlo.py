import jax.numpy as jnp
import jax
import optax
from dataclasses import dataclass
from functools import partial
from fab.types import Params
import chex

# TODO: I think we need to do stateful transformation
# or some fancy tree stuff - ask Alex
@dataclass
class HMCStatePAccept:
    step_size: jnp.ndarray = jnp.array(1.0)


@dataclass
class HMCStateGradientBased:
    step_size_params: Params
    optimizer_state: optax.OptState


@dataclass
class HMCAuxInfoGradientBased:
    pass


@dataclass
class HMCAuxInfoPAccept:
    pass


class HamiltoneanMonteCarlo:
    def __init__(self, dim, n_intermediate_distributions, intermediate_target_log_prob_fn,
                 step_tuning_method="p_accept", n_outer_steps=1, n_inner_steps=5,
                 initial_step_size: float = 1.0, lr=1e-3):
        """ Everything inside init is fixed throughout training, as self is static"""
        self.dim = dim
        self.intermediate_target_log_prob_fn = intermediate_target_log_prob_fn
        self.n_intermediate_distributions = n_intermediate_distributions
        self.step_tuning_method = step_tuning_method
        self.n_outer_steps = n_outer_steps
        self.n_inner_steps = n_inner_steps
        self._initial_step_size = initial_step_size
        if self.step_tuning_method == "gradient_based":
            self.lr = lr

        def U(learnt_dist_params, q, i):
            j = i + 1 # j is loop iter param in AnnealedImportanceSampling.py
            return - self.intermediate_target_log_prob_fn(learnt_dist_params, q, j)
        def grad_U(learnt_dist_params, q, j):
            return jnp.clip(jax.grad(U, argnums=1)(learnt_dist_params, q, j), a_min=-100.0,
                            a_max=100.0)
        self.U = U
        self.grad_U = grad_U
        self.inner_most_loop_func = partial(self._inner_most_loop_func, grad_U,
                                            self.n_inner_steps)
        self.outer_loop_func = partial(U, grad_U, self.inner_most_loop_func, self.n_inner_steps)


    def get_init_state(self):
        if self.step_tuning_method == "gradient_based":
            step_params = {}
            for i in range(self.n_intermediate_distributions):
                step_params_per_dist = {}
                for n in range(self.n_outer_steps):
                    step_params_per_dist[n] = jnp.log(
                        jnp.ones(self.dim))
                step_params[i] = step_params_per_dist
            self.optimizer = optax.adam(self.lr)
            initial_opt_state = self.optimizer.init(step_params)
            initial_state = HMCStateGradientBased(step_params, initial_opt_state)
        elif self.step_tuning_method == "p_accept":
            step_size = jnp.array(self._initial_step_size)
            initial_state = HMCStatePAccept(step_size)
        else:
            raise NotImplemented
        return initial_state

    def get_step_size_param_for_dist(self, transition_operator_state, i):
        if self.step_tuning_method == "p_accept":
            return transition_operator_state[i]
        elif self.step_tuning_method == "gradient_based":
            return transition_operator_state[i]
        else:
            raise NotImplemented


    @staticmethod
    def _inner_most_loop_func(grad_U, L, carry, xs):
        p, q = carry
        l, epsilon, learnt_dist_params, i = xs
        q = q + epsilon * p
        p = jax.lax.cond(l != (L - 1), lambda p: p - epsilon * grad_U(learnt_dist_params, q, i),
                         lambda p: p, p)
        return (p, q), None

    @staticmethod
    def _outer_loop_func(U, grad_U, inner_most_loop_func, L, carry, xs):
        current_q = carry
        rng_key = xs["rng_key"]
        epsilon = xs["step_size"]
        learnt_dist_params = xs["learnt_dist_params"]
        i = xs["i"]
        q = current_q
        rng_key, subkey = jax.random.split(rng_key)
        p = jax.random.normal(key=subkey, shape=(q.shape))
        current_p = p
        p = p - epsilon * grad_U(learnt_dist_params, q, i) / 2
        xs = ([])
        (p, q), _ = jax.lax.scan(inner_most_loop_func, init=(p, q), xs=xs)
        p = p - epsilon * grad_U(q) / 2
        p = -p

        U_current = U(current_q)
        U_proposed = U(q)
        current_K = jnp.sum(current_p ** 2) / 2
        proposed_K = jnp.sum(p ** 2) / 2

        acceptance_probability = jnp.clip(jnp.exp(U_current - U_proposed + current_K - proposed_K),
                                          a_max=1.0)
        # reject samples that have nan acceptance prob
        acceptance_probability_clip_low = jnp.nan_to_num(acceptance_probability, nan=0.0, posinf=0.0,
                                                         neginf=0.0)
        accept = (acceptance_probability_clip_low > jax.random.uniform(key=subkey,
                                                                       shape=acceptance_probability_clip_low.shape))
        current_q = jax.lax.cond(accept, lambda q: q, lambda q: current_q, q)

        return current_q, acceptance_probability_clip_low

    def run(self, key, learnt_distribution_params, transition_operator_state, x, i):
        loss = 0
        current_q = x # set current_q equal to input x from AIS
        chex.assert_shape(current_q == (self.dim,))
        step_size = self.get_step_size(i)
        xs = {"rng_key": jax.random.split(self.n_outer_steps),
              "step_size": step_size}
        current_q, acceptance_probabilities = \
            jax.lax.scan(self.outer_loop_func, init=(current_q), xs=transition_params_dist)
        return current_q, loss, acceptance_probabilities


    # @jax.jit
    def vectorised_run(step_tuning_method, current_q_batched, params, i, params_transition,
                       rng_key):
        run_func = lambda x, seed: run(x, params, i, params_transition, seed)
        seeds = jax.random.split(rng_key, current_q_batched.shape[0])
        x_out, loss, acceptance_probs = jax.vmap(run_func)(current_q_batched, seeds)
        # compute loss here for gradient based training
        return jnp.mean(loss), (x_out, jnp.mean(acceptance_probs, axis=0))



        # @jax.jit
        def vectorised_run_with_update(current_q_batched, flow_params, i, params_transition, rng_key):
            epsilon, opt_state, _ = params_transition
            loss_func = lambda params_transition: vectorised_run(current_q_batched, flow_params, i,
                                                                 params_transition, rng_key)
            (loss, (x_out, mean_acceptance_prob)), grads = jax.value_and_grad(loss_func, has_aux=True)(
                epsilon)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            epsilon_new = optax.apply_updates(epsilon, updates)
            return x_out, (epsilon_new, new_opt_state, jnp.mean(mean_acceptance_prob))
    else:
        initial_opt_state = None


        # @jax.jit
        def vectorised_run_with_update(current_q_batched, flow_params, i, params_transition, rng_key):
            epsilon, opt_state, _ = params_transition
            loss_func = lambda params_transition: vectorised_run(current_q_batched, flow_params, i,
                                                                 params_transition, rng_key)
            (loss, (x_out, mean_acceptance_prob)), grads = jax.value_and_grad(loss_func, has_aux=True)(
                epsilon)
            epsilon_shared_new = jax.lax.cond(jnp.mean(mean_acceptance_prob) > target_p_accept,
                                              lambda x: x * 1.05, lambda x: x / 1.05, epsilon["shared"])
            epsilon_dist_i_new = batch_cond(mean_acceptance_prob > target_p_accept, lambda x: x * 1.1,
                                            lambda x: x / 1.1,
                                            epsilon["per_dist"][i])
            eps_per_dist = jnp.where(
                jnp.tile(jnp.expand_dims(jnp.arange(n_distributions - 2) == i, axis=-1),
                         epsilon["per_dist"].shape[-1]),
                jnp.broadcast_to(epsilon_dist_i_new, epsilon["per_dist"].shape),
                epsilon["per_dist"])
            epsilon["shared"] = epsilon_shared_new
            epsilon["per_dist"] = eps_per_dist
            return x_out, (epsilon, None, jnp.mean(mean_acceptance_prob))
