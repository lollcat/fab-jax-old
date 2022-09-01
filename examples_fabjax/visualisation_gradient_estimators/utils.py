import jax
import jax.numpy as jnp
import numpy as np
import distrax
from functools import partial



def get_dist(mean_q, mean_p = None):
    """If mean_p is None we use a default setting where it is centered on -mean_q for each dim."""
    assert len(mean_q.shape) == 1
    n_dim = mean_q.shape[0]
    if mean_p is None:
        mean_p = - jax.lax.stop_gradient(mean_q)
    assert mean_q.shape == mean_p.shape
    dist_p = distrax.MultivariateNormalDiag(loc=mean_p,
                                            scale_diag=jnp.ones(n_dim))
    dist_q = distrax.MultivariateNormalDiag(loc=mean_q,
                                            scale_diag=jnp.ones(n_dim))
    return dist_q, dist_p


# Setup gradient estimators.

def loss_over_p(mean_q, batch_size, key, mean_p=None, vanilla_form=True):
    """Added vanilla_form=False option to check that it doesn't give different results
    (plots look the same)."""
    dist_q, dist_p = get_dist(mean_q, mean_p)
    x, log_p = dist_p.sample_and_log_prob(seed=key, sample_shape=(batch_size,))
    log_q = dist_q.log_prob(x)
    log_w = log_p - log_q
    w = jnp.exp(log_w)
    if vanilla_form:
        return jnp.mean(w)
    else:
        return - jnp.mean(jax.lax.stop_gradient(w) * log_q)


def loss_over_q(mean_q, batch_size, key, mean_p = None, vanilla=True):
    """Added vanilla_form=False option to check that it doesn't give different results
    (plots look the same)."""
    dist_q, dist_p = get_dist(mean_q, mean_p)
    x, log_q = dist_q.sample_and_log_prob(seed=key, sample_shape=(batch_size,))
    log_p = dist_p.log_prob(x)
    log_w = log_p - log_q
    w_sq = jnp.exp(2*(log_w))
    if vanilla:
        return jnp.mean(w_sq)
    else:
        return - jnp.mean(jax.lax.stop_gradient(w_sq) * log_q)


def grad_over_p(mean_q, batch_size, key, mean_p = None):
    return jax.grad(loss_over_p)(mean_q, batch_size, key, mean_p)

def grad_over_q(mean_q, batch_size, key, mean_p=None):
    return jax.grad(loss_over_q)(mean_q, batch_size, key, mean_p=mean_p)

def plot_snr(batch_sizes, loss_hist, ax, c="b", label="", draw_style="-", log_scale=True):
    means = np.array([np.mean(loss_hist[i]) for i in range(len(loss_hist))])
    stds = np.array([np.std(loss_hist[i]) for i in range(len(loss_hist))])
    ax.plot(batch_sizes, means / stds, draw_style, color=c, label=label)
    if log_scale:
        ax.set_xscale("log")

def plot(batch_sizes, loss_or_grad_hist, ax, draw_style="-", c="b", label="", log_scale=False):
    means = np.array([np.mean(loss_or_grad_hist[i]) for i in range(len(loss_or_grad_hist))])
    stds = np.array([np.std(loss_or_grad_hist[i]) for i in range(len(loss_or_grad_hist))])
    ax.plot(batch_sizes, means, draw_style, color=c, label=label)
    ax.fill_between(batch_sizes, means - stds, means + stds, alpha=0.1, color=c)
    if log_scale:
        ax.set_xscale("log")


@partial(jax.jit, static_argnums=(2, 4, 5))
def ais_forward(mean_q,
                key,
                batch_size,
                transition_operator_state,
                p_target: bool,
                ais,
                mean_p = None):
    dist_q, dist_p = get_dist(mean_q, mean_p)
    key1, key2 = jax.random.split(key)
    base_log_prob = dist_q.log_prob
    if p_target:
        target_log_prob = dist_p.log_prob
    else:
        target_log_prob = lambda x: 2 * dist_p.log_prob(x) - dist_q.log_prob(x)
    x, log_q = dist_q.sample_and_log_prob(seed=key1, sample_shape=(batch_size,))
    x_ais, log_w_ais, new_transition_operator_state, info = \
        ais.run(
        x,
        log_q,
        key2,
        transition_operator_state,
        base_log_prob=base_log_prob,
        target_log_prob=target_log_prob)
    return x_ais, log_w_ais, new_transition_operator_state

def ais_get_info(mean, key, batch_size, transition_operator_state, p_target, ais, mean_p = None):
    """Run multiple AIS forward passed to get `batch_size` many samples.
    No updating of the transition operator state."""
    n_samples_inner = 100
    # assert batch_size % n_samples_inner == 0
    x_ais_list, log_w_ais_list = [], []
    for i in range(batch_size // n_samples_inner):
        key, subkey = jax.random.split(key)
        x_ais, log_w_ais, _ = ais_forward(mean, subkey, n_samples_inner, transition_operator_state,
                                          p_target, ais=ais, mean_p=mean_p)
        x_ais_list.append(x_ais)
        log_w_ais_list.append(log_w_ais)
    log_w_ais = jnp.concatenate(log_w_ais_list)
    x_ais = jnp.concatenate(x_ais_list)
    return log_w_ais, x_ais


def grad_with_ais_p_target(mean_q, x_ais, log_w_ais, mean_p=None):
    def loss(mean_q):
        dist_q, dist_p = get_dist(mean_q, mean_p)
        log_p_x = dist_p.log_prob(x_ais)
        log_q_x = dist_q.log_prob(x_ais)
        f_x = jnp.exp(log_p_x - log_q_x)
        return jnp.sum(jax.nn.softmax(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean_q)


def grad_with_ais_p2_over_q(mean, x_ais, log_w_ais):
    def loss(mean_q):
        dist_q, _ = get_dist(mean_q)
        log_q_x = dist_q.log_prob(x_ais)
        f_x = - log_q_x
        return jnp.mean(jnp.exp(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean)