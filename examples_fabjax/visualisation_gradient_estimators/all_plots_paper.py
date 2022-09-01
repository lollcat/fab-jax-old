raise Exception("old")

import jax
import jax.numpy as jnp
import numpy as np
import distrax
import matplotlib.pyplot as plt
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from functools import partial
from matplotlib import rc
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('axes', titlesize=24, labelsize=24)  # fontsize of the axes title and labels
rc('legend', fontsize=24)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)
rc("lines", linewidth=4)
figsize=5.5
figsize = (figsize, figsize+1.0)

# Setup distributions p and q.
loc = 0.5
mean_q = loc
key = jax.random.PRNGKey(0)
batch_sizes = [100, 1000, 10000]
dist_q = distrax.Independent(distrax.Normal(loc=[loc], scale=1), reinterpreted_batch_ndims=1)
dist_p = distrax.Independent(distrax.Normal(loc=[-loc], scale=1), reinterpreted_batch_ndims=1)

# Setup AIS.
n_ais_dist = 3
AIS_kwargs = {"transition_operator_type": "hmc_tfp",
        "additional_transition_operator_kwargs": {
                       "n_inner_steps": 5,
                       "init_step_size": 1.6}
                  }
# AIS_kwargs = {"transition_operator_type": "metropolis_tfp",
#         "additional_transition_operator_kwargs": {
#                        "init_step_size": 1.0,
#                         "n_inner_steps": 10},
#               }
ais = AnnealedImportanceSampler(
    dim=1, n_intermediate_distributions=n_ais_dist,
    **AIS_kwargs
)
transition_operator_state = ais.transition_operator_manager.get_init_state()


# Setup gradient estimators.

def loss_over_p(mean_q, batch_size, key):
    dist_q = distrax.Independent(distrax.Normal(loc=[mean_q], scale=1), reinterpreted_batch_ndims=1)
    x, log_p = dist_p.sample_and_log_prob(seed=key, sample_shape=(batch_size,))
    log_q = dist_q.log_prob(x)
    return jnp.mean(jnp.exp(log_p - log_q))


def loss_over_q(mean_q, batch_size, key):
    dist_q = distrax.Independent(distrax.Normal(loc=[mean_q], scale=1), reinterpreted_batch_ndims=1)
    x, log_q = dist_q.sample_and_log_prob(seed=key, sample_shape=(batch_size,))
    log_p = dist_p.log_prob(x)
    return jnp.mean(jnp.exp(2*log_p - 2*log_q))


def grad_over_p(mean, batch_size, key):
    return jax.grad(loss_over_p)(mean, batch_size, key)

def grad_over_q(mean, batch_size, key):
    return jax.grad(loss_over_q)(mean, batch_size, key)

def plot_snr(batch_sizes, loss_hist, ax, c="b", label="", linestyle="-", log_scale=True):
    means = np.array([np.mean(loss_hist[i]) for i in range(len(loss_hist))])
    stds = np.array([np.std(loss_hist[i]) for i in range(len(loss_hist))])
    ax.plot(batch_sizes, means/stds, color=c, label=label, linestyle=linestyle)
    if log_scale:
        ax.set_xscale("log")

def plot(batch_sizes, loss_or_grad_hist, ax, c="b", label="", log_scale=False):
    means = np.array([np.mean(loss_or_grad_hist[i]) for i in range(len(loss_or_grad_hist))])
    stds = np.array([np.std(loss_or_grad_hist[i]) for i in range(len(loss_or_grad_hist))])
    ax.plot(batch_sizes, means, color=c, label=label)
    ax.fill_between(batch_sizes, means - stds, means + stds, alpha=0.1, color=c)
    if log_scale:
        ax.set_xscale("log")


@partial(jax.jit, static_argnums=(2, 4, 5))
def ais_forward(mean, key, batch_size, transition_operator_state, p_target, ais):
    key1, key2 = jax.random.split(key)
    # Define event dim for dist q and p for AIS.
    dist_q = distrax.Independent(distrax.Normal(loc=[mean], scale=1), reinterpreted_batch_ndims=1)
    base_log_prob = dist_q.log_prob
    if p_target:
        target_log_prob = dist_p.log_prob
    else:
        target_log_prob = lambda x: 2 * dist_p.log_prob(x) - dist_q.log_prob(x)
    x, log_q = dist_q.sample_and_log_prob(seed=key1, sample_shape=(batch_size,))
    x_ais, log_w_ais, new_transition_operator_state, info = ais.run(
        x, log_q,
     key2,
     transition_operator_state,
     base_log_prob=base_log_prob,
     target_log_prob=target_log_prob)
    return x_ais, log_w_ais, new_transition_operator_state

def ais_get_info(mean, key, batch_size, transition_operator_state, p_target, ais=ais):
    """Run multiple AIS forward passed to get `batch_size` many samples.
    No updating of the transition operator state."""
    n_samples_inner = 100
    # assert batch_size % n_samples_inner == 0
    x_ais_list, log_w_ais_list= [], []
    for i in range(batch_size // n_samples_inner):
        key, subkey = jax.random.split(key)
        x_ais, log_w_ais, _ = ais_forward(mean, subkey, n_samples_inner, transition_operator_state,
                                          p_target, ais=ais)
        x_ais_list.append(x_ais)
        log_w_ais_list.append(log_w_ais)
    log_w_ais = jnp.concatenate(log_w_ais_list)
    x_ais = jnp.concatenate(x_ais_list)
    return log_w_ais, x_ais


def grad_with_ais_p_target(mean, x_ais, log_w_ais):
    log_p_x = dist_p.log_prob(x_ais)
    def loss(mean_q):
        dist_q = distrax.Independent(distrax.Normal(loc=[mean_q], scale=1), reinterpreted_batch_ndims=1)
        log_q_x = dist_q.log_prob(x_ais)
        f_x = jnp.exp(log_p_x - log_q_x)
        return jnp.sum(jax.nn.softmax(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean)


def grad_with_ais_p2_over_q(mean, x_ais, log_w_ais):
    def loss(mean_q):
        dist_q = distrax.Independent(distrax.Normal(loc=[mean_q], scale=1), reinterpreted_batch_ndims=1)
        log_q_x = dist_q.log_prob(x_ais)
        f_x =  - log_q_x
        return jnp.mean(jnp.exp(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean)



if __name__ == '__main__':
    # ************************ PART 1 *************************
    grad_hist_p = []
    grad_hist_q = []
    grad_ais_p_target_hist = []
    grad_ais_p2_over_q_hist = []

    total_samples = batch_sizes[-1] * 50
    log_w_ais_p_target_all, x_ais_p_target_all = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state, p_target=True)
    log_w_ais_p2_over_q_all, x_ais_p2_over_q_all = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state, p_target=False)

    for batch_size in batch_sizes:
        n_runs = total_samples // batch_size
        key_batch = jax.random.split(key, n_runs)

        # over p and q
        grad_p = np.asarray(jax.vmap(grad_over_p,
                                     in_axes=(None, None, 0))(
            mean_q, batch_size, key_batch))
        grad_q = np.asarray(jax.vmap(grad_over_q,
                                     in_axes=(None, None, 0))(
            mean_q, batch_size, key_batch))
        grad_hist_p.append(grad_p)
        grad_hist_q.append(grad_q)

        # AIS with p target.
        log_w_ais_p_target = jnp.reshape(log_w_ais_p_target_all, (n_runs, batch_size))
        x_ais_p_target = jnp.reshape(x_ais_p_target_all, (n_runs, batch_size, 1))
        loss_ais_p_target, grad_ais_p_target = jax.vmap(grad_with_ais_p_target,
                                                        in_axes=(None, 0, 0))(
            mean_q, x_ais_p_target, log_w_ais_p_target)
        grad_ais_p_target_hist.append(grad_ais_p_target)

        # AIS with p^2/q target.
        log_w_ais_p2_over_q = jnp.reshape(log_w_ais_p2_over_q_all, (n_runs, batch_size))
        x_ais_p2_over_q = jnp.reshape(x_ais_p2_over_q_all, (n_runs, batch_size, 1))
        loss_ais_p2_over_q, grad_ais_p2_over_q = jax.vmap(grad_with_ais_p2_over_q,
                                                          in_axes=(None, 0, 0))(
            mean_q, x_ais_p2_over_q, log_w_ais_p2_over_q)
        grad_ais_p2_over_q_hist.append(grad_ais_p2_over_q)

    fig, ax = plt.subplots(figsize=figsize)
    plot_snr(batch_sizes, grad_hist_p, linestyle="dotted", ax=ax, c="black", label="IS with p")
    plot_snr(batch_sizes, grad_hist_q, linestyle="dashed", ax=ax, c="black", label="IS with q")
    plot_snr(batch_sizes, grad_ais_p_target_hist, ax=ax, c="b", label="AIS with g = p")
    plot_snr(batch_sizes, grad_ais_p2_over_q_hist, ax=ax, c="r",
             label="AIS with $g=p^2/q$")
    plt.ylabel("SNR")
    plt.xlabel("Number of samples")
    plt.ylim(0, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig("empgrad_SNR_nsamples.png", bbox_inches='tight')
    plt.show()

    # ************************ PART 2 *************************
    grad_ais_hist_p2_over_q = []
    grad_ais_hist_p = []
    mean_q = loc
    key = jax.random.PRNGKey(0)
    n_ais_dists = [1, 2, 4, 8, 16, 32]
    n_runs = 2000
    batch_size = 100
    total_batch_size = n_runs*batch_size

    grad_p = np.asarray(jax.vmap(grad_over_p, in_axes=(None, None, 0))(
        mean_q, batch_size, jax.random.split(key, n_runs)))
    grad_q = np.asarray(jax.vmap(grad_over_q, in_axes=(None, None, 0))(
        mean_q, batch_size, jax.random.split(key, n_runs)))

    for n_ais_dist in n_ais_dists:
        ais = AnnealedImportanceSampler(
            dim=1, n_intermediate_distributions=n_ais_dist,
            **AIS_kwargs
        )
        transition_operator_state = ais.transition_operator_manager.get_init_state()

        # over p
        log_w_ais, x_ais = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=True,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, 1))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p_target, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_p.append(grad_ais)

        # over p^2/q
        log_w_ais, x_ais = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=False,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, 1))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p2_over_q, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_p2_over_q.append(grad_ais)


    n_ais_dists = [0] + n_ais_dists
    grad_ais_hist_p2_over_q = [grad_q] + grad_ais_hist_p2_over_q
    grad_ais_hist_p = [grad_q] + grad_ais_hist_p

    fig, ax = plt.subplots()
    plot(n_ais_dists, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$")
    plot(n_ais_dists, jnp.repeat(grad_p[None, ...], len(n_ais_dists), axis=0),
                                 ax=ax, c="b", label="IS with p")
    plot(n_ais_dists, grad_ais_hist_p,
         ax=ax, c="b", label="AIS with g = p")
    ax.legend()
    plt.xlabel("number of intermediate AIS distributions")
    plt.ylabel("gradient w.r.t mean of q")
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    plot_snr(n_ais_dists, jnp.repeat(grad_p[None, ...], len(n_ais_dists), axis=0),
             ax=ax, c="black", label="IS with p", linestyle="dotted", log_scale=False)
    plot_snr(n_ais_dists, jnp.repeat(grad_q[None, ...], len(n_ais_dists), axis=0),
             ax=ax, c="black", label="IS with q", linestyle="dashed", log_scale=False)
    plot_snr(n_ais_dists, grad_ais_hist_p, ax=ax, c="b", label="AIS with g = p", log_scale=False)
    plot_snr(n_ais_dists, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$", log_scale=False)

    # ax.legend(loc="best") # , bbox_to_anchor=(0.5, 0.25, 0.5, 0.9))
    plt.xlabel("Number of AIS distributions")
    plt.ylim(0)
    plt.ylabel("SNR")
    plt.tight_layout()
    plt.savefig("empgrad_SNR_n_dist.png", bbox_inches='tight')
    plt.show()


    # Plot p and q.
    plt.figure(figsize=figsize)
    x = jnp.linspace(-4, 4, 50)[:, None]
    dist_q = distrax.Independent(distrax.Normal(loc=[loc], scale=1), reinterpreted_batch_ndims=1)
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig("empgrad_PDF.png", bbox_inches='tight')
    plt.show()
