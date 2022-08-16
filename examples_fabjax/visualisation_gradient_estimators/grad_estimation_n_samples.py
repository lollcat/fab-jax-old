import jax
import jax.numpy as jnp
import numpy as np
import distrax
import matplotlib.pyplot as plt
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from functools import partial
from matplotlib import rc
import matplotlib as mpl
# jax.config.update("jax_enable_x64", True)

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
mean_q = jnp.array([loc])
key = jax.random.PRNGKey(0)
batch_sizes = [100, 1000, 10000]
# dist_q = distrax.Independent(distrax.Normal(loc=[loc], scale=1), reinterpreted_batch_ndims=1)
# dist_p = distrax.Independent(distrax.Normal(loc=[-loc], scale=1), reinterpreted_batch_ndims=1)

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


def get_dist(mean_q, mean_p = None):
    """If mean_p is None we use a default setting where it is centered on -loc for each dim."""
    n_dim = mean_q.shape[0]
    if mean_p is None:
        mean_p = jnp.array([-loc] * n_dim)
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
def ais_forward(mean_q, key, batch_size, transition_operator_state, p_target, ais, mean_p = None):
    dist_q, dist_p = get_dist(mean_q, mean_p)
    key1, key2 = jax.random.split(key)
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

def ais_get_info(mean, key, batch_size, transition_operator_state, p_target, ais, mean_p = None):
    """Run multiple AIS forward passed to get `batch_size` many samples.
    No updating of the transition operator state."""
    n_samples_inner = 100
    # assert batch_size % n_samples_inner == 0
    x_ais_list, log_w_ais_list= [], []
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


def grad_with_ais_p2_over_q(mean, x_ais, log_w_ais, mean_p=None):
    def loss(mean_q):
        dist_q, dist_p = get_dist(mean_q, mean_p)
        log_q_x = dist_q.log_prob(x_ais)
        f_x = - log_q_x
        return jnp.mean(jnp.exp(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean)


if __name__ == '__main__':
    dist_q, dist_p = get_dist(mean_q)
    # Plot p and q.
    plt.figure(figsize=figsize)
    x = jnp.linspace(-loc*10, loc*10, 50)[:, None]
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()


    grad_hist_p = []
    grad_hist_q = []
    grad_ais_p_target_hist = []
    grad_ais_p2_over_q_hist = []

    total_samples = batch_sizes[-1] * 50
    log_w_ais_p_target_all, x_ais_p_target_all = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state, p_target=True, ais=ais)
    log_w_ais_p2_over_q_all, x_ais_p2_over_q_all = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state, p_target=False, ais=ais)

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
    # plt.ylim(0, 85)
    ax.legend()
    plt.savefig("empgrad_SNR_nsamples.png", bbox_inches='tight')
    plt.show()
