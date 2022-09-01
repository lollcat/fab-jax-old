import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.utils import get_dist, ais_get_info, \
    grad_over_p, grad_over_q, plot_snr, grad_with_ais_p2_over_q, grad_with_ais_p_target, plot
from matplotlib import rc
import matplotlib as mpl
jax.config.update("jax_enable_x64", True)

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
dim = 1
mean_q = jnp.array([loc]*dim)
key = jax.random.PRNGKey(0)
batch_sizes = [100, 1000, 10000]
total_samples = batch_sizes[-1] * 100


# Setup AIS.
n_ais_dist = 8
AIS_kwargs = {
    "transition_operator_type": "hmc",
    "additional_transition_operator_kwargs": {
        "n_inner_steps": 5,
        "init_step_size": 0.1,
        "n_outer_steps": 1,
        "step_tuning_method": None
    }
            }

# TODO: tune step sizes


if __name__ == '__main__':
    ais = AnnealedImportanceSampler(
        dim=dim, n_intermediate_distributions=n_ais_dist,
        **AIS_kwargs
    )
    transition_operator_state = ais.transition_operator_manager.get_init_state()


    grad_hist_p = []
    grad_hist_q = []
    grad_ais_p_target_hist = []
    grad_ais_p2_over_q_hist = []
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
        grad_hist_p.append(grad_p[:, 0])
        grad_hist_q.append(grad_q[:, 0])

        # AIS with p target.
        log_w_ais_p_target = jnp.reshape(log_w_ais_p_target_all, (n_runs, batch_size))
        x_ais_p_target = jnp.reshape(x_ais_p_target_all, (n_runs, batch_size, dim))
        loss_ais_p_target, grad_ais_p_target = jax.vmap(grad_with_ais_p_target,
                                                        in_axes=(None, 0, 0))(
            mean_q, x_ais_p_target, log_w_ais_p_target)
        grad_ais_p_target_hist.append(grad_ais_p_target[:, 0])

        # AIS with p^2/q target.
        log_w_ais_p2_over_q = jnp.reshape(log_w_ais_p2_over_q_all, (n_runs, batch_size))
        x_ais_p2_over_q = jnp.reshape(x_ais_p2_over_q_all, (n_runs, batch_size, dim))
        loss_ais_p2_over_q, grad_ais_p2_over_q = jax.vmap(grad_with_ais_p2_over_q,
                                                          in_axes=(None, 0, 0))(
            mean_q, x_ais_p2_over_q, log_w_ais_p2_over_q)
        grad_ais_p2_over_q_hist.append(grad_ais_p2_over_q[:, 0])


    fig, ax = plt.subplots(figsize=figsize)
    plot_snr(batch_sizes, grad_hist_p, draw_style=":", ax=ax, c="black", label="IS with p")
    plot_snr(batch_sizes, grad_hist_q, draw_style="--", ax=ax, c="black", label="IS with q")
    plot_snr(batch_sizes, grad_ais_p_target_hist, ax=ax, c="b", label="AIS with g = p")
    plot_snr(batch_sizes, grad_ais_p2_over_q_hist, ax=ax, c="r",
             label="AIS with $g=p^2/q$")
    plt.ylabel("SNR")
    plt.xlabel("Number of samples")
    # plt.ylim(0, 85)
    ax.legend()
    plt.savefig("empgrad_SNR_nsamples.png", bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots()
    plot(batch_sizes, grad_ais_p2_over_q_hist, ax=ax, c="r", label="AIS with $g=p^2/q$")
    plot(batch_sizes, grad_hist_p,
                                 ax=ax, draw_style="--", c="b", label="IS with p")
    plot(batch_sizes, grad_ais_p_target_hist,
         ax=ax, c="b", label="AIS with g = p")
    ax.legend()
    plt.xlabel("number of intermediate AIS distributions")
    plt.ylabel("gradient w.r.t mean of q")
    plt.show()


    # Check histogram of samples from AIS.

    dist_q, dist_p = get_dist(mean_q)
    # Plot p and q.
    plt.figure(figsize=figsize)
    samples = x_ais_p2_over_q_all[:, 0]
    x = jnp.linspace(-loc*10, loc*10, 50)[:, None]
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.hist(samples, density=True, bins=100)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()

    # Plot p and q.
    plt.figure(figsize=figsize)
    samples = x_ais_p_target_all[:, 0]
    x = jnp.linspace(-loc*10, loc*10, 50)[:, None]
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.hist(samples, density=True, bins=100)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()


