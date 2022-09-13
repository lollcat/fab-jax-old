"""
Note: Having a solid step size is important for these experiments.
To obtain a good step size we visualise the samples from HMC in a normalised histogram, p
lotted against the target distribution. This worked better than tuning the step size
with p_accept=0.65.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.utils import get_dist, ais_get_info, \
    grad_over_p, grad_over_q, plot_snr, grad_with_ais_p2_over_q, grad_with_ais_p_target, plot, \
    get_step_size_ais, true_gradient_alpha_2_div, analytic_alpha_2_div
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

figsize= 5.5
figsize = (figsize, figsize+1.0)

# Setup distributions p and q.
loc = 0.5
dim = 1
mean_q = jnp.array([loc]*dim)
mean_p = None  # use default
key = jax.random.PRNGKey(1)
batch_sizes = [100, 500, 1000, 5000, 10000]
total_samples = batch_sizes[-1] * 100


# Setup AIS.
tune_step_size_ais = False
n_ais_dist = 4


# Note: It is important to tune the below step size. If set to large the gradient for FAB becomes
# biased.
AIS_kwargs_custom = {
    "transition_operator_type": "hmc",
    "additional_transition_operator_kwargs": {
        "n_inner_steps": 5,
        "init_step_size": 0.5,
        "n_outer_steps": 1,
        "step_tuning_method": None
    }
            }
AIS_kwargs_blackjax = {
    "transition_operator_type": "hmc_blackjax",
    "additional_transition_operator_kwargs": {
        "n_inner_steps": 5,
        "init_step_size": 1.0,
        "n_outer_steps": 1,
    }
            }
use_blackjax = True

AIS_kwargs = AIS_kwargs_blackjax

if __name__ == '__main__':
    ais = AnnealedImportanceSampler(
        dim=dim, n_intermediate_distributions=n_ais_dist,
        **AIS_kwargs
    )

    if tune_step_size_ais:
        tune_batch_size = 20
        tune_n_iter = 1000
        transition_operator_state_p = get_step_size_ais(
            mean_q=mean_q,
            n_ais_dist=n_ais_dist,
            ais_kwargs=AIS_kwargs,
            p_target=True,
            batch_size=tune_batch_size,
            n_iter=tune_n_iter,
            mean_p=mean_p,
            )
        transition_operator_state_p_2_div_q = get_step_size_ais(
            mean_q=mean_q,
            n_ais_dist=n_ais_dist,
            ais_kwargs=AIS_kwargs,
            p_target=False,
            batch_size=tune_batch_size,
            n_iter=tune_n_iter,
            mean_p=mean_p,
            )
    else:
        transition_operator_state_p = ais.transition_operator_manager.get_init_state()
        transition_operator_state_p_2_div_q = transition_operator_state_p

    grad_hist_p = []
    grad_hist_q = []
    grad_ais_p_target_hist = []
    grad_ais_p2_over_q_hist = []
    log_w_ais_p_target_all, x_ais_p_target_all, info_ais_p = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state_p, p_target=True,
                     ais=ais)
    log_w_ais_p2_over_q_all, x_ais_p2_over_q_all, info_ais_p2_div_q = \
        ais_get_info(mean_q, key, total_samples, transition_operator_state_p_2_div_q,
                     p_target=False, ais=ais)


    print("\np_accepts ais with p target:")
    print([info_ais_p[f"mean_p_accept_dist{i}"] for i in range(n_ais_dist)])
    if not use_blackjax:
        print("\nstep sizes ais with p target: ")
        print(transition_operator_state_p.step_size_params)

    print("\np_accepts ais with p^2/q target:")
    print([info_ais_p2_div_q[f"mean_p_accept_dist{i}"] for i in range(n_ais_dist)])
    if not use_blackjax:
        print("\nstep sizes ais with p^2/q target:")
        print(transition_operator_state_p_2_div_q.step_size_params)


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

    # Now plots:
    mean_q_1d = jnp.array([loc])

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
    plt.savefig(f"empgrad_SNR_nsamples.png", bbox_inches='tight')
    plt.show()

    analytic_grad = true_gradient_alpha_2_div(mean_q)[0]

    fig, axs = plt.subplots(2, 2, sharex=True, figsize= (15, 15))
    axs[0, 0].set_ylabel("gradient w.r.t $\mu_1$")
    axs[0, 1].set_ylabel("gradient w.r.t $\mu_1$")
    axs[1, 1].set_xlabel("batch size")
    axs[1, 0].set_xlabel("batch size")
    axs = axs.flatten()
    plot(batch_sizes, grad_ais_p2_over_q_hist, ax=axs[0], c="b", label="AIS with $g=p^2/q$")
    plot(batch_sizes, grad_ais_p_target_hist, ax=axs[1], c="b", label="AIS with g = p")
    plot(batch_sizes, grad_hist_p, ax=axs[2], c="b", label="IS with p")
    plot(batch_sizes, grad_hist_q, ax=axs[3], c="b", label="IS with q")
    for i in range(4):
        axs[i].plot(batch_sizes, [analytic_grad for i in range(len(batch_sizes))], "--", c="black",
                 label="true gradient")
        axs[i].legend()
    plt.tight_layout()
    plt.show()


    # Plot p and q, p^2/q PDFS vs AIS samples.
    dist_q, dist_p = get_dist(mean_q_1d)
    alpha_2_div_1d = analytic_alpha_2_div(mean_q_1d)
    samples_ais_p2_div_q = x_ais_p2_over_q_all[:, 0]
    samples_ais_p = x_ais_p_target_all[:, 0]
    x = jnp.linspace(-5, 5, 50)[:, None]

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    ax.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    ax.plot(x, jnp.exp(2*dist_p.log_prob(x) - dist_q.log_prob(x)) / alpha_2_div_1d, label="$g \propto p^2/q$")
    ax.hist(samples_ais_p2_div_q, density=True, bins=200, label="AIS with $g=p^2/q$")
    ax.hist(samples_ais_p, density=True, bins=200, label="AIS with $g=p$")
    plt.legend(loc=(1.04, 0.))
    ax.set_xlabel("x")
    ax.set_ylabel("PDF")
    fig.savefig("ais_samples.png", bbox_inches="tight")
    fig.show()
