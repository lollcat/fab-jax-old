import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import numpy as np
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.utils import get_dist, ais_get_info, \
    grad_over_p, grad_over_q, plot_snr, grad_with_ais_p2_over_q, grad_with_ais_p_target, plot
from examples_fabjax.visualisation_gradient_estimators.grad_estimation_n_samples import \
    loc, AIS_kwargs, figsize


if __name__ == '__main__':
    # mpl.rcParams['figure.dpi'] = 300
    # rc('font', **{'family': 'serif', 'serif': ['Times']})
    # rc('text', usetex=True)
    # rc('axes', titlesize=15, labelsize=15)  # fontsize of the axes title and labels
    # rc('legend', fontsize=15)
    # rc('xtick', labelsize=12)
    # rc('ytick', labelsize=12)
    distribution_spacing_type = "linear"  # "linear"

    AIS_kwargs = {
        "transition_operator_type": "hmc",
        "additional_transition_operator_kwargs": {
            "n_inner_steps": 5,
            "init_step_size": 0.4,
            "n_outer_steps": 1,
            "step_tuning_method": None
        }
    }

    grad_ais_hist_p2_over_q = []
    grad_ais_hist_p = []
    dim = 1
    mean_q = jnp.array([loc] * dim)
    key = jax.random.PRNGKey(0)
    n_ais_dists = [1, 2, 4, 8, 16, 32]
    n_runs = 10000
    batch_size = 100
    total_batch_size = n_runs*batch_size

    grad_p = np.asarray(jax.vmap(grad_over_p, in_axes=(None, None, 0))(
        mean_q, batch_size, jax.random.split(key, n_runs)))[:, 0]
    grad_q = np.asarray(jax.vmap(grad_over_q, in_axes=(None, None, 0))(
        mean_q, batch_size, jax.random.split(key, n_runs)))[:, 0]

    for n_ais_dist in n_ais_dists:
        ais = AnnealedImportanceSampler(
            dim=dim, n_intermediate_distributions=n_ais_dist,
            distribution_spacing_type=distribution_spacing_type,
            **AIS_kwargs
        )
        transition_operator_state = ais.transition_operator_manager.get_init_state()

        # over p
        log_w_ais, x_ais, info_ais_p = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=True,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais,
                                        mean_p=None)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, dim))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p_target, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_p.append(grad_ais[:, 0])

        # over p^2/q
        log_w_ais, x_ais, info_ais_p2_div_q = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=False,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais,
                                        mean_p=None)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, dim))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p2_over_q,
                                      in_axes=(None, 0, 0))(mean_q, x_ais,
                                                            log_w_ais)
        grad_ais_hist_p2_over_q.append(grad_ais[:, 0])


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
             ax=ax, c="black", label="IS with p", draw_style=":", log_scale=False)
    plot_snr(n_ais_dists, jnp.repeat(grad_q[None, ...], len(n_ais_dists), axis=0),
             ax=ax, c="black", label="IS with q", draw_style="--", log_scale=False)
    plot_snr(n_ais_dists, grad_ais_hist_p, ax=ax, c="b", label="AIS with g = p", log_scale=False)
    plot_snr(n_ais_dists, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$", log_scale=False)

    # ax.legend(loc="best") # , bbox_to_anchor=(0.5, 0.25, 0.5, 0.9))
    plt.xlabel("Number of AIS distributions")
    plt.ylim(0)
    plt.ylabel("SNR")
    plt.savefig("empgrad_SNR_n_dist.png", bbox_inches='tight')
    plt.show()


    dist_q, dist_p = get_dist(mean_q)
    # Plot p and q.
    plt.figure(figsize=figsize)
    x = jnp.linspace(-4, 4, 50)[:, None]
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.savefig("empgrad_PDF.png", bbox_inches='tight')
    plt.show()
