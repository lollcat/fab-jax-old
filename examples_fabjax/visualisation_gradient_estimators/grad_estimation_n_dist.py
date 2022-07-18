import jax
import jax.numpy as jnp
import distrax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import matplotlib as mpl
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.grad_estimation_n_samples import \
    ais_get_info, grad_with_ais_p2_over_q, plot, plot_snr, grad_over_p, grad_with_ais_p_target, \
    grad_over_q, dist_p, loc, AIS_kwargs


# loc = 0.25
# dist_p = distrax.Independent(distrax.Normal(loc=[-loc], scale=1), reinterpreted_batch_ndims=1)
# AIS_kwargs = {"transition_operator_type": "hmc_tfp",
#         "additional_transition_operator_kwargs": {
#                        "n_inner_steps": 5,
#                        "init_step_size": 1.6}
#                   }


if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 300
    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    rc('axes', titlesize=15, labelsize=15)  # fontsize of the axes title and labels
    rc('legend', fontsize=15)
    rc('xtick', labelsize=12)
    rc('ytick', labelsize=12)


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

    fig, ax = plt.subplots()
    plot_snr(n_ais_dists, grad_ais_hist_p, ax=ax, c="b", label="AIS with g = p", log_scale=False)
    plot_snr(n_ais_dists, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$", log_scale=False)
    plot_snr(n_ais_dists, jnp.repeat(grad_p[None, ...], len(n_ais_dists), axis=0),
             ax=ax, c="black", label="IS with p", linestyle="dotted", log_scale=False)
    plot_snr(n_ais_dists, jnp.repeat(grad_q[None, ...], len(n_ais_dists), axis=0),
             ax=ax, c="black", label="IS with q", linestyle="dashed", log_scale=False)
    ax.legend(loc="best", bbox_to_anchor=(0.5, 0.25, 0.5, 0.9))
    plt.xlabel("Number of intermediate AIS distributions")
    plt.ylim(0)
    plt.ylabel("SNR")
    plt.savefig("empgrad_SNR_n_dist.png")
    plt.show()


    # Plot p and q.
    x = jnp.linspace(-4, 4, 50)[:, None]
    dist_q = distrax.Independent(distrax.Normal(loc=[loc], scale=1), reinterpreted_batch_ndims=1)
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.savefig("empgrad_PDF.png")
    plt.show()
