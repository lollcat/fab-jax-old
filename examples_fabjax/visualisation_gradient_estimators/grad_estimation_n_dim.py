import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.grad_estimation_n_samples import \
    ais_get_info, grad_with_ais_p2_over_q, plot, plot_snr, grad_over_p, grad_with_ais_p_target, \
    grad_over_q, loc, AIS_kwargs, figsize, get_dist


if __name__ == '__main__':
    # mpl.rcParams['figure.dpi'] = 300
    # rc('font', **{'family': 'serif', 'serif': ['Times']})
    # rc('text', usetex=True)
    # rc('axes', titlesize=15, labelsize=15)  # fontsize of the axes title and labels
    # rc('legend', fontsize=15)
    # rc('xtick', labelsize=12)
    # rc('ytick', labelsize=12)


    grad_ais_hist_p2_over_q = []
    grad_hist_over_p = []
    grad_hist_over_q = []
    grad_ais_hist_p = []
    key = jax.random.PRNGKey(0)
    n_dims = [1, 2, 4, 8, 16, 32, 48, 64]
    n_intermediate_dist = 3
    n_runs = 2000
    batch_size = 100
    total_batch_size = n_runs*batch_size

    for n_dim in n_dims:
        mean_q = jnp.array([loc] + [0.0] * (n_dim - 1))
        # using samples from p and q
        grad_p = np.asarray(jax.vmap(grad_over_p, in_axes=(None, None, 0))(
            mean_q, batch_size, jax.random.split(key, n_runs)))
        grad_q = np.asarray(jax.vmap(grad_over_q, in_axes=(None, None, 0))(
            mean_q, batch_size, jax.random.split(key, n_runs)))

        grad_hist_over_q.append(grad_q[:, 0])
        grad_hist_over_p.append(grad_p[:, 0])

        # AIS based gradient estimators
        ais = AnnealedImportanceSampler(
            dim=1, n_intermediate_distributions=n_intermediate_dist,
            **AIS_kwargs
        )
        transition_operator_state = ais.transition_operator_manager.get_init_state()

        # over p
        log_w_ais, x_ais = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=True,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, n_dim))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p_target, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_p.append(grad_ais[:, 0])

        # over p^2/q
        log_w_ais, x_ais = ais_get_info(mean_q, key, total_batch_size,
                                        p_target=False,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais)
        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, n_dim))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p2_over_q, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_p2_over_q.append(grad_ais[:, 0])


    fig, ax = plt.subplots()

    plot(n_dims, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$")
    plot(n_dims, grad_hist_over_p, ax=ax, c="b", label="IS with p")
    plot(n_dims, grad_ais_hist_p, ax=ax, c="b", label="AIS with g = p")
    ax.legend()
    plt.xlabel("number of intermediate AIS distributions")
    plt.ylabel("gradient w.r.t mean of q")
    plt.show()

    fig, ax = plt.subplots()
    plot(n_dims, grad_hist_over_q, ax=ax, c="green", label="IS with q")
    plt.title("SNR over q")
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    plot_snr(n_dims, grad_hist_over_p,
             ax=ax, c="black", label="IS with p", linestyle="dotted", log_scale=False)
    plot_snr(n_dims, grad_hist_over_q,
             ax=ax, c="black", label="IS with q", linestyle="dashed", log_scale=False)
    plot_snr(n_dims, grad_ais_hist_p, ax=ax, c="b", label="AIS with g = p", log_scale=False)
    plot_snr(n_dims, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$", log_scale=False)

    # ax.legend(loc="best") # , bbox_to_anchor=(0.5, 0.25, 0.5, 0.9))
    plt.xlabel("Number of dimensions")
    plt.ylim(0)
    plt.ylabel("SNR")
    plt.legend()
    plt.savefig(f"empgrad_SNR_n_dim_ais_dist{n_intermediate_dist}.png", bbox_inches='tight')
    plt.show()

    mean_q = jnp.array([loc])
    dist_q, dist_p = get_dist(mean_q)
    # Plot p and q.
    plt.figure(figsize=figsize)
    x = jnp.linspace(-4, 4, 50)[:, None]
    plt.plot(x, jnp.exp(dist_q.log_prob(x)), label="q")
    plt.plot(x, jnp.exp(dist_p.log_prob(x)), label="p")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.legend()
    plt.show()
