import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.utils import get_dist, ais_get_info, \
    grad_over_p, grad_over_q, plot_snr, grad_with_ais_p2_over_q, grad_with_ais_p_target, plot, \
    log_w_over_p
from examples_fabjax.visualisation_gradient_estimators.grad_estimation_n_samples import \
    figsize  # loc, AIS_kwargs,


if __name__ == '__main__':
    # mpl.rcParams['figure.dpi'] = 300
    # rc('font', **{'family': 'serif', 'serif': ['Times']})
    # rc('text', usetex=True)
    # rc('axes', titlesize=15, labelsize=15)  # fontsize of the axes title and labels
    # rc('legend', fontsize=15)
    # rc('xtick', labelsize=12)
    # rc('ytick', labelsize=12)

    # Whether or not the dims besides the first one are the same.
    # del loc
    loc = 0.5
    AIS_kwargs = {
        "transition_operator_type": "hmc",
        "additional_transition_operator_kwargs": {
            "n_inner_steps": 5,
            "init_step_size": 0.4,
            "n_outer_steps": 1,
            "step_tuning_method": None
        }
    }
    common_alt_dims = False
    distribution_spacing_type = "linear"  # "geometric"
    ais_samples_hist = []
    ais_log_w_hist = []
    p_IS_log_w_hist = []
    grad_hist_over_p = []
    grad_ais_hist_p2_over_q = []

    key = jax.random.PRNGKey(0)
    n_dims = [1, 2, 4, 8] # , 16, 32]  # , 16, 32, 48, 64]
    n_ais_dist_s = [n_dim*2 for n_dim in n_dims]
    n_runs = 10000
    batch_size = 100
    total_batch_size = n_runs*batch_size

    for (n_ais_dist, n_dim) in zip(n_ais_dist_s, n_dims):
        print(n_ais_dist, n_dim)
        n_intermediate_dist = n_ais_dist - 1
        mean_q = jnp.array([loc] * n_dim)
        if common_alt_dims:
            mean_p = jnp.array([-loc] + [loc] * (n_dim - 1))
        else:
            mean_p = -mean_q
        assert mean_q.shape == mean_p.shape

        grad_p = np.asarray(jax.vmap(grad_over_p, in_axes=(None, None, 0, None))(
            mean_q, batch_size, jax.random.split(key, n_runs), mean_p))
        grad_hist_over_p.append(grad_p[:, 0])
        log_w_p = log_w_over_p(mean_q, batch_size*n_runs, key, mean_p)
        p_IS_log_w_hist.append(log_w_p)

        # AIS based gradient estimators
        ais = AnnealedImportanceSampler(
            dim=n_dim, n_intermediate_distributions=n_intermediate_dist,
            distribution_spacing_type=distribution_spacing_type,
            **AIS_kwargs
        )
        transition_operator_state = ais.transition_operator_manager.get_init_state()

        # over p^2/q
        log_w_ais, x_ais, ais_info = ais_get_info(mean_q,
                                        key,
                                        total_batch_size,
                                        p_target=False,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais,
                                        mean_p=mean_p)
        ais_samples_hist.append(x_ais)
        ais_log_w_hist.append(log_w_ais)

        log_w_ais = jnp.reshape(log_w_ais, (n_runs, batch_size))
        x_ais = jnp.reshape(x_ais, (n_runs, batch_size, n_dim))
        loss_ais, grad_ais = jax.vmap(grad_with_ais_p2_over_q, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                                     log_w_ais)
        grad_ais_hist_p2_over_q.append(grad_ais[:, 0])


    fig, ax = plt.subplots()
    plt.plot(n_dims, jnp.var(jnp.asarray(ais_log_w_hist), axis=1), "o-", label="fab")
    plt.plot(n_dims, jnp.var(jnp.asarray(p_IS_log_w_hist), axis=1), "o-", label="is with p")
    plt.ylabel("var(log_w)")
    plt.xlabel("number of dimensions")
    plt.legend()
    plt.savefig(f"empgrad_var_log_w_n_dim_and_n_dist.png", bbox_inches='tight')
    plt.show()


    fig, ax = plt.subplots(figsize=figsize)
    plot_snr(n_dims, grad_ais_hist_p2_over_q, ax=ax, c="r", label="AIS with $g=p^2/q$",
             log_scale=False, draw_style="o-")
    plot_snr(n_dims, grad_hist_over_p,
             ax=ax, c="black", label="IS with p", draw_style=":", log_scale=False)
    # ax.legend(loc="best") # , bbox_to_anchor=(0.5, 0.25, 0.5, 0.9))
    plt.xlabel("Number of dimensions")
    plt.ylim(0)
    plt.ylabel("SNR")
    plt.legend()
    plt.savefig(f"empgrad_SNR_n_dim_and_n_dist.png", bbox_inches='tight')
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
