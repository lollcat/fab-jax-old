import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from examples_fabjax.visualisation_gradient_estimators.utils import get_dist, ais_get_info, \
    grad_over_p, grad_over_q, plot_snr, grad_with_ais_p2_over_q, grad_with_ais_p_target, plot
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
            "init_step_size": 1.6,  # 1.6,
            "n_outer_steps": 3,
            "step_tuning_method": None
        }
    }

    if loc != 0.5:  # tuned for loc=0.5 only
        assert AIS_kwargs["additional_transition_operator_kwargs"]["init_step_size"] != 1.6
    common_alt_dims = False
    distribution_spacing_type = "linear"  # "geometric"
    ais_samples_hist = []
    ais_log_w_hist = []
    key = jax.random.PRNGKey(0)
    n_dim = 4
    n_ais_dist_s = [2, 4, 8, 16, 32]
    n_runs = 10000
    batch_size = 100
    total_batch_size = n_runs*batch_size

    for n_ais_dist in n_ais_dist_s:
        n_intermediate_dist = n_ais_dist - 1
        mean_q = jnp.array([loc] * n_dim)
        if common_alt_dims:
            mean_p = jnp.array([-loc] + [loc] * (n_dim - 1))
        else:
            mean_p = -mean_q
        assert mean_q.shape == mean_p.shape

        # AIS based gradient estimators
        ais = AnnealedImportanceSampler(
            dim=n_dim, n_intermediate_distributions=n_intermediate_dist,
            distribution_spacing_type=distribution_spacing_type,
            **AIS_kwargs
        )
        transition_operator_state = ais.transition_operator_manager.get_init_state()

        # over p^2/q
        log_w_ais, x_ais = ais_get_info(mean_q,
                                        key,
                                        total_batch_size,
                                        p_target=False,
                                        transition_operator_state=transition_operator_state,
                                        ais=ais,
                                        mean_p=mean_p)
        ais_samples_hist.append(x_ais)
        ais_log_w_hist.append(log_w_ais)


    fig, ax = plt.subplots()
    plt.plot(n_ais_dist_s, 1/jnp.var(jnp.asarray(ais_log_w_hist), axis=1), "o-")
    plt.title("1/var(log_w) as number of ais distributions increases")
    plt.xlabel("n dist")
    plt.ylabel('1/var(log_w)')
    plt.show()
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
