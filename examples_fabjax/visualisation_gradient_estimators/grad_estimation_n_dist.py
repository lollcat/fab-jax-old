import jax
import jax.numpy as jnp
import numpy as np
import distrax
import matplotlib.pyplot as plt
from fabjax.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from functools import partial


loc = 0.5
dist_p = distrax.Independent(distrax.Normal(loc=[-loc], scale=1), reinterpreted_batch_ndims=1)


@partial(jax.jit, static_argnums=(0, 3))
def ais_forward_alt(ais, mean, key, batch_size, transition_operator_state):
    key1, key2 = jax.random.split(key)
    dist_q = distrax.Independent(distrax.Normal(loc=[mean], scale=1), reinterpreted_batch_ndims=1)
    base_log_prob = dist_q.log_prob
    target_log_prob = lambda x: 2 * dist_p.log_prob(x) - dist_q.log_prob(x)
    x, log_q = dist_q.sample_and_log_prob(seed=key1, sample_shape=(batch_size,))
    x_ais, log_w_ais, new_transition_operator_state, info = ais.run(
        x, log_q,
        key2,
        transition_operator_state,
        base_log_prob=base_log_prob,
        target_log_prob=target_log_prob)
    return x_ais, log_w_ais, new_transition_operator_state


def ais_get_info_alt(mean, key, batch_size, n_ais_dist=5):
    AIS_kwargs = {"transition_operator_type": "hmc_tfp",
                  "additional_transition_operator_kwargs": {
                      "n_inner_steps": 5,
                      "init_step_size": 0.1}
                  }
    ais = AnnealedImportanceSampler(
        dim=1, n_intermediate_distributions=n_ais_dist, **AIS_kwargs
    )
    print(f"n_ais_dist: {n_ais_dist}")
    # print(f"mean, key, batch_size, n_ais_dist, {mean, key, batch_size, n_ais_dist}")
    # print(f"beta space: {ais.beta_space}")

    transition_operator_state = ais.transition_operator_manager.get_init_state()
    n_samples_inner = 100
    # assert batch_size % n_samples_inner == 0
    x_ais_list, log_w_ais_list = [], []
    for i in range(batch_size // n_samples_inner):
        key, subkey = jax.random.split(key)
        x_ais, log_w_ais, _ = ais_forward_alt(ais, mean, subkey, n_samples_inner,
                                              transition_operator_state)
        x_ais_list.append(x_ais)
        log_w_ais_list.append(log_w_ais)
    log_w_ais = jnp.concatenate(log_w_ais_list)
    x_ais = jnp.concatenate(x_ais_list)
    return log_w_ais, x_ais

def grad_with_ais_alt(mean, x_ais, log_w_ais):
    def loss(mean_q):
        dist_q = distrax.Independent(distrax.Normal(loc=[mean_q], scale=1), reinterpreted_batch_ndims=1)
        log_q_x = dist_q.log_prob(x_ais)
        f_x = - log_q_x
        return jnp.mean(jnp.exp(log_w_ais) * f_x)
    return jax.value_and_grad(loss)(mean)


def plot(batch_sizes, loss_hist, ax, c="b", label=""):
    means = np.array([np.mean(loss_hist[i]) for i in range(len(loss_hist))])
    stds = np.array([np.std(loss_hist[i]) for i in range(len(loss_hist))])
    ax.plot(batch_sizes, means, color=c, label=label)
    ax.fill_between(batch_sizes, means - stds, means + stds, alpha=0.1, color=c)
    ax.set_xscale("log")


if __name__ == '__main__':
    grad_ais_hist_alt = []
    mean_q = 0.5
    key = jax.random.PRNGKey(0)
    n_ais_dists = [1, 10, 50, 100]
    n_runs = 100
    batch_size = 1000
    for n_ais_dist in n_ais_dists:
        log_w_ais, x_ais = ais_get_info_alt(mean_q, key, batch_size, n_ais_dist)
        loss_ais, grad_ais = jax.vmap(grad_with_ais_alt, in_axes=(None, 0, 0))(mean_q, x_ais,
                                                                               log_w_ais)
        grad_ais_hist_alt.append(grad_ais)

    fig, ax = plt.subplots()

    plot(n_ais_dists, grad_ais_hist_alt, ax=ax, c="r", label="with ais targetting p^2/q")
    ax.legend()
    plt.xlabel("n ais distributions")
    plt.ylabel("gradient w.r.t mean of q")
    plt.show()
