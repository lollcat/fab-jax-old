from fab.learnt_distributions.real_nvp import make_realnvp_dist_funcs
from fab.target_distributions.bnn import BNNEnergyFunction
from fab.agent.fab_agent import AgentFAB
from fab.utils.plotting import plot_history, plot_marginal_pair, plot_contours_2D
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp

from neural_testbed import generative


flow_num_layers = 10
mlp_hidden_size_per_x_dim = 5
layer_norm = False
act_norm = True

bnn_mlp_units = (5, 5)
bnn_problem = BNNEnergyFunction(bnn_mlp_units=bnn_mlp_units)
dim = bnn_problem.dim
print(f"running bnn with {dim} parameters")

target_log_prob = bnn_problem.log_prob
real_nvp_flo = make_realnvp_dist_funcs(dim, flow_num_layers,
                                       mlp_hidden_size_per_x_dim=mlp_hidden_size_per_x_dim,
                                      layer_norm=layer_norm, act_norm=act_norm)



batch_size = 64
eval_batch_size = batch_size*10
n_evals = 20
n_iter = int(5e3)
lr = 2e-4
n_intermediate_distributions: int = 2
AIS_kwargs = {"transition_operator_type": "hmc_tfp"}
optimizer = optax.chain(optax.zero_nans(), optax.adam(lr))
loss_type = "forward_kl"  # "forward_kl"  "alpha_2_div"
style = "vanilla"


def make_enn(fab_agent):
    @jax.jit
    def enn_sampler(x, key):
        key1, key2, key3 = jax.random.split(key, 3)
        base_log_prob = fab_agent.get_base_log_prob(fab_agent.state.learnt_distribution_params)
        x_base, log_q_x_base = fab_agent.learnt_distribution.sample_and_log_prob.apply(
            fab_agent.state.learnt_distribution_params, rng=key1,
            sample_shape=(batch_size,))
        x_ais_loss, log_w_ais, _, _ = \
            fab_agent.annealed_importance_sampler.run(
                x_base, log_q_x_base, key2,
                fab_agent.state.transition_operator_state,
                base_log_prob=base_log_prob,
                target_log_prob=target_log_prob
            )
        index = jax.random.choice(jax.random.PRNGKey(0), log_w_ais.shape[0],
                                  p=jax.nn.softmax(log_w_ais), shape=(),
                                  replace=True)
        theta_tree = bnn_problem.array_to_tree(x_ais_loss[index])
        dist_y = bnn_problem.bnn.apply(theta_tree, x)
        return jnp.squeeze(dist_y.distribution.logits, axis=-2)  # dist_y.sample(seed=key3)

    return enn_sampler

def plotter(fab_agent):
    enn_sampler = make_enn(fab_agent)
    plots = generative.sanity_plots(bnn_problem.problem, enn_sampler)
    p = plots['more_enn']
    _ = p.draw(show=True)
    p = plots['sample_enn']
    _ = p.draw(show=True)


# def plotter(fab_agent):
#     @jax.jit
#     def get_info(state):
#         base_log_prob = fab_agent.get_base_log_prob(state.learnt_distribution_params)
#         target_log_prob = fab_agent.get_target_log_prob(state.learnt_distribution_params)
#         x_base, log_q_x_base = fab_agent.learnt_distribution.sample_and_log_prob.apply(
#             state.learnt_distribution_params, rng=state.key,
#             sample_shape=(batch_size,))
#         x_ais_loss, _, _, _ = \
#             fab_agent.annealed_importance_sampler.run(
#                 x_base, log_q_x_base, state.key,
#                 state.transition_operator_state,
#                 base_log_prob=base_log_prob,
#                 target_log_prob=target_log_prob
#             )
#         return x_base, x_ais_loss
#
#     fig, axs = plt.subplots(1, 2)
#     x_base, x_ais_target = get_info(fab_agent.state)
#     plot_marginal_pair(x_base, ax=axs[0])
#     axs[0].set_title("base samples")
#     plot_marginal_pair(x_ais_target, ax=axs[1])
#     axs[1].set_title("ais samples")
#     plt.show()
#     return [fig]


fab_agent = AgentFAB(learnt_distribution=real_nvp_flo,
                     target_log_prob=target_log_prob,
                     n_intermediate_distributions=n_intermediate_distributions,
                     AIS_kwargs=AIS_kwargs,
                     optimizer=optimizer,
                     plotter=plotter,
                     loss_type=loss_type,
                     style=style)


if __name__ == '__main__':
    plotter(fab_agent)
    fab_agent.run(n_iter=n_iter, batch_size=batch_size, n_plots=5, n_evals=n_evals,
                  eval_batch_size=eval_batch_size)
    plt.plot(fab_agent.logger.history["ess_base"])
    plt.title("ess_base")
    plt.show()
    plt.plot(fab_agent.logger.history["ess_ais"])
    plt.title("ess_ais")
    plt.show()

    plt.plot(fab_agent.logger.history["eval_ess_flow"])
    plt.title("ess_base eval")
    plt.show()
    plt.plot(fab_agent.logger.history["eval_ess_ais"])
    plt.title("ess_ais eval")
    plt.show()


    plotter(fab_agent)
    # x = bnn_problem.problem.train_data.x
    # dist_y = enn_sampler(bnn_problem.problem.train_data.x, jax.random.PRNGKey(0))