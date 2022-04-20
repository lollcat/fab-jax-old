
from typing import Tuple, Sequence, NamedTuple, Optional

import chex
import haiku as hk
import distrax
import jax.numpy as jnp
import jax.random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability.substrates.jax as tfp
import time


class Linear(hk.Module):
  """Linear module, adjusted for scaling, initialisation not used as we sample from prior. """

  def __init__(
      self,
      output_size: int,
      b_scale: float = 0.5,
      use_bias: bool = True,
      name: Optional[str] = None,

  ):
    super().__init__(name=name)
    self.use_bias = use_bias
    self.input_size = None
    self.output_size = output_size
    self.w_init = jnp.zeros
    self.b_init = jnp.zeros
    self.b_scale = b_scale

  def __call__(
      self,
          inputs: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w = hk.get_parameter("w", [input_size, output_size], dtype, init=self.w_init)
    w_stddev = 1. / np.sqrt(self.input_size)
    w = w * w_stddev

    out = jnp.dot(inputs, w)

    if self.use_bias:
        b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
        b = jnp.broadcast_to(b, out.shape)
        b = b * self.b_scale
        out = out + b

    return out


class BNN(hk.Module):
    def __init__(self, input_dim, num_classes, mlp_units: Sequence[int], temperature):
        super(BNN, self).__init__()
        self.hidden_layers = [Linear(mlp_unit, b_scale=0.5 if i == 0 else 1.0, use_bias=i == 0)
                              for i, mlp_unit in enumerate(mlp_units)]
        self.output_layer = Linear(num_classes, use_bias=False)
        self.temperature = temperature

    def __call__(self, x):
        """We add a dimension to the logits, so that our event shape is 1, to match y."""
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.relu(x)
        logits = self.output_layer(x)[..., None, :] * 1/self.temperature
        dist = distrax.Independent(distrax.Categorical(logits=logits), reinterpreted_batch_ndims=1)
        return dist


class Data(NamedTuple):
    x: chex.Array
    y: chex.Array

class BNNEnergyFunction:
    def __init__(self,
                 seed: int = 0,
                 prior_scale: float = 1.0,
                 bnn_mlp_units: Sequence[int] = (3, 3),
                 train_n_points: int = 10,
                 x_scale: float = 2.0,
                 temperature: float = 0.2,
                 ):
        input_dim = 2
        num_classes = 2
        self.prior_scale = prior_scale
        self.x_scale = x_scale
        def forward(x):
            bnn = BNN(input_dim, num_classes, mlp_units=bnn_mlp_units, temperature=temperature)
            return bnn(x)
        self.bnn = hk.without_apply_rng(hk.transform(forward))
        key = jax.random.PRNGKey(seed)
        key1, key2, key3 = jax.random.split(key, 3)
        init_params = self.bnn.init(key, jnp.zeros(input_dim))
        flat_params, self.tree_struc = jax.tree_flatten(init_params)
        self.shapes = jax.tree_map(lambda x: x.shape, flat_params)
        self.block_n_params = [np.prod(shape) for shape in self.shapes]
        self.split_indices = np.cumsum(self.block_n_params[:-1])
        self.dim = np.sum(self.block_n_params)
        self.target_params = self.array_to_tree(self.prior.sample(seed=key))
        self.train_data = self.generate_data(key3, train_n_points)
        self._check_flatten_unflatten_consistency()


    def array_to_tree(self, theta: chex.Array) -> hk.Params:
        """Converts array into necessary hk.Params object"""
        blocks = jnp.split(theta, self.split_indices)
        blocks = jax.tree_map(lambda a, shape: a.reshape(shape), blocks, self.shapes)
        tree = jax.tree_unflatten(self.tree_struc, blocks)
        return tree

    def tree_to_array(self, theta: chex.ArrayTree) -> chex.Array:
        theta = jax.tree_map(lambda x: x.flatten(), theta)
        theta, _ = jax.tree_flatten(theta)
        theta = jnp.concatenate(theta)
        return theta

    @property
    def prior(self) -> distrax.Distribution:
        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.dim),
                                               scale_diag=jnp.ones(self.dim)*self.prior_scale)
        return prior

    def prior_prob(self, theta):
        return self.prior.log_prob(theta)

    def target_prob(self, x, y):
        return self.bnn.apply(self.target_params, x).log_prob(y)

    def _log_prob_single(self, theta: chex.Array):
        theta_tree = self.array_to_tree(theta)
        dist_y_given_x = self.bnn.apply(theta_tree, self.train_data.x)
        log_p_y_given_x = jnp.sum(dist_y_given_x.log_prob(self.train_data.y), axis=-1)
        prior = self.prior_prob(theta)
        chex.assert_equal_shape((log_p_y_given_x, prior))
        return log_p_y_given_x + prior

    def log_prob(self, theta: chex.Array):
        if len(theta.shape) == 1:
            return self._log_prob_single(theta)
        else:
            assert len(theta.shape) == 2  # batch_dim, event_dim
            return jax.vmap(self._log_prob_single)(theta)


    def _check_flatten_unflatten_consistency(self):
        """Check that the flattening / unflattening is consistent."""
        keys = jax.random.split(jax.random.PRNGKey(0), 6)
        init_params = jax.vmap(self.bnn.init, in_axes=(0, None))(
            keys, self.train_data.x)
        init_params_flat = jax.vmap(self.tree_to_array)(init_params)
        init_params_ = jax.vmap(self.array_to_tree)(init_params_flat)
        chex.assert_trees_all_equal(init_params, init_params_)
        # init_params['bnn/~/mlp/~/linear_2']['w']
        log_probs = self.log_prob(init_params_flat)
        init_params_single = self.bnn.init(keys[0], self.train_data.x)
        init_params_single_flat = self.tree_to_array(init_params_single)
        log_prob_single = self._log_prob_single(init_params_single_flat)
        assert log_prob_single == log_probs[0]


    def generate_data(self, key, n_points) -> Data:
        key1, key2 = jax.random.split(key)
        x = jax.random.normal(key, shape=(n_points, 2))*self.x_scale
        dist_y = self.bnn.apply(self.target_params, x)
        y = dist_y.sample(seed=key2)
        return Data(x=x, y=y)

    def plot(self, params, ax: Optional[plt.Axes] = None):
        width = self.x_scale*3
        x1, x2 = jnp.meshgrid(jnp.linspace(-width, width, 20), jnp.linspace(-width, width, 20))
        x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
        y_dist: distrax.Categorical = self.bnn.apply(params, x)
        y = y_dist.distribution.logits
        if ax is None:
            fig, ax = plt.subplots(1)
        prob_1 = jax.nn.softmax(jnp.squeeze(y, axis=1), axis=-1)[:, 0].reshape(x1.shape)
        cs = ax.contourf(x1, x2, prob_1, cmap=plt.get_cmap("coolwarm"))
        plt.colorbar(cs, ax=ax)
        x_red = self.train_data.x[jnp.squeeze(self.train_data.y == 0, axis=-1)]
        x_blue = self.train_data.x[jnp.squeeze(self.train_data.y == 1, axis=-1)]
        ax.plot(x_red[:, 0], x_red[:, 1], "o", color="red")
        ax.plot(x_blue[:, 0], x_blue[:, 1], "o", color="blue")

    def plot_params_batch(self, params, ax: Optional[plt.Axes] = None):
        width = self.x_scale*3
        x1, x2 = jnp.meshgrid(jnp.linspace(-width, width, 20), jnp.linspace(-width, width, 20))
        x = jnp.stack([x1.flatten(), x2.flatten()], axis=-1)
        def get_logits(params):
            return self.bnn.apply(params, x).distribution.logits
        y = jnp.mean(jax.vmap(get_logits)(params), axis=0)
        if ax is None:
            fig, ax = plt.subplots(1)
        prob_1 = jax.nn.softmax(jnp.squeeze(y, axis=1), axis=-1)[:, 0].reshape(x1.shape)
        cs = ax.contourf(x1, x2, prob_1, cmap=plt.get_cmap("coolwarm"))
        plt.colorbar(cs, ax=ax)
        x_red = self.train_data.x[jnp.squeeze(self.train_data.y == 0, axis=-1)]
        x_blue = self.train_data.x[jnp.squeeze(self.train_data.y == 1, axis=-1)]
        ax.plot(x_red[:, 0], x_red[:, 1], "o", color="red")
        ax.plot(x_blue[:, 0], x_blue[:, 1], "o", color="blue")


    def setup_posterior_dataset(self, total_size=1000, batch_size=100,
                                shuffle=True, key=jax.random.PRNGKey(0)):
        num_burnin_steps = int(2e3)
        num_results = max(int(1e4), total_size)
        # see https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX
        init_key, sample_key, shuffle_key = jax.random.split(key)
        @jax.jit
        def run_chain(key):
            kernel = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=self.log_prob,
                    num_leapfrog_steps=5,
                    step_size=1.),
                num_adaptation_steps=int(num_burnin_steps * 0.8))
            states, acceptance_probs = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                num_steps_between_results=100,
                current_state=jnp.zeros((self.dim,)),
                kernel=kernel,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                seed=key,
                parallel_iterations=batch_size
            )
            return states[:total_size]

        start_time = time.time()
        states = run_chain(sample_key)
        print(f"time to generate dataset: {(time.time() - start_time) / 60}  min")
        if shuffle:
            states = jax.random.shuffle(x=states, key=shuffle_key, axis=0)
        return states





if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    bnn_target = BNNEnergyFunction(prior_scale=1.0, seed=2, train_n_points=50,
                                   bnn_mlp_units=(5, 5), x_scale=2.0, temperature=0.2)
    print(bnn_target.dim)
    init_params = bnn_target.bnn.init(key, bnn_target.train_data.x)
    bnn_target._log_prob_single(bnn_target.tree_to_array(init_params))
    bnn_target._check_flatten_unflatten_consistency()
    bnn_target.plot(bnn_target.target_params)
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    posterior_samples = bnn_target.setup_posterior_dataset()
    posterior_samples_tree = jax.vmap(bnn_target.array_to_tree)(posterior_samples[0:10])

    bnn_target.plot(bnn_target.array_to_tree(posterior_samples[0]), axs[0, 0])
    bnn_target.plot(bnn_target.array_to_tree(posterior_samples[1]), axs[0, 1])
    bnn_target.plot(bnn_target.array_to_tree(posterior_samples[2]), axs[1, 0])
    bnn_target.plot_params_batch(params=posterior_samples_tree, ax=axs[1, 1])
    axs[0, 0].set_title("p(y | x) for sample from p(theta | X, Y)")
    axs[0, 1].set_title("p(y | x) for sample from p(theta | X, Y)")
    axs[1, 0].set_title("p(y | x) for sample from p(theta | X, Y)")
    axs[1, 1].set_title("Approximate Posterior marginilisng over theta")
    plt.tight_layout()
    plt.show()

