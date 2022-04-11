from typing import Tuple, Sequence

import chex
import haiku as hk
import distrax
import jax.numpy as jnp
import jax.random
import numpy as np
from neural_testbed import leaderboard

problem = leaderboard.problem_from_id('classification_2d/75')


class BNN(hk.Module):
    def __init__(self, input_dim, num_classes, mlp_units: Sequence[int] = (5, 5)):
        super(BNN, self).__init__()
        self.mlp = hk.nets.MLP(mlp_units + (num_classes,), activate_final=False)


    def __call__(self, x):
        """We add a dimension to the logits, so that our event shape is 1, to match y."""
        logits = self.mlp(x)[..., None, :]
        dist = distrax.Independent(distrax.Categorical(logits=logits), reinterpreted_batch_ndims=1)
        return dist



class BNNEnergyFunction:
    def __init__(self,
                 seed = 0,
                 prior_scale: float = 1.0,
                 bnn_mlp_units: Sequence[int] = (3, )
                 ):
        self.problem = problem
        input_dim = problem.prior_knowledge.input_dim
        num_classes = problem.prior_knowledge.num_classes

        self.prior_scale = prior_scale
        def forward(x):
            bnn = BNN(input_dim, num_classes, mlp_units=bnn_mlp_units)
            return bnn(x)
        self.bnn = hk.without_apply_rng(hk.transform(forward))
        key = jax.random.PRNGKey(seed)
        init_params = self.bnn.init(key, self.problem.train_data.x)
        flat_params, self.tree_struc = jax.tree_flatten(init_params)
        self.shapes = jax.tree_map(lambda x: x.shape, flat_params)
        self.block_n_params = [np.prod(shape) for shape in self.shapes]
        self.split_indices = np.cumsum(self.block_n_params[:-1])
        self.dim = np.sum(self.block_n_params)
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

    def prior_prob(self, theta):
        return jnp.sum(distrax.Normal(jnp.array(0.0), jnp.array(1.0)).log_prob(theta))

    def _log_prob_single(self, theta: chex.Array):
        theta_tree = self.array_to_tree(theta)
        dist_y_given_x = self.bnn.apply(theta_tree, self.problem.train_data.x)
        log_p_y_given_x = jnp.sum(dist_y_given_x.log_prob(self.problem.train_data.y), axis=-1)
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
            keys, self.problem.train_data.x)
        init_params_flat = jax.vmap(self.tree_to_array)(init_params)
        init_params_ = jax.vmap(self.array_to_tree)(init_params_flat)
        chex.assert_trees_all_equal(init_params, init_params_)
        # init_params['bnn/~/mlp/~/linear_2']['w']
        log_probs = self.log_prob(init_params_flat)

        #
        init_params_single = self.bnn.init(keys[0], self.problem.train_data.x)
        init_params_single_flat = self.tree_to_array(init_params_single)
        log_prob_single = self._log_prob_single(init_params_single_flat)
        assert log_prob_single == log_probs[0]





if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    bnn_target = BNNEnergyFunction()
    print(bnn_target.dim)
    init_params = bnn_target.bnn.init(key, bnn_target.problem.train_data.x)
    bnn_target._log_prob_single(bnn_target.tree_to_array(init_params))
    bnn_target._check_flatten_unflatten_consistency()
