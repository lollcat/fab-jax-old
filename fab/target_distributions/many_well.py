import jax
import jax.numpy as jnp
import numpy as np

class Energy:
    """
    https://zenodo.org/record/3242635#.YNna8uhKjIW
    """
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        e_func = lambda x: jnp.sum(self.energy(x, temperature=temperature))
        return -jax.grad(e_func)(x)


class DoubleWellEnergy(Energy):
    def __init__(self, dim, a=-0.5, b=-6, c=1.):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d**2 + self._c * d**4
        e2 = jnp.sum(0.5 * v**2, axis=-1, keepdims=True)
        return e1 + e2

    def log_prob(self, x):
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)
        return jnp.squeeze(-self.energy(x))


class ManyWellEnergy:
    def __init__(self, dim=4, *args, **kwargs):
        assert dim % 2 == 0
        self.n_wells = dim // 2
        self.double_well_energy = DoubleWellEnergy(dim=2, *args, **kwargs)
        self.dim = dim
        self.centre = 1.7
        self.max_dim_for_all_modes = 40 # otherwise we get memory issues on huuuuge test set
        if self.dim < self.max_dim_for_all_modes:
            dim_1_vals_grid = np.meshgrid(*[np.array([-self.centre, self.centre])for _ in
                                          range(self.n_wells)])
            dim_1_vals = np.stack([dim.flatten() for dim in dim_1_vals_grid], axis=-1)
            n_modes = 2**self.n_wells
            assert n_modes == dim_1_vals.shape[0]
            self.test_set__ = np.zeros((n_modes, dim))
            self.test_set__[:, np.arange(dim) % 2 == 0] = dim_1_vals
            self.test_set__ = jnp.array(self.test_set__)
        else:
            raise NotImplementedError

        self.true_energy_difference = 1.73 # calculated by by evaluating linspace of points changing x1, setting x2=0
        self.shallow_well_bounds = [-1.75, -1.65]
        self.deep_well_bounds = [1.7, 1.8]

        # TODO: test set stuff needs to be written and checked

    def log_prob(self, x):
        return sum([self.double_well_energy.log_prob(x[..., i*2:i*2+2]) for i in range(
                self.n_wells)])

    def log_prob_2D(self, x):
        # for plotting, given 2D x
        return self.double_well_energy.log_prob(x)

if __name__ == '__main__':
    dim = 2
    energy = DoubleWellEnergy(dim=2)
    x = jax.random.normal(jax.random.PRNGKey(42), shape=(3, dim))
    print(energy.log_prob(x))
    print(energy.force(x))

    import itertools
    import matplotlib.pyplot as plt
    from fab.utils.plotting import plot_3D
    import numpy as np
    bound = -3
    n_points = 200
    x_points_dim1 = np.linspace(-bound, bound, n_points)
    x_points_dim2 = np.linspace(-bound, bound, n_points)
    x_points = np.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_probs = energy.log_prob(x_points)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plot_3D(x_points, np.exp(log_probs), n_points, ax, title="log p(x)")
    plt.show()