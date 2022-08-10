import chex
import jax_md
import jax


class LennardJones:
    def __init__(self, n_particles: int, d_space: int, sigma: float = 1.0, epsilon: float = 1.0):
        """
        Args:
            n_particles: number of particles in the system
            d_space: dim of particle coordinates
            sigma: parameter of lennard jones energy function
            epsilon: parameter of lennard jones energy function
        """
        self.n_particles = n_particles
        self.d_space = d_space
        self.sigma = sigma
        self.epsilon = epsilon

        displacement_fn, shift_fn = jax_md.space.free()
        self.energy_fn = jax_md.energy.lennard_jones_pair(displacement_fn,
                                                          sigma=sigma,
                                                          epsilon=epsilon)

        self.dim = n_particles*d_space


    def log_prob(self, x):
        """Takes in flattened x, reshapes it to match the number of particles and dimension,
        and compute the log prob (equal to the negative lennard jones energy). """
        if len(x.shape) == 1:
            x = x[:, None].reshape(self.n_particles, self.d_space)
            return - self.energy_fn(x)
        elif len(x.shape) == 2:
            x = x[:, :, None].reshape(-1, self.n_particles, self.d_space)
            return - jax.vmap(self.energy_fn)(x)
        else:
            raise ValueError("x must be of shape [batch_size, n*d] or [n*d")


if __name__ == '__main__':
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    d_space = 1
    n = 2
    lj = LennardJones(n_particles=n, d_space=d_space)
    n_points = 100
    max_ = 10.0
    x = jnp.concatenate(
        [jnp.zeros((n_points,))[:, None],
         jnp.linspace(0.01, max_, n_points)[:, None]], axis=-1)
    log_probs = lj.log_prob(x)
    chex.assert_shape(log_probs, (x.shape[0], ))
    plt.plot(x[:, 1], log_probs, "o-")
    plt.title("log pdf")
    plt.yscale("symlog")
    plt.show()

    plt.plot(x[:, 1], jnp.exp(log_probs), "o-")
    plt.title("pdf")
    plt.show()



