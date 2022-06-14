from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import jax.random


class PointCloud3DTransform:
    """Converts x in 3D space into polar coordinates where x[0] is the centre, and
    theta, and phi for x[1] are set to 0.0, and sign(theta(x_2)) is positive.
    Finally the angle dimensions dimensions are normalised to be between -1.0 and 1.0, and the
    distance is normalised by the provided length scale.
    We refer to the original space as x, and the transformed, lower dimensional space as z."""
    def __init__(self, length_scale: float = 1.0) -> None:
        self.length_scale = length_scale

    def x_to_z(self, x: chex.Array) -> chex.Array:
        chex.assert_shape(x[0, ...], (3, ))  # works for single x, use vmap for batch.
        x = x[1:] - x[0]  # centre.
        r, theta, phi = self._cartesian_to_polar(x)
        # set theta(x_1) & phi(x_1) to 0
        theta = theta[1:] - theta[0]
        phi = phi[1:] - phi[0]
        # normalise theta and phi
        theta = theta / jnp.pi
        phi = phi / jnp.pi
        # ensure we always map to the same theta and phi
        theta = theta % 2.0
        phi = phi % 2.0
        # set theta(x_2) & phi(x_2) to be between 0 and 1
        theta = jax.lax.select(theta[0] < 1, theta, 2 - theta)
        phi = jax.lax.select(phi[0] < 1, phi, 2 - phi)
        r = r / self.length_scale
        z = jnp.concatenate([r, theta, phi], axis=0)
        return z

    def z_to_x(self, z: chex.Array) -> chex.Array:
        r, theta, phi = jnp.array_split(z, 3)
        # reverse normalisation of angles and distances
        r = r * self.length_scale
        theta = theta * jnp.pi
        phi = phi * jnp.pi
        # we fixed theta(x_2) & phi(x_2) to be equal to 0
        theta = jnp.concatenate((jnp.array([0.0]), theta), axis=0)
        phi = jnp.concatenate((jnp.array([0.0]), phi), axis=0)
        x = self._polar_to_cartesian(r, theta, phi)
        # we fixed x_0 to be [0, 0, 0]
        x = jnp.concatenate((jnp.zeros((1, 3)), x), axis=0)
        return x


    def _cartesian_to_polar(self, x: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Converts x to polar co-ordinates."""
        r = jnp.sqrt(jnp.sum(x**2, axis=-1))
        theta = jnp.arctan2(x[:, 1], x[:, 0])
        phi = jnp.arccos(x[:, 2] / r)
        return r, theta, phi


    def _polar_to_cartesian(self, r, theta, phi) -> chex.Array:
        chex.assert_tree_all_equal_shapes(r, theta, phi)
        x = r * jnp.cos(theta) * jnp.sin(phi)
        y = r * jnp.sin(theta) * jnp.sin(phi)
        z = r * jnp.cos(phi)
        return jnp.concatenate((x[:, None], y[:, None], z[:, None]), axis=-1)


if __name__ == '__main__':
    # check that the function runs without errors, and that it obeys an energy function
    # that is rotation, translation, reflection invariant.
    def distance_fn(a, b):
        return jnp.sum((a - b)**2, axis=-1)

    n_particles = 10
    x = jax.random.normal(jax.random.PRNGKey(0), shape=(n_particles, 3))
    transform_manager = PointCloud3DTransform()
    z = transform_manager.x_to_z(x)
    chex.assert_shape(z, (np.prod(x.shape) - 5, ))
    x_ = transform_manager.z_to_x(z)
    chex.assert_equal_shape((x, x_))
    z_ = transform_manager.x_to_z(x_)
    print(z, z_)
    print(z - z_)
    r, theta, phi = jnp.array_split(z, 3)
    r_, theta_, phi_ = jnp.array_split(z_, 3)
    for i in range(n_particles):
        for j in range(n_particles):
            distance = distance_fn(x[i], x[j])
            distance_ = distance_fn(x_[i], x_[j])
            print(distance - distance_)


