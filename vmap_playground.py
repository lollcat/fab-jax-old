import haiku as hk
import distrax
import jax.numpy as jnp
import jax


class TestModule(hk.Module):

    def __call__(self, x, *args, **kwargs):
        mean = hk.nets.MLP((5, 5))(x)
        return distrax.Normal(loc=mean, scale=jnp.ones_like(mean))


if __name__ == '__main__':
    def f(x):
        return TestModule()(x)
    fun = hk.without_apply_rng(hk.transform(f))
    x = jnp.ones(3)
    params = fun.init(x=x, rng=jax.random.PRNGKey(0))
    dist = fun.apply(params, x)


    def z(x, func):
        return func(x)

    jax.vmap(z, in_axes=(0, None))(jnp.ones((3, 4)), lambda x: x + 1)

    # can we return vmapped distribution object?
    dist2 = jax.vmap(fun.apply, in_axes=(None, 0))(params, jnp.ones((4, 3)))

    def g(dist):
        return dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(3,))

    # but we cant vmap over distribution objects.
    batch_samples = jax.vmap(g)(dist2)

