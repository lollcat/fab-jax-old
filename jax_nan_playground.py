import jax
import jax.numpy as jnp
import haiku as hk

def f(x):
    weight = jnp.ones_like(x).at[0].set(-jnp.array(float("inf")))
    valid_samples = jnp.isfinite(weight)
    inner_term = x + weight
    inner_term = jnp.where(valid_samples, inner_term, -jnp.ones_like(inner_term) * float("inf"))
    y = jax.nn.logsumexp(inner_term, axis=0)
    return y


def g(x):
    weight = - jnp.ones_like(x) * jnp.array(float("inf"))
    valid_samples = jnp.isfinite(weight)
    inner_term = x + weight
    inner_term = jnp.where(valid_samples, inner_term, -jnp.ones_like(inner_term) * float("inf"))
    y = jax.nn.logsumexp(inner_term, axis=0)
    return y

def h(params):
    x = forward.apply(params)
    loss = f(x)
    return loss


@hk.without_apply_rng
@hk.transform
def forward():
    x = jnp.ones(3)
    y = hk.nets.MLP([3, 3], b_init=hk.initializers.RandomNormal())(x)
    return y


if __name__ == '__main__':
    val, grad = jax.value_and_grad(f)(jnp.ones(3))
    print(val, grad)
    val, grad = jax.value_and_grad(g)(jnp.ones(3))
    print(val, grad)

    params = forward.init(jax.random.PRNGKey(0))
    val, grad = jax.value_and_grad(h)(params)
    print(grad)

    # print(params)
