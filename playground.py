import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=0)
def func(static_arg):
    if static_arg == 0:
        return jnp.array(0.0)
    else:
        return jnp.array(1.0)

if __name__ == '__main__':
    print(func(0))
    print(func(1))