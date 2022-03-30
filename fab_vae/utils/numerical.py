import jax.numpy as jnp
import chex

def remove_inf_and_nan(z_samples: chex.Array, log_w_ais: chex.Array):
  valid_samples = jnp.isfinite(log_w_ais) & jnp.all(jnp.isfinite(z_samples), axis=-1)
  # remove invalid z_samples so we don't get NaN gradients.
  z_samples = jnp.where(valid_samples[..., None].repeat(z_samples.shape[-1], axis=-1),
                        z_samples, jnp.zeros_like(z_samples))
  log_w_ais = jnp.where(valid_samples, log_w_ais, -jnp.ones_like((log_w_ais)) * float("inf"))
  return z_samples, log_w_ais