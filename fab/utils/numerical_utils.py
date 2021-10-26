
import jax.numpy as jnp
import jax

def effective_sample_size_from_unnormalised_log_weights(unnormalised_sampling_log_weights):
    assert len(unnormalised_sampling_log_weights.shape) == 1
    n = unnormalised_sampling_log_weights.shape[0]
    unnormalised_sampling_log_weights = jnp.nan_to_num(unnormalised_sampling_log_weights,
                                                       nan=-1e6, neginf=-1e6)
    normalised_sampling_weights = jax.nn.softmax(unnormalised_sampling_log_weights, axis=-1)
    return jnp.squeeze(1.0 / jnp.sum(normalised_sampling_weights ** 2) / n)