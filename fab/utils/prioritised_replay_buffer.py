from typing import NamedTuple, Tuple, Iterable, Callable

import jax.lax
import jax.numpy as jnp
import chex

class AISData(NamedTuple):
    """Log weights and samples generated by annealed importance sampling."""
    x: chex.Array
    log_w: chex.Array
    log_q_old: chex.Array


class PrioritisedBufferState(NamedTuple):
    data: AISData
    is_full: jnp.bool_
    can_sample: jnp.bool_
    current_index: jnp.int32


class PrioritisedReplayBuffer:
    def __init__(self, dim: int,
                 max_length: int,
                 min_sample_length: int,
                 ):
        """
        Create replay buffer for batched sampling and adding of data.
        Args:
            dim: dimension of x data
            max_length: maximum length of the buffer
            min_sample_length: minimum length of buffer required for sampling
            device: replay buffer device

        The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
        to the replay data. For example, if `min_sample_length` is equal to the
        sampling batch size, then we may overfit to the first batch of data, as we would update
        on it many times during the start of training.
        """
        assert min_sample_length < max_length
        self.dim = dim
        self.max_length = max_length
        self.min_sample_length = min_sample_length

    def init(self, key, initial_sampler: Callable[[chex.PRNGKey], Tuple[chex.Array, chex.Array,
                                                                        chex.Array]]):
        """
        key: source of randomness
        initial_sampler: sampler producing x and log_w, used to fill the buffer up to
            the min sample length. The initialised flow + AIS may be used here,
            or we may desire to use AIS with more distributions to give the flow a "good start".
        """
        current_index = 0
        is_full = False  # whether the buffer is full
        can_sample = False  # whether the buffer is full enough to begin sampling
        # init data to have -inf log_w to prevent these values being sampled.
        data = AISData(x=jnp.zeros((self.max_length, self.dim)) * float("nan"),
                              log_w=-jnp.ones(self.max_length, ) * float("inf"),
                       log_q_old=jnp.zeros(self.max_length,) * float("nan")
                       )

        buffer_state = PrioritisedBufferState(data=data, is_full=is_full, can_sample=can_sample,
                                              current_index=current_index)
        while buffer_state.can_sample is False:
            # fill buffer up minimum length
            key, subkey = jax.random.split(key)
            x, log_w, log_q_old = initial_sampler(subkey)
            buffer_state = self.add(x, log_w, log_q_old, buffer_state)
        return buffer_state


    def add(self, x: chex.Array, log_w: chex.Array, log_q_old: chex.Array,
            buffer_state: PrioritisedBufferState) -> PrioritisedBufferState:
        """Add a batch of generated data to the replay buffer"""
        batch_size = x.shape[0]
        indices = (jnp.arange(batch_size) + buffer_state.current_index) % self.max_length
        x = buffer_state.data.x.at[indices].set(x)
        log_w = buffer_state.data.log_w.at[indices].set(log_w)
        log_q_old = buffer_state.data.log_q_old.at[indices].set(log_q_old)
        new_index = buffer_state.current_index + batch_size
        is_full = jax.lax.select(buffer_state.is_full, buffer_state.is_full,
                                 new_index >= self.max_length)
        can_sample = jax.lax.select(buffer_state.is_full, buffer_state.can_sample,
                                    new_index >= self.min_sample_length)
        current_index = new_index % self.max_length
        state = PrioritisedBufferState(data=AISData(x, log_w, log_q_old),
                                       current_index=current_index,
                                       is_full=is_full,
                                       can_sample=can_sample)
        return state


    def sample(self, buffer_state: PrioritisedBufferState, key: chex.PRNGKey,
               batch_size: int) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Return a batch of sampled data."""
        # we use nan's in the buffer state initialisation
        # max_index = jax.lax.select(buffer_state.is_full,
        #                            self.max_length, buffer_state.current_index)
        # probs = jnp.where(jnp.arange(self.max_length) < max_index, jnp.ones(self.max_length,),
        #                   jnp.zeros((self.max_length,)))
        probs = jax.nn.softmax(buffer_state.data.log_w, axis=0)
        indices = jax.random.choice(key, jnp.arange(self.max_length), shape=(batch_size,),
                                    replace=False, p=probs)
        x = buffer_state.data.x[indices]
        log_w = buffer_state.data.log_w[indices]
        log_q_old = buffer_state.data.log_q_old[indices]
        return x, log_w, log_q_old, indices


    def sample_n_batches(self, buffer_state: PrioritisedBufferState, key: chex.PRNGKey,
                         batch_size: int, n_batches: int) -> \
            Iterable[Tuple[chex.Array, chex.Array, chex.Array, chex.Array]]:
        """Returns dataset with n-batches on the leading axis."""
        x, log_w, log_q_old, indices = self.sample(buffer_state, key, batch_size*n_batches)
        dataset = jax.tree_map(lambda x: x.reshape((n_batches, batch_size, *x.shape[1:])),
                               (x, log_w, log_q_old, indices))
        return dataset

    def adjust(self, log_w_adjustment: chex.Array, log_q: chex.Array, indices: chex.Array,
               buffer_state: PrioritisedBufferState) \
            -> PrioritisedBufferState:
        """Adjust log weights and log q to match new value of theta, this is typically performed
        over minibatches, rather than over the whole dataset at once."""
        log_w = buffer_state.data.log_w.at[indices].set(buffer_state.data.log_w[indices] +
                                                 log_w_adjustment)
        new_data = AISData(x=buffer_state.data.x, log_w=log_w,
                           log_q_old=buffer_state.data.log_q_old.at[indices].set(log_q))
        return PrioritisedBufferState(data=new_data, current_index=buffer_state.current_index,
                                      can_sample=buffer_state.can_sample,
                                      is_full=buffer_state.is_full)



if __name__ == '__main__':
    # to check that the replay buffer runs
    dim = 5
    batch_size = 3
    n_batches_total_length = 10
    length = n_batches_total_length * batch_size
    min_sample_length = int(length * 0.5)
    rng_key = jax.random.PRNGKey(0)
    initial_sampler = lambda _: (jnp.zeros((batch_size, dim)), jnp.zeros(batch_size), jnp.zeros(
        batch_size,))
    buffer = PrioritisedReplayBuffer(dim, length, min_sample_length)
    buffer_state = buffer.init(rng_key, initial_sampler)
    n_batches = 3
    for i in range(100):
        buffer_state = buffer.add(jnp.zeros((batch_size, dim)), jnp.zeros(batch_size), jnp.zeros(
            batch_size),
                                  buffer_state)
        rng_key, subkey = jax.random.split(rng_key)
        x, log_w, log_q_old, indices = buffer.sample(buffer_state, subkey, batch_size)
        buffer_state = buffer.adjust(jnp.ones_like(log_w), jnp.ones_like(log_q_old), indices,
                                     buffer_state)

