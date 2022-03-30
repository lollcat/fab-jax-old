import chex
import haiku as hk
import jax.nn
import numpy as np
import jax.numpy as jnp
from typing import Sequence

class EncoderTorso(hk.Module):
    def __init__(self):
        super(EncoderTorso, self).__init__()
        self.conv1 = hk.Conv2D(output_channels=8, kernel_shape=(3, 3), stride=2)
        self.conv2 = hk.Conv2D(output_channels=8, kernel_shape=(3, 3), stride=2)
        self.mlp = hk.nets.MLP([256, ])

    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = self.conv1(x)
        x = jax.nn.relu((x))
        x = self.conv2(x)
        x = jax.nn.relu((x))
        if len(x.shape) == 4:  # with batch dim
            x = jnp.reshape(x, [x.shape[0], -1])  # [B, D]
            x = hk.Flatten()(x)
            return self.mlp(x)
        elif len(x.shape) == 3:
            x = jnp.reshape(x, [-1])  # [D]
            x = x[None, :]
            x = hk.Flatten()(x)
            return jnp.squeeze(self.mlp(x), axis=0)
        else:
            raise ValueError


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(
      self,
      output_shape: Sequence[int],
  ):
    super().__init__()
    self._pre_deconv_shape = (output_shape[0] // 4, output_shape[1] // 4, 5)
    self.mlp = hk.nets.MLP([256, np.prod(self._pre_deconv_shape)])

    self.deconv1 = hk.Conv2DTranspose(output_channels=8, kernel_shape=(3, 3), stride=2)
    self.deconv2 = hk.Conv2DTranspose(output_channels=1, kernel_shape=(3, 3), stride=2)
    self._output_shape = output_shape

  def __call__(self, z: chex.Array) -> chex.Array:
    logits = self.mlp(z)
    if len(z.shape) == 2:  # with batch
        logits = jnp.reshape(logits, (-1, *self._pre_deconv_shape))
    elif len(z.shape) == 1:
        logits = jnp.reshape(logits, self._pre_deconv_shape)
    else:
        raise ValueError
    logits = self.deconv1(logits)
    logits = jax.nn.relu((logits))
    logits = self.deconv2(logits)
    return logits