from typing import Any, Iterator, Mapping, NamedTuple, Sequence, Tuple, Dict, Callable

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from tqdm import tqdm

from fab_vae.models.nets import DecoderConv, EncoderTorsoConv, EncoderTorsoMLP, DecoderMLP
from fab_vae.learnt_distributions.real_nvp import RealNVP
from fab_vae.utils.data import load_dataset, Batch, MNIST_IMAGE_SHAPE
from fab_vae.sampling_methods.annealed_importance_sampling import AnnealedImportanceSampler
from fab_vae.models.fab_types import VAENetworks, EncoderNetworks, Params



class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, latent_size: int, use_flow: bool, use_conv: bool):
    super().__init__()
    self._encoder_torso = EncoderTorsoConv() if use_conv else EncoderTorsoMLP()
    self._latent_size = latent_size
    self.use_flow = use_flow
    if self.use_flow:
        self._flow_transform = RealNVP(x_ndim=latent_size, flow_num_layers=2)

  def __call__(self, x: jnp.ndarray) -> distrax.Distribution:
    x = self._encoder_torso(x)
    mean = hk.Linear(self._latent_size)(x)
    log_stddev = hk.Linear(self._latent_size)(x)
    h = hk.Linear(self._latent_size)(x)
    stddev = jnp.exp(log_stddev)
    base_dist = distrax.MultivariateNormalDiag(
        loc=mean, scale_diag=stddev)
    if self.use_flow:
        flow_transform = self._flow_transform(h)
        dist = distrax.Transformed(distribution=base_dist, bijector=flow_transform)
        return dist
    else:
        return base_dist


def make_vae_networks(latent_size: int,
                      output_shape: chex.Shape,
                      use_flow: bool = True,
                      use_conv: bool = True) -> VAENetworks:
    Decoder = DecoderConv if use_conv else DecoderMLP
    prior_z = distrax.MultivariateNormalDiag(
        loc=jnp.zeros((latent_size,)),
        scale_diag=jnp.ones((latent_size,)))

    @hk.without_apply_rng
    @hk.transform
    def encoder_log_prob(x, z):
        encoder = Encoder(latent_size, use_flow, use_conv)
        dist = encoder(x)
        return dist.log_prob(z)

    @hk.transform
    def encoder_sample_and_log_prob(x, sample_shape):
        encoder = Encoder(latent_size, use_flow, use_conv)
        dist = encoder(x)
        return dist.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)

    @hk.without_apply_rng
    @hk.transform
    def decoder_log_prob(x, z):
        logits = Decoder(output_shape)(z)
        likelihood_distrib = distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(output_shape))  # 3 non-batch dims
        return likelihood_distrib.log_prob(x)

    @hk.without_apply_rng
    @hk.transform
    def decoder_forward(z):
        logits = Decoder(output_shape)(z)
        likelihood_distrib = distrax.Independent(
            distrax.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=len(output_shape))  # 3 non-batch dims
        return likelihood_distrib

    def init(rng_key: chex.PRNGKey) -> Params:
        key1, key2 = jax.random.split(rng_key)
        dummy_x = jnp.zeros(output_shape)
        dummy_z = jnp.zeros(latent_size)
        encoder_params = encoder_log_prob.init(key1, dummy_x, dummy_z)
        decoder_params = decoder_log_prob.init(key2, dummy_x, dummy_z)
        return Params(encoder_params, decoder_params)

    encoder_net = EncoderNetworks(sample_and_log_prob=encoder_sample_and_log_prob,
                                  log_prob=encoder_log_prob)
    network = VAENetworks(encoder_network=encoder_net,
                          decoder_log_prob=decoder_log_prob,
                          prior_log_prob=prior_z.log_prob,
                          init=init,
                          prior_sample_and_log_prob=prior_z.sample_and_log_prob,
                          decoder_forward=decoder_forward)
    return network




