from typing import Any, Iterator, Mapping, NamedTuple, Sequence, Tuple, Dict, Callable

import chex
import haiku as hk

Info = Mapping[str, chex.Array]

class Params(NamedTuple):
  encoder: chex.ArrayTree
  decoder: chex.ArrayTree


class State(NamedTuple):
  params: Params
  opt_state: chex.ArrayTree
  transition_operator_state: chex.ArrayTree
  rng_key: chex.PRNGKey


class EncoderNetworks(NamedTuple):
  sample_and_log_prob: hk.Transformed
  log_prob: hk.Transformed


class VAENetworks(NamedTuple):
  init: Callable[[chex.PRNGKey], Params]
  encoder_network: EncoderNetworks
  prior_log_prob: Callable[[chex.Array], chex.Array]
  decoder_log_prob: hk.Transformed


class AISOutput(NamedTuple):
  z_ais: chex.Array
  log_w_ais: chex.Array
  transition_operator_state: chex.ArrayTree
  info: Dict[str, chex.Array]