from typing import NamedTuple

import chex
from enn import base as enn_base
from enn import networks
from neural_testbed import base as testbed_base
from neural_testbed.agents.factories import preconditioner as pre
from neural_testbed.agents.factories import sgld_optimizer
from neural_testbed.agents.factories import utils
import numpy as np

from fab.types import HaikuDistribution
from fab.sampling_methods import AnnealedImportanceSampler

class FabConfig(NamedTuple):
  num_hidden: int


# ENN sampler for MCMC
def make_enn_sampler(flow_network: HaikuDistribution,
                     ais: AnnealedImportanceSampler,
                     params: chex.ArrayTree,
                     enn_network: enn_base.EpistemicNetwork,
                     use_ais: bool) -> testbed_base.EpistemicSampler:
  """ENN sampler for MCMC."""
  def enn_sampler(x: enn_base.Array, key: chex.PRNGKey) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
  return enn_sampler  # jax.jit(enn_sampler)



def agent_factor(config: FabConfig):
  # Create the model

  def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
      output_sizes=[config.num_hidden, config.num_hidden, prior.num_classes],
      num_ensemble=1,
      nonzero_bias=False,
    )

  # Configure the model and start training

  # model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=0)


  return enn_sampler