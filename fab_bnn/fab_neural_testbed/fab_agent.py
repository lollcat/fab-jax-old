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


class FabSampler(testbed_base.EpistemicSampler):
    def __call__(self, x: chex.Array, key: chex.PRNGKey) -> chex.Array:
        """Generate a random sample from approximate posterior distribution."""


class FabAgent(testbed_base.TestbedAgent):
    def __call__(self, data: testbed_base.Data, prior: testbed_base.PriorKnowledge) -> \
            FabSampler:
        """Sets up a training procedure given ENN prior knowledge."""