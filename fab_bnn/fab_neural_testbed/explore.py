from typing import NamedTuple

import chex
import jax.random
from enn import base as enn_base
from enn import networks
from neural_testbed import base as testbed_base
from neural_testbed import generative
from neural_testbed.agents.factories import preconditioner as pre
from neural_testbed.agents.factories import sgld_optimizer
from neural_testbed.agents.factories import utils
import numpy as np


class FabConfig(NamedTuple):
  num_hidden: int


config = FabConfig(10)


def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
    return networks.make_einsum_ensemble_mlp_enn(
        output_sizes=[config.num_hidden, config.num_hidden, prior.num_classes],
        num_ensemble=1,
        nonzero_bias=False,
    )





if __name__ == '__main__':
    from neural_testbed import leaderboard
    problem = leaderboard.problem_from_id('classification_2d/75')
    enn = make_enn(problem.prior_knowledge)
    key = jax.random.PRNGKey(0)
    x = problem.train_data.x
    y = problem.train_data.y
    params_init = enn.init(key, x, y)
    y = enn.apply(params_init, x, 0)


    def enn_sampler(x: chex.Array, key: chex.PRNGKey) -> chex.Array:
        del key  # key does not affect the agent.
        return enn.apply(params_init, x, 0)


    plots = generative.sanity_plots(problem, enn_sampler)
    p = plots['more_enn']
    _ = p.draw()

