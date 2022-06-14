import abc
from typing import Dict, Tuple

import chex

X = chex.Array
LogWeights = chex.Array
TransitionOperatorState = chex.ArrayTree
Info = Dict[str, chex.Array]

class AnnealedImportanceSamplerBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self,
                 dim: int,
                 n_intermediate_distributions: int,
                 *args,
                 **kwargs
                 ):
        """Initialise AIS"""

    @abc.abstractmethod
    def run(self, x_base, log_prob_p0, key, transition_operator_state,
            base_log_prob, target_log_prob) -> Tuple[X, LogWeights, TransitionOperatorState, Info]:
        """Perform AIS"""


    @abc.abstractmethod
    def get_init_state(self):
        """Returns initial state of ais transition operator."""

