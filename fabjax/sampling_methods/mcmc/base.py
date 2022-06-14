import abc

from typing import Tuple
import chex
from fabjax.types import LogProbFunc


X = chex.Array
TransitionOperatorState = chex.ArrayTree
AuxTransitionInfo = chex.ArrayTree

class TransitionOperator(abc.ABC):

    @abc.abstractmethod
    def run(self,
            key: chex.PRNGKey,
            transition_operator_state: chex.ArrayTree,
            x: chex.Array,
            i: chex.Array,
            transition_target_log_prob: LogProbFunc) -> \
            Tuple[X, TransitionOperatorState, AuxTransitionInfo]:
        """
        Run mcmc transition towards transition_target_prob
        """

    @abc.abstractmethod
    def get_init_state(self) -> chex.ArrayTree:
        """Returns the initial state of the transition operator."""