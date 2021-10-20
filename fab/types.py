from typing import Callable, Any
import jax.numpy as jnp
from dataclasses import dataclass
import haiku as hk

XPoints = jnp.ndarray
LogProbs = jnp.ndarray
TargetLogProbFunc = Callable[[XPoints], LogProbs]
MCMCTransitionManager = Any
Params = Any

@dataclass
class HaikuDistribution:  # this distribution also takes in trainable parameters into all of it's
    # functions
    dim: int
    log_prob: hk.Transformed  # all log_prob.apply(params, x)
    sample_and_log_prob: hk.Transformed
    sample: hk.Transformed

if __name__ == '__main__':
    pass