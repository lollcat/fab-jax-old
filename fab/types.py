from typing import Callable
import jax.numpy as jnp
from dataclasses import dataclass
import haiku as hk

XPoints = jnp.ndarray
LogProbs = jnp.ndarray
TargetLogProbFunc = Callable[[XPoints], LogProbs]


@dataclass
class HaikuDistribution:  # this distribution also takes in trainable parameters into all of it's
    # functions
    log_prob: hk.Transformed
    sample_and_log_prob: hk.Transformed
    sample: hk.Transformed

if __name__ == '__main__':
    pass