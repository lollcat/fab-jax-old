from typing import Callable, Any
import jax.numpy as jnp
from dataclasses import dataclass
import haiku as hk

XPoints = jnp.ndarray
LogProbs = jnp.ndarray
LogProbFunc = Callable[[XPoints], LogProbs]
MCMCTransitionManager = Any
Params = Any
# TODO: rather define init, and then log prob like this
# GetLogProb = Callable[[Params, XPoints], LogProbs]

@dataclass
class HaikuDistribution:  # this distribution also takes in trainable parameters into all of it's
    # functions
    dim: int
    log_prob: hk.Transformed  # all log_prob.apply(params, x)
    sample_and_log_prob: hk.Transformed
    sample: hk.Transformed
    z_log_prob: hk.Transformed
    z_sample_and_log_prob: hk.Transformed  # for initialising ais chain in latent space
    z_x_and_log_det_forward: hk.Transformed  # for calculating target log prob

if __name__ == '__main__':
    pass