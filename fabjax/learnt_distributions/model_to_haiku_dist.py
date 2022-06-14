from typing import Callable, Tuple

import distrax
import haiku as hk
from fabjax.types import XPoints, LogProbs, HaikuDistribution

GetModelFn = Callable[[], distrax.Distribution]

def model_to_haiku_dist(get_model_fn: GetModelFn, x_dim) -> HaikuDistribution:

    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: XPoints) -> LogProbs:
        model = get_model_fn()
        return model.log_prob(data)


    @hk.transform
    def sample_and_log_prob(sample_shape: Tuple = ()) \
            -> Tuple[XPoints, LogProbs]:
        model = get_model_fn()
        return model.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


    @hk.transform
    def sample(sample_shape: Tuple = ()) -> XPoints:
        model = get_model_fn()
        return model.sample(seed=hk.next_rng_key(), sample_shape=sample_shape)

    return HaikuDistribution(x_dim, log_prob, sample_and_log_prob, sample)