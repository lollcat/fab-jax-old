from typing import Mapping, Callable
import chex

Batch = Mapping[str, chex.Array]
LogProbs = chex.Array
XPoints = chex.Array
LogProbFunc = Callable[[XPoints], LogProbs]