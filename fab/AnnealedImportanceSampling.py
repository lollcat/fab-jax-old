import distrax
from types import TargetLogProbFunc

class AnnealedImportanceSampler:
    def __init__(self, learnt_distribution: distrax.Distribution,
                 target_log_prob: TargetLogProbFunc):
        self.learnt_distribution = learnt_distribution
        self.target_log_prob = target_log_prob
