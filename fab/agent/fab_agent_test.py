from absl.testing import absltest
from fab.learnt_distributions.distrax_flo import make_rational_quadratic_spline_dist_funcs
from fab.target_distributions.double_well import DoubleWellEnergy
from fab.agent.fab_agent import AgentFAB


class Test_AgentFAB(absltest.TestCase):
    dim = 3
    flow_num_layers = 2
    quadratic_split_haiku_dist = make_rational_quadratic_spline_dist_funcs(
        x_ndim=dim, flow_num_layers=flow_num_layers)
    target_log_prob = DoubleWellEnergy(dim=dim).log_prob
    batch_size = 32
    n_iter = 100
    n_intermediate_distributions: int = 3
    AIS_kwargs = {"additional_transition_operator_kwargs": {"step_tuning_method": "p_accept"}}

    fab_agent = AgentFAB(learnt_distribution=quadratic_split_haiku_dist,
                         target_log_prob=target_log_prob,
                         batch_size=batch_size, n_iter=n_iter,
                         n_intermediate_distributions=n_intermediate_distributions,
                         AIS_kwargs=AIS_kwargs)

    def test_fab_agent(self):
        self.fab_agent.run()