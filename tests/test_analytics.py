import numpy as np

from acexec.analytics import cost_in_bps, cost_variance, expected_cost, mean_variance_objective
from acexec.model import ExecutionParams
from acexec.trajectory import optimal_trajectory


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=10, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)


def test_expected_cost_positive():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    assert expected_cost(p, holdings) > 0


def test_twap_has_lowest_expected_cost_of_any_trajectory():
    """TWAP (lam=0) is, by construction, the expected-cost minimizer ignoring risk."""
    p = _params()
    twap = optimal_trajectory(p, lam=0.0)
    risk_averse = optimal_trajectory(p, lam=1e-5)
    assert expected_cost(p, twap) < expected_cost(p, risk_averse)


def test_twap_has_highest_variance():
    p = _params()
    twap = optimal_trajectory(p, lam=0.0)
    risk_averse = optimal_trajectory(p, lam=1e-5)
    assert cost_variance(p, twap) > cost_variance(p, risk_averse)


def test_mean_variance_objective_is_additive():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    lam = 1e-6
    assert np.isclose(
        mean_variance_objective(p, holdings, lam),
        expected_cost(p, holdings) + lam * cost_variance(p, holdings),
    )


def test_optimal_trajectory_beats_twap_on_its_own_objective():
    """The whole point: for lam>0, the AC trajectory scores better on U(x) than TWAP does."""
    p = _params()
    lam = 1e-6
    ac = optimal_trajectory(p, lam)
    twap = optimal_trajectory(p, lam=0.0)
    assert mean_variance_objective(p, ac, lam) < mean_variance_objective(p, twap, lam)


def test_cost_in_bps_scales_with_notional():
    p = _params()
    assert cost_in_bps(p, p.total_shares * p.arrival_price * 0.0001) == 1.0
