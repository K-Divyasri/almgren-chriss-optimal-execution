import numpy as np

from acexec.model import ExecutionParams
from acexec.trajectory import kappa, optimal_trajectory, trades_from_holdings


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=10, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)


def test_trajectory_hits_boundary_conditions():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    assert holdings[0] == p.total_shares
    assert abs(holdings[-1]) < 1e-6


def test_lam_zero_is_exactly_linear_twap():
    p = _params()
    holdings = optimal_trajectory(p, lam=0.0)
    expected = p.total_shares * (1.0 - np.arange(p.n_intervals + 1) / p.n_intervals)
    np.testing.assert_allclose(holdings, expected)


def test_kappa_zero_when_lam_zero():
    p = _params()
    assert kappa(p, 0.0) == 0.0


def test_kappa_increases_with_lam():
    p = _params()
    k1 = kappa(p, 1e-7)
    k2 = kappa(p, 1e-5)
    assert k2 > k1 > 0


def test_higher_lam_front_loads_the_trajectory():
    """A more risk-averse trader holds FEWER shares at every interior point."""
    p = _params()
    low = optimal_trajectory(p, lam=1e-7)
    high = optimal_trajectory(p, lam=1e-5)
    assert np.all(high[1:-1] <= low[1:-1])
    assert np.any(high[1:-1] < low[1:-1])


def test_holdings_are_monotonically_decreasing():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    assert np.all(np.diff(holdings) <= 0)


def test_trades_sum_to_total_shares():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    trades = trades_from_holdings(holdings)
    assert np.isclose(trades.sum(), p.total_shares)
    assert np.all(trades >= -1e-9)


def test_very_large_lam_front_loads_almost_everything_into_first_interval():
    p = _params()
    holdings = optimal_trajectory(p, lam=1.0)  # huge risk aversion for these units
    trades = trades_from_holdings(holdings)
    assert trades[0] / p.total_shares > 0.9
