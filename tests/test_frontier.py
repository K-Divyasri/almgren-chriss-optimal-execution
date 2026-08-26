import numpy as np

from acexec.frontier import efficient_frontier, half_life
from acexec.model import ExecutionParams


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=15, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)


def test_frontier_is_monotonic():
    """Higher risk (cost_std) should never buy you higher expected cost too --
    the frontier trades one off against the other, monotonically, in both directions."""
    p = _params()
    df = efficient_frontier(p)
    # sorted by cost_std descending (see efficient_frontier's docstring)
    assert np.all(np.diff(df["cost_std"]) <= 1e-9)
    assert np.all(np.diff(df["expected_cost"]) >= -1e-9)


def test_frontier_endpoints_are_twap_and_near_immediate():
    p = _params()
    df = efficient_frontier(p, lambdas=np.array([0.0, 1e-3]))
    twap_row = df[df["lam"] == 0.0].iloc[0]
    urgent_row = df[df["lam"] == 1e-3].iloc[0]
    assert twap_row["cost_std"] > urgent_row["cost_std"]
    assert twap_row["expected_cost"] < urgent_row["expected_cost"]


def test_half_life_shrinks_as_lam_grows():
    p = _params()
    hl_low = half_life(p, 1e-8)
    hl_high = half_life(p, 1e-4)
    assert hl_high <= hl_low
