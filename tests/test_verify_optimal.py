from acexec.model import ExecutionParams
from acexec.verify_optimal import closed_form_matches_numerical, numerically_optimal_trajectory


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=8, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)


def test_closed_form_matches_numerical_optimum_moderate_lam():
    p = _params()
    ok, max_diff = closed_form_matches_numerical(p, lam=1e-6, rtol=1e-3)
    assert ok, f"max relative diff was {max_diff:.2e}"


def test_closed_form_matches_numerical_optimum_low_lam():
    p = _params()
    ok, max_diff = closed_form_matches_numerical(p, lam=1e-8, rtol=1e-3)
    assert ok, f"max relative diff was {max_diff:.2e}"


def test_closed_form_matches_numerical_optimum_high_lam():
    p = _params()
    ok, max_diff = closed_form_matches_numerical(p, lam=1e-4, rtol=1e-3)
    assert ok, f"max relative diff was {max_diff:.2e}"


def test_numerical_trajectory_also_respects_boundary_conditions():
    p = _params()
    holdings = numerically_optimal_trajectory(p, lam=1e-6)
    assert abs(holdings[0] - p.total_shares) < 1.0
    assert abs(holdings[-1]) < 1.0
