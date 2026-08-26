import numpy as np

from acexec.analytics import cost_variance, expected_cost
from acexec.model import ExecutionParams
from acexec.simulate import monte_carlo_summary
from acexec.trajectory import optimal_trajectory


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=10, sigma=0.95, eta=2.5e-6, gamma=2.5e-7)


def test_monte_carlo_mean_converges_to_closed_form_expected_cost():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    closed_form = expected_cost(p, holdings)
    mc = monte_carlo_summary(p, holdings, n_paths=20_000, seed=42)
    # within 1% of the closed-form value -- generous but non-trivial given MC noise
    assert abs(mc["mean_cost"] - closed_form) / abs(closed_form) < 0.01


def test_monte_carlo_std_converges_to_closed_form_cost_std():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    closed_form_std = np.sqrt(cost_variance(p, holdings))
    mc = monte_carlo_summary(p, holdings, n_paths=20_000, seed=42)
    assert abs(mc["std_cost"] - closed_form_std) / closed_form_std < 0.05


def test_more_paths_gets_closer_to_the_closed_form():
    p = _params()
    holdings = optimal_trajectory(p, lam=1e-6)
    closed_form = expected_cost(p, holdings)
    err_small = abs(monte_carlo_summary(p, holdings, n_paths=50, seed=1)["mean_cost"] - closed_form)
    err_large = abs(monte_carlo_summary(p, holdings, n_paths=20_000, seed=1)["mean_cost"] - closed_form)
    assert err_large < err_small
