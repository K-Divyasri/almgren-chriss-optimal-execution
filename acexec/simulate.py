"""Monte Carlo simulation — the empirical check on the closed-form formulas.

`analytics.py` computes expected cost and cost variance algebraically, straight
from the trajectory, no randomness involved. This file does the opposite: draw
thousands of random price paths, actually execute the trajectory against each
one, and measure the realized cost. As the number of paths grows, the measured
average and variance should converge to `analytics.py`'s closed-form numbers —
that convergence IS the Law of Large Numbers doing its job, and it's a genuine,
useful sanity check that the closed form actually describes the same random
process the simulation does, not two unrelated calculations that happen to have
similar-looking formulas.

Price/execution model for one simulated path (a direct implementation of the
model described in `model.py`'s docstring, not borrowed from anywhere):

    S_k = S_{k-1} + sigma * Z_k - gamma * n_k      (mid-price: random walk + permanent impact)
    fill_price_k = S_{k-1} - (eta / tau) * n_k     (you execute worse than last mid by temporary impact)

    realized cost = X * S_0 - sum_k n_k * fill_price_k   (shortfall vs. selling all X at arrival price)
"""

from __future__ import annotations

import numpy as np

from .model import ExecutionParams
from .trajectory import trades_from_holdings


def simulate_shocks(params: ExecutionParams, n_paths: int, seed: int = 7) -> np.ndarray:
    """(n_paths, N) array of iid N(0, sigma) price innovations, one per interval."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, params.sigma, size=(n_paths, params.n_intervals))


def realized_cost(params: ExecutionParams, holdings: np.ndarray, shocks: np.ndarray) -> float:
    """Realized cost for ONE path, given that path's shocks (shape (N,))."""
    n = trades_from_holdings(holdings)
    S = np.empty(params.n_intervals + 1)
    S[0] = params.arrival_price
    for k in range(1, params.n_intervals + 1):
        S[k] = S[k - 1] + shocks[k - 1] - params.gamma * n[k - 1]
    fill_prices = S[:-1] - (params.eta / params.tau) * n
    proceeds = float(np.sum(n * fill_prices))
    return params.total_shares * params.arrival_price - proceeds


def simulate_costs(params: ExecutionParams, holdings: np.ndarray, n_paths: int, seed: int = 7) -> np.ndarray:
    """Realized cost for every path — an (n_paths,) array, vectorized over paths."""
    shocks = simulate_shocks(params, n_paths, seed=seed)
    n = trades_from_holdings(holdings)
    S = np.empty((n_paths, params.n_intervals + 1))
    S[:, 0] = params.arrival_price
    for k in range(1, params.n_intervals + 1):
        S[:, k] = S[:, k - 1] + shocks[:, k - 1] - params.gamma * n[k - 1]
    fill_prices = S[:, :-1] - (params.eta / params.tau) * n
    proceeds = fill_prices @ n
    return params.total_shares * params.arrival_price - proceeds


def monte_carlo_summary(params: ExecutionParams, holdings: np.ndarray, n_paths: int, seed: int = 7) -> dict:
    costs = simulate_costs(params, holdings, n_paths, seed=seed)
    return {
        "n_paths": n_paths,
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs, ddof=1)),
    }
