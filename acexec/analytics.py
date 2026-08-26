"""Pricing a trajectory: expected cost and cost variance, in closed form.

No simulation anywhere in this file. Both formulas are exact, algebraic
functions of the trajectory `x` — the same E[x] and V[x] the trajectory was
derived to minimize a weighted sum of (see trajectory.py's docstring). Being
able to compute the efficient frontier without running a single Monte Carlo
path is the whole reason Almgren-Chriss became the textbook model: the
trade-off is a formula, not an empirical measurement.

`simulate.py` exists to CHECK these formulas against a real Monte Carlo
simulation, not to replace them.
"""

from __future__ import annotations

import numpy as np

from .model import ExecutionParams
from .trajectory import trades_from_holdings


def expected_cost(params: ExecutionParams, holdings: np.ndarray) -> float:
    """E[x] = (1/2) gamma X^2 + (eta_tilde / tau) * sum(n_k^2).

    The first term is a fixed cost every strategy pays equally (you permanently
    move the price by trading X shares total, no matter how you spread it out)
    — it does not depend on the trajectory's shape, only on X. The second term
    is what the trajectory actually controls: trading in bigger, faster bursts
    costs quadratically more temporary impact.
    """
    X = params.total_shares
    n = trades_from_holdings(holdings)
    fixed = 0.5 * params.gamma * X**2
    variable = (params.eta_tilde / params.tau) * float(np.sum(n**2))
    return fixed + variable


def cost_variance(params: ExecutionParams, holdings: np.ndarray) -> float:
    """V[x] = sigma^2 * sum_{k=1}^{N-1} x_k^2 — variance from price risk on
    whatever's still held. x_0 (not yet exposed to any interval's move) and
    x_N (=0, nothing left to be exposed) are excluded from the sum."""
    held = holdings[1:-1]
    return float(params.sigma**2 * np.sum(held**2))


def cost_std(params: ExecutionParams, holdings: np.ndarray) -> float:
    return float(np.sqrt(cost_variance(params, holdings)))


def mean_variance_objective(params: ExecutionParams, holdings: np.ndarray, lam: float) -> float:
    """U(x) = E[x] + lam * V[x] — the exact quantity `optimal_trajectory` minimizes."""
    return expected_cost(params, holdings) + lam * cost_variance(params, holdings)


def cost_in_bps(params: ExecutionParams, cost_dollars: float) -> float:
    """Express a dollar cost as basis points of the order's notional value."""
    notional = params.total_shares * params.arrival_price
    return cost_dollars / notional * 1e4
