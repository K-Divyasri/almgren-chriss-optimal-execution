"""The efficient frontier — the paper's headline picture.

For every risk aversion `lam`, there's one optimal trajectory, with one expected
cost and one cost variance. Sweep `lam` from 0 to a large number and each point
traces out a curve: the menu of honest choices between "cheap but risky" and
"expensive but safe." You cannot do better than this curve — it IS the
definition of optimal for every risk aversion on it. That curve is what
Almgren-Chriss (2000) calls the efficient frontier of execution.

Two boundary cases anchor the whole curve, and both are checked in
`tests/test_frontier.py`:

  lam = 0     -> the linear trajectory, i.e. TWAP. Cheapest possible expected
                 cost, but the most time exposed to price risk (highest variance).
  lam -> large -> near-immediate liquidation. Almost the entire order trades in
                 the first interval or two — the highest possible impact cost,
                 but almost no time held, so almost no variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import cost_std, expected_cost
from .model import ExecutionParams
from .trajectory import optimal_trajectory


def frontier_point(params: ExecutionParams, lam: float) -> dict:
    holdings = optimal_trajectory(params, lam)
    return {
        "lam": lam,
        "expected_cost": expected_cost(params, holdings),
        "cost_std": cost_std(params, holdings),
    }


def efficient_frontier(params: ExecutionParams, lambdas: np.ndarray | None = None) -> pd.DataFrame:
    """One row per risk aversion level: (lam, expected_cost, cost_std).

    Sorted by cost_std ascending, since that's how the frontier is usually drawn
    (risk on the x-axis, cost on the y-axis, both non-decreasing along the curve
    as `lam` rises — see `tests/test_frontier.py::test_frontier_is_monotonic`).
    """
    if lambdas is None:
        lambdas = np.concatenate(([0.0], np.logspace(-8, -4, 24)))
    rows = [frontier_point(params, float(lam)) for lam in lambdas]
    df = pd.DataFrame(rows)
    return df.sort_values("cost_std", ascending=False).reset_index(drop=True)


def half_life(params: ExecutionParams, lam: float) -> float:
    """Intervals needed to sell half the order — a one-number read on urgency."""
    holdings = optimal_trajectory(params, lam)
    X = holdings[0]
    sold = X - holdings
    idx = np.searchsorted(sold, 0.5 * X)
    return float(idx)
