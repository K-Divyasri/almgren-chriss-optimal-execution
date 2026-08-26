"""Plots: the trajectory shape and the efficient frontier — the paper's two figures."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_trajectories(params, lambdas: list[float], ax=None):
    """Overlay several holdings trajectories on one axis, one per risk aversion."""
    import matplotlib.pyplot as plt

    from .trajectory import optimal_trajectory

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    t = params.times()
    for lam in lambdas:
        holdings = optimal_trajectory(params, lam)
        label = "TWAP (lam=0)" if lam <= 0 else f"lam={lam:g}"
        ax.plot(t, holdings, marker="o", markersize=3, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("shares still held")
    ax.set_title("Optimal holdings trajectories at different risk aversions")
    ax.legend()
    return ax


def plot_frontier(frontier_df: pd.DataFrame, ax=None):
    """The efficient frontier: cost std (risk) on x, expected cost on y."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frontier_df["cost_std"], frontier_df["expected_cost"], marker="o")
    ax.set_xlabel("cost standard deviation (timing risk, $)")
    ax.set_ylabel("expected cost ($)")
    ax.set_title("Almgren-Chriss efficient frontier")
    return ax


def frontier_table_bps(frontier_df: pd.DataFrame, params) -> pd.DataFrame:
    """The frontier expressed in basis points of notional, the units a trading
    desk actually reports in, rather than raw dollars."""
    from .analytics import cost_in_bps

    out = frontier_df.copy()
    out["expected_cost_bps"] = out["expected_cost"].apply(lambda c: cost_in_bps(params, c))
    out["cost_std_bps"] = out["cost_std"].apply(lambda c: cost_in_bps(params, c))
    return out[["lam", "expected_cost_bps", "cost_std_bps"]]
