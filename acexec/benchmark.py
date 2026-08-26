"""TWAP and VWAP, self-contained, for the comparison the roadmap asks for.

`29-execution-algo-simulator/` (elsewhere in this repo) already implements a
full TWAP/VWAP/POV execution simulator with its own Monte Carlo transaction-cost
analysis — reuse that project if you want the fuller practical tool. The two
small functions here exist only so THIS project's labs and notebooks can put
Almgren-Chriss side by side with TWAP and VWAP without importing across
project folders (each project here ships to its own repo).

The interesting relationship, covered in `knowledge/02_market_impact_and_the_ac_model.md`:
TWAP is not just "a different, simpler algorithm" — it is the exact lam=0
special case of Almgren-Chriss (see `trajectory.optimal_trajectory`). VWAP is
NOT a special case of anything here: it tracks a volume curve for a completely
different reason (to minimize market-impact signaling and track a specific
benchmark), a goal orthogonal to the pure cost-vs-risk trade-off Almgren-Chriss
optimizes. Comparing them honestly means comparing tools built for different jobs,
not picking a "winner."
"""

from __future__ import annotations

import numpy as np

from .model import ExecutionParams


def twap_holdings(params: ExecutionParams) -> np.ndarray:
    """Equal-sized trades every interval — the linear trajectory."""
    return params.total_shares * (1.0 - params.times() / params.horizon)


def intraday_volume_profile(n_intervals: int) -> np.ndarray:
    """A classic U-shaped intraday volume curve (heavy at the open/close, light
    midday), normalized to sum to 1. VWAP schedules trade in proportion to this."""
    x = np.linspace(0.0, 1.0, n_intervals)
    curve = 1.0 + 1.8 * (2.0 * x - 1.0) ** 2
    return curve / curve.sum()


def vwap_holdings(params: ExecutionParams) -> np.ndarray:
    """Holdings implied by trading proportionally to the expected volume profile."""
    profile = intraday_volume_profile(params.n_intervals)
    trades = params.total_shares * profile
    remaining = params.total_shares - np.cumsum(trades)
    return np.concatenate(([params.total_shares], remaining))
