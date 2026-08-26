"""The execution problem: what you're trading, and the market you're trading it into.

Almgren-Chriss makes two simplifying assumptions that turn "sell a million shares
without wrecking the price" into a solvable calculus problem:

  1. The price follows an arithmetic random walk (each interval, the price moves by
     a Normal shock with standard deviation `sigma` — no drift, no fat tails).
  2. Market impact is LINEAR in your trading rate, split into two pieces:
       temporary impact  h(v) = eta * v   — you pay this on the shares you trade
                                             THIS interval; the market recovers
                                             right after.
       permanent impact  g(v) = gamma * v — every share you trade shifts the
                                             price for good, for every share
                                             still unsold.

Neither assumption is exactly true of a real market. Both are true enough, and
simple enough, to get a closed-form answer instead of a black-box numerical
solver — which is exactly why this is the model every execution-algo interview
question traces back to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExecutionParams:
    """One parent order, plus the market/impact parameters that define its cost.

    All the roadmap's "existing TWAP/VWAP/POV execution simulator" (project 29,
    `29-execution-algo-simulator/`) needs is a schedule; this project additionally
    needs `sigma` (price volatility) and the two impact coefficients because it's
    deriving and pricing the trade-off, not just executing a fixed plan.
    """

    total_shares: float = 1_000_000.0   # X: shares to liquidate (sell) or acquire
    n_intervals: int = 20               # N: number of equal-length trading slices
    tau: float = 1.0                    # length of one interval (time units of your choice)
    sigma: float = 0.95                 # per-interval price volatility, in price units
    eta: float = 2.5e-6                 # temporary-impact coefficient (price / (shares/tau))
    gamma: float = 2.5e-7               # permanent-impact coefficient (price / share)
    arrival_price: float = 50.0         # S0, the decision-time benchmark price

    @property
    def horizon(self) -> float:
        """T = N * tau, the total time allotted to finish trading."""
        return self.n_intervals * self.tau

    @property
    def eta_tilde(self) -> float:
        """The impact coefficient actually used in the trajectory formula.

        Selling shares under permanent impact "gives away" gamma * tau / 2 of the
        temporary-impact cost for free (the permanent price drop you cause also
        lowers what you'd have paid anyway) — see knowledge/03 for the derivation.
        This only matters for the SHAPE of the optimal trajectory; the paper's
        variance and total-cost formulas use `eta` directly.
        """
        return self.eta - 0.5 * self.gamma * self.tau

    def times(self) -> np.ndarray:
        """The N+1 grid points t_0=0, t_1, ..., t_N=T."""
        return np.arange(self.n_intervals + 1) * self.tau
