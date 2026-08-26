"""Deriving the optimal trajectory — the actual mathematics of the paper.

The trader chooses how many shares to hold at each grid point, x_0, x_1, ..., x_N,
with x_0 = X (the full order) and x_N = 0 (fully done). Trading in interval k
means selling n_k = x_{k-1} - x_k shares. Two costs come from this choice:

  Expected cost   E[x] = (1/2) gamma X^2 + (eta_tilde / tau) * sum_{k=1}^{N} n_k^2
                  (a fixed cost from permanent impact on the whole order, plus a
                  cost that's QUADRATIC in how fast you trade each interval —
                  trading twice as fast in one slice costs 4x as much temporary
                  impact in that slice).

  Cost variance   V[x] = sigma^2 * sum_{k=1}^{N-1} x_k^2
                  (you're exposed to price risk on whatever you're still holding;
                  holding more, for longer, means more variance).

  eta_tilde = eta - gamma*tau/2 — selling under permanent impact effectively
  gives back half an interval's worth of temporary-impact cost, because the
  price drop you cause also lowers what the "no-impact" benchmark would have
  cost you. See `model.ExecutionParams.eta_tilde`.

The trader minimizes U(x) = E[x] + lam * V[x] for a chosen risk aversion `lam`.
This is a plain calculus problem: U is quadratic in the free variables
x_1, ..., x_{N-1} (x_0 and x_N are fixed), so setting dU/dx_j = 0 for each
interior j gives the optimum. Differentiating:

    dU/dx_j = (2 eta_tilde/tau) * (2 x_j - x_{j-1} - x_{j+1}) + 2 lam sigma^2 tau x_j = 0

Rearranging (dividing by 2 eta_tilde/tau):

    x_{j+1} - 2 x_j + x_{j-1} = kappa_tilde^2 * tau^2 * x_j,     kappa_tilde^2 = lam * sigma^2 / eta_tilde

This is a linear, constant-coefficient difference equation — the discrete analog
of a second-derivative ODE, x'' = kappa^2 x, whose solutions are exponentials.
Trying x_j = A z^j turns it into the characteristic equation
z^2 - (2 + kappa_tilde^2 tau^2) z + 1 = 0, whose roots are z = e^{±kappa tau} for
a kappa solving cosh(kappa tau) = 1 + kappa_tilde^2 tau^2 / 2 (a real, positive
kappa exists whenever lam > 0). The general solution built from those two roots,
fitted to the boundary conditions x_0 = X and x_N = 0, collapses to a single
clean closed form:

    x_j = X * sinh(kappa (T - t_j)) / sinh(kappa T)

`optimal_trajectory` below computes exactly this. `verify_optimal.py` checks it
against an *independent* numerical minimization of U(x) that never uses this
formula at all — the point of a paper replication is not to trust algebra on
faith.
"""

from __future__ import annotations

import numpy as np

from .model import ExecutionParams


def kappa_tilde_squared(params: ExecutionParams, lam: float) -> float:
    """kappa_tilde^2 = lam * sigma^2 / eta_tilde — the urgency-squared driving the ODE."""
    if lam <= 0:
        return 0.0
    return lam * params.sigma**2 / params.eta_tilde


def kappa(params: ExecutionParams, lam: float) -> float:
    """Solve cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2 for kappa >= 0.

    lam=0 is the degenerate case (no risk aversion at all): the difference
    equation becomes x_{j+1} - 2x_j + x_{j-1} = 0, whose solution is linear in
    j, not hyperbolic. We return kappa=0 and let `optimal_trajectory` special-case
    it, since sinh(0*something)/sinh(0*something) is 0/0.
    """
    kt2 = kappa_tilde_squared(params, lam)
    if kt2 <= 0:
        return 0.0
    arg = 1.0 + 0.5 * kt2 * params.tau**2
    return float(np.arccosh(arg) / params.tau)


def optimal_trajectory(params: ExecutionParams, lam: float) -> np.ndarray:
    """The closed-form optimal holdings x_0, ..., x_N for risk aversion `lam`.

    lam=0 (risk-neutral: minimize expected cost alone, ignore variance) collapses
    to the linear trajectory x_j = X * (1 - j/N) — this is exactly TWAP (trade
    the same amount every interval). Increasing lam front-loads the trajectory:
    a more risk-averse trader accepts higher expected impact cost to cut down
    how long — and how much — they're exposed to price risk. See notebook 03 for
    this collapsing-to-TWAP limit shown numerically, and knowledge/02 for why.
    """
    X = params.total_shares
    N = params.n_intervals
    t = params.times()
    T = params.horizon

    if lam <= 0:
        return X * (1.0 - t / T)

    k = kappa(params, lam)
    return X * np.sinh(k * (T - t)) / np.sinh(k * T)


def trades_from_holdings(holdings: np.ndarray) -> np.ndarray:
    """n_k = x_{k-1} - x_k for k = 1..N — shares actually traded in each interval."""
    return -np.diff(holdings)
