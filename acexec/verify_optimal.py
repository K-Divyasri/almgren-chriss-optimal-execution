"""Independent proof that the closed-form trajectory really is optimal.

`trajectory.optimal_trajectory` comes from solving a difference equation with
sinh and cosh. That's real calculus, not a guess — but a paper replication
should not just trust its own algebra. This file finds the minimizing
trajectory a *completely different way* — generic numerical optimization, with
no sinh/cosh/kappa anywhere in it — and checks the two agree.

If they didn't agree, one of two things would be true: a mistake in the
derivation, or a mistake in the numerical setup. Either way, that's a real bug
worth finding before trusting the "closed form" for anything downstream — this
is what "verify a formula independently" means in practice, not just a phrase.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .analytics import mean_variance_objective
from .model import ExecutionParams
from .trajectory import optimal_trajectory


def numerically_optimal_trajectory(
    params: ExecutionParams, lam: float, x0: np.ndarray | None = None
) -> np.ndarray:
    """Minimize U(x) = E[x] + lam*V[x] over the free interior holdings x_1..x_{N-1}
    with a generic gradient-based optimizer (L-BFGS-B), holding x_0=X and x_N=0 fixed.

    `x0` is the optimizer's starting guess for the interior holdings; defaults to
    the linear (TWAP) trajectory, which is a reasonable guess but NOT the answer
    for lam > 0 — the optimizer has to actually move away from it.

    The gradient passed to the optimizer is `dU/dx_j` differentiated directly from
    U(x) itself (plain calculus on the same objective `mean_variance_objective`
    computes) — it never touches kappa, sinh, or anything from `trajectory.py`.
    Without it, L-BFGS-B has to estimate the gradient by finite differences, which
    converges too loosely near the far (small-holdings) end of the trajectory to
    tell a correct closed form apart from a subtly wrong one — the whole point of
    this check.
    """
    X = params.total_shares
    N = params.n_intervals

    if x0 is None:
        x0 = X * (1.0 - np.arange(1, N) / N)  # interior points of the linear trajectory

    def objective(interior: np.ndarray) -> float:
        holdings = np.concatenate(([X], interior, [0.0]))
        return mean_variance_objective(params, holdings, lam)

    def gradient(interior: np.ndarray) -> np.ndarray:
        holdings = np.concatenate(([X], interior, [0.0]))
        left, mid, right = holdings[:-2], holdings[1:-1], holdings[2:]
        impact_term = (2.0 * params.eta_tilde / params.tau) * (2.0 * mid - left - right)
        variance_term = 2.0 * lam * params.sigma**2 * mid
        return impact_term + variance_term

    result = minimize(
        objective, x0, jac=gradient, method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-18, "gtol": 1e-12},
    )
    return np.concatenate(([X], result.x, [0.0]))


def closed_form_matches_numerical(
    params: ExecutionParams, lam: float, rtol: float = 1e-4
) -> tuple[bool, float]:
    """Compare the two trajectories share-for-share; return (matches, max relative diff).

    The diff is scaled by `total_shares`, not by each point's own holdings. Late in
    an aggressive (large-lam) trajectory, holdings decay to a sliver of the order —
    a few thousandths of one share out of a million. At that scale U(x) is so flat
    that the optimizer can land on a slightly different sliver and still be
    correct; dividing by the (tiny) local holdings value would blow that noise up
    into a fake "mismatch". Dividing by the order size instead asks the question
    that actually matters: does this differ by a meaningful fraction of the order?
    """
    closed_form = optimal_trajectory(params, lam)
    numerical = numerically_optimal_trajectory(params, lam)
    interior_closed = closed_form[1:-1]
    interior_numerical = numerical[1:-1]
    diff = np.abs(interior_closed - interior_numerical)
    max_rel_diff = float(np.max(diff) / params.total_shares)
    return max_rel_diff <= rtol, max_rel_diff
