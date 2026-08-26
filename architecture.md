# Architecture

## Data flow

```
ExecutionParams (model.py)
        │
        ├──▶ trajectory.py ──▶ optimal_trajectory(params, lam)  [the closed form]
        │            │
        │            └──▶ verify_optimal.py ──▶ numerically_optimal_trajectory(params, lam)
        │                         (independent scipy L-BFGS-B optimum, exact gradient supplied)
        │
        ├──▶ analytics.py ──▶ expected_cost, cost_variance, mean_variance_objective
        │            │            (pure algebra, works on ANY holdings array, not just optimal ones)
        │            │
        │            └──▶ simulate.py ──▶ monte_carlo_summary(params, holdings, n_paths)
        │                         (independent Monte Carlo estimate of the same two quantities)
        │
        ├──▶ frontier.py ──▶ efficient_frontier(params, lambdas)
        │            (sweeps trajectory.py + analytics.py across many lam values, no simulation)
        │
        ├──▶ benchmark.py ──▶ twap_holdings, vwap_holdings
        │            (self-contained; does not import trajectory.py, though TWAP is
        │             mathematically identical to optimal_trajectory(params, lam=0))
        │
        └──▶ report.py ──▶ plot_trajectories, plot_frontier
                     (thin matplotlib wrappers, used by both app.py and the notebooks)

cli.py    -- argparse front door to trajectory.py / frontier.py / verify_optimal.py
app.py    -- Streamlit front door to the same three, plus report.py's plots
```

## Why the modules are split this way

**`model.py` has no logic beyond derived properties.** `ExecutionParams` is
a plain dataclass; `eta_tilde`, `horizon`, and `times()` are the only
derived values, because they're used identically by `trajectory.py` and
`analytics.py` and shouldn't be recomputed slightly differently in each
place.

**`trajectory.py` and `analytics.py` are separate, deliberately.**
`trajectory.py` answers "what's the best schedule?", a question with one
right answer for a given `(params, lam)`. `analytics.py` answers "what does
*any* schedule cost?", a question that has to work on arbitrary holdings
arrays (TWAP, VWAP, a hand-typed array), not just the optimal one.

**`verify_optimal.py` and `simulate.py` depend on `trajectory.py` and
`analytics.py`, never the reverse.** Verification code needs to see the
thing it's checking; the thing being checked must never import its own
checker; that dependency direction is what makes the checks meaningful
rather than circular.

**`benchmark.py` has zero dependency on `trajectory.py`.** This is the one
deliberate piece of duplication in the package: TWAP's formula
(`X * (1 - t/T)`) is mathematically identical to `optimal_trajectory(params,
lam=0)`, and `tests/test_benchmark.py` confirms the two independent formulas
agree to floating-point precision. Keeping them as two separate
implementations, rather than having one call the other, is what makes that
agreement a real check instead of a tautology.

## Numerical notes worth knowing

**`verify_optimal.py` needs an exact gradient, not a finite-difference
estimate.** `scipy.optimize.minimize`'s default (finite-difference) gradient
estimation converges too loosely in the low-magnitude tail of an aggressive
trajectory, understating agreement with the closed form by up to two orders
of magnitude. `numerically_optimal_trajectory` differentiates `U(x)`
directly (plain calculus, no `sinh`/`cosh` involved) and passes that exact
gradient to `L-BFGS-B` via `jac=`, which converges to the true optimum to
about one part in a hundred million instead.

**Comparing trajectories near zero needs an absolute, not relative, scale.**
`closed_form_matches_numerical` measures disagreement as a fraction of
`total_shares` (the order size), not as a fraction of each point's own
holdings value. Late in an aggressive trajectory, holdings decay to a sliver
of the order where the objective function is nearly flat — many different
tiny values are equally optimal up to floating-point noise — so a
local-relative-difference metric reports a large, spurious "mismatch" on a
difference of a fraction of one share out of a million. Scaling by order
size instead asks the question that actually matters operationally.

## Testing strategy

- `test_trajectory.py`: the closed-form formula's own properties (boundary
  conditions, TWAP collapse at `lam=0`, monotonic front-loading as `lam`
  grows).
- `test_analytics.py`: the cost/variance formulas, plus the "AC beats every
  other schedule on its own objective" property that makes optimality
  meaningful.
- `test_verify_optimal.py`: the independent numerical optimizer agrees with
  the closed form across low/moderate/high `lam`.
- `test_frontier.py`: the frontier is monotonic and its two named endpoints
  (TWAP, near-immediate liquidation) behave as expected.
- `test_simulate.py`: the from-scratch Monte Carlo simulator converges to
  the closed-form cost formulas as `n_paths` grows (not just "is close
  once," but "gets closer with more paths," the actual signature of correct
  convergence).
- `test_benchmark.py`: TWAP/VWAP's own boundary conditions and shape.
- `test_cli.py`: every subcommand runs and produces parseable output.

34 tests, all passing; see `README.md` for how to run them.
