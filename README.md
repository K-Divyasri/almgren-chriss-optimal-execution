# acexec

A from-scratch replication of Almgren & Chriss (2000), built as a proper
Python package with tests. Everything is closed-form algebra or a local
simulation: no market data subscription, no API key, no account, nothing
that needs the internet to run.

> Educational project, not financial advice.

## What it does

Given an order (how many shares, over how many intervals) and a market
(volatility, temporary/permanent impact coefficients), `acexec`:

- computes the **closed-form optimal trajectory** for any chosen risk
  aversion `lambda`, derived from first-principles calculus (`trajectory.py`),
- prices any trajectory's **expected cost and variance** algebraically, with
  no simulation (`analytics.py`),
- **independently verifies** the closed form against a generic numerical
  optimizer that never sees the sinh/cosh formula (`verify_optimal.py`),
- sweeps `lambda` to compute the **efficient frontier**, the full cost-vs-
  risk trade-off curve, in closed form (`frontier.py`),
- **independently checks** the cost formulas with its own from-scratch
  Monte Carlo simulator (`simulate.py`),
- builds self-contained **TWAP and VWAP** benchmarks for honest comparison,
  reimplemented from scratch rather than borrowed from elsewhere
  (`benchmark.py`),
- exposes all of the above through a **CLI** (`cli.py`) and a **Streamlit
  dashboard** (`app.py`).

## The pieces (in `acexec/`)

| file | what it does |
|------|--------------|
| `model.py` | `ExecutionParams`, the order and the market it's trading into |
| `trajectory.py` | the closed-form optimal holdings, derived and documented in full |
| `analytics.py` | `expected_cost`, `cost_variance`, the mean-variance objective, bps conversion |
| `verify_optimal.py` | independent numerical optimization (`scipy`, exact gradient) checking the closed form |
| `frontier.py` | sweeps `lambda` into the efficient frontier; `half_life` as a one-number urgency read |
| `simulate.py` | an independent Monte Carlo simulator, checking the cost formulas empirically |
| `benchmark.py` | self-contained TWAP/VWAP, for honest comparison |
| `report.py` | matplotlib plotting helpers for the trajectory and frontier |
| `cli.py` | `python -m acexec ...` |

## Run it

```powershell
# from this folder
pip install -e ".[dev,plots,dashboard]"

# the optimal trajectory for one risk aversion
python -m acexec --intervals 20 trajectory --lam 1e-6

# the full efficient frontier
python -m acexec --intervals 20 frontier
python -m acexec --intervals 20 frontier --json

# check the closed form against independent numerical optimization
python -m acexec --intervals 20 verify

# the interactive dashboard
streamlit run app.py

# the tests
python -m pytest -q                 # 34 passed
```

Every global flag (`--shares`, `--intervals`, `--sigma`, `--eta`, `--gamma`,
`--price`) comes *before* the subcommand name, standard `argparse` style:
`python -m acexec --intervals 10 trajectory --lam 1e-6`, not
`python -m acexec trajectory --intervals 10`.

## Why it's verified two independent ways

`trajectory.py`'s closed-form optimal holdings are derived here from the
difference equation up, not copied from a reference implementation.
`verify_optimal.py` then checks that closed form against a generic numerical
optimizer that never sees the sinh/cosh formula, and `simulate.py` checks the
resulting cost and variance formulas against an independent from-scratch
Monte Carlo simulator. Two different failure modes, two different checks:
the numerical optimizer would catch an algebra mistake in the closed form,
and the Monte Carlo simulator would catch a mismatch between the formulas
and the actual random process a real execution would face. Passing both is
much stronger evidence than passing either alone.

## What "no simulation needed" buys you

Because `analytics.py`'s cost and variance formulas are exact algebra,
`frontier.py` computes an entire efficient frontier, dozens of `(cost, risk)`
points, in a few milliseconds, with nothing random in it at all.
`simulate.py` exists purely as an independent check that those formulas
actually describe the same random process a real execution would face, not
as a replacement for them.
