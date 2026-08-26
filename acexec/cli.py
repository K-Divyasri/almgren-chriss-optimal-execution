"""Command line: python -m acexec <command>."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .analytics import cost_in_bps, cost_std, expected_cost
from .frontier import efficient_frontier, half_life
from .model import ExecutionParams
from .trajectory import optimal_trajectory
from .verify_optimal import closed_form_matches_numerical


def _params_from_args(args: argparse.Namespace) -> ExecutionParams:
    return ExecutionParams(
        total_shares=args.shares,
        n_intervals=args.intervals,
        sigma=args.sigma,
        eta=args.eta,
        gamma=args.gamma,
        arrival_price=args.price,
    )


def cmd_trajectory(args: argparse.Namespace) -> None:
    params = _params_from_args(args)
    holdings = optimal_trajectory(params, args.lam)
    e = expected_cost(params, holdings)
    s = cost_std(params, holdings)
    print(f"lam={args.lam:g}  expected cost = {cost_in_bps(params, e):.2f} bps  "
          f"cost std = {cost_in_bps(params, s):.2f} bps  half-life = {half_life(params, args.lam):.1f} intervals")
    for j, x in enumerate(holdings):
        print(f"  t={j:>2}  holdings={x:,.0f}")


def cmd_frontier(args: argparse.Namespace) -> None:
    params = _params_from_args(args)
    df = efficient_frontier(params)
    df["expected_cost_bps"] = df["expected_cost"].apply(lambda c: cost_in_bps(params, c))
    df["cost_std_bps"] = df["cost_std"].apply(lambda c: cost_in_bps(params, c))
    if args.json:
        print(df.to_json(orient="records", indent=2))
    else:
        print(df[["lam", "expected_cost_bps", "cost_std_bps"]].to_string(index=False))


def cmd_verify(args: argparse.Namespace) -> None:
    params = _params_from_args(args)
    lambdas = [0.0, 1e-7, 1e-6, 1e-5]
    all_ok = True
    for lam in lambdas:
        ok, max_diff = closed_form_matches_numerical(params, lam)
        all_ok = all_ok and ok
        print(f"lam={lam:g}  closed-form vs numerical optimum: max relative diff = {max_diff:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")
    if not all_ok:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="acexec", description="Almgren-Chriss optimal execution replication")
    p.add_argument("--shares", type=float, default=1_000_000.0)
    p.add_argument("--intervals", type=int, default=20)
    p.add_argument("--sigma", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=2.5e-6)
    p.add_argument("--gamma", type=float, default=2.5e-7)
    p.add_argument("--price", type=float, default=50.0)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("trajectory", help="print the optimal holdings trajectory for one risk aversion")
    t.add_argument("--lam", type=float, default=1e-6)
    t.set_defaults(func=cmd_trajectory)

    f = sub.add_parser("frontier", help="print the efficient frontier (cost vs risk, swept over lam)")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_frontier)

    v = sub.add_parser("verify", help="check the closed-form trajectory against numerical optimization")
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
