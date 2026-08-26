import numpy as np

from acexec.benchmark import intraday_volume_profile, twap_holdings, vwap_holdings
from acexec.model import ExecutionParams


def _params():
    return ExecutionParams(total_shares=1_000_000.0, n_intervals=10, tau=1.0)


def test_twap_starts_and_ends_correctly():
    p = _params()
    holdings = twap_holdings(p)
    assert holdings[0] == p.total_shares
    assert abs(holdings[-1]) < 1e-6


def test_twap_is_linear_equal_sized_trades():
    p = _params()
    holdings = twap_holdings(p)
    trades = -np.diff(holdings)
    assert np.allclose(trades, trades[0])


def test_intraday_volume_profile_sums_to_one_and_is_u_shaped():
    profile = intraday_volume_profile(10)
    assert abs(profile.sum() - 1.0) < 1e-9
    # heaviest at the open and close, lightest in the middle
    mid = len(profile) // 2
    assert profile[0] > profile[mid]
    assert profile[-1] > profile[mid]


def test_vwap_starts_and_ends_correctly():
    p = _params()
    holdings = vwap_holdings(p)
    assert holdings[0] == p.total_shares
    assert abs(holdings[-1]) < 1e-6


def test_vwap_trades_more_at_open_and_close_than_midday():
    p = _params()
    holdings = vwap_holdings(p)
    trades = -np.diff(holdings)
    mid = len(trades) // 2
    assert trades[0] > trades[mid]
