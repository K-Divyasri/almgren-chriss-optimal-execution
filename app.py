"""Interactive Almgren-Chriss frontier explorer.

Run: streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from acexec.analytics import cost_in_bps, cost_std, expected_cost
from acexec.frontier import efficient_frontier, half_life
from acexec.model import ExecutionParams
from acexec.report import plot_frontier, plot_trajectories
from acexec.trajectory import optimal_trajectory
from acexec.verify_optimal import closed_form_matches_numerical

st.set_page_config(page_title="Almgren-Chriss optimal execution", layout="wide")
st.title("Almgren-Chriss optimal execution")
st.caption(
    "A replication of Almgren & Chriss (2000): the optimal trajectory for liquidating "
    "a large order, and the efficient frontier trading expected cost off against timing risk."
)

with st.sidebar:
    st.header("Order & market")
    shares = st.number_input("Shares to trade (X)", value=1_000_000.0, step=100_000.0)
    n_intervals = st.slider("Number of intervals (N)", 5, 60, 20)
    sigma = st.number_input("Per-interval volatility (sigma, $)", value=0.95, step=0.05)
    eta = st.number_input("Temporary impact (eta)", value=2.5e-6, format="%.2e")
    gamma = st.number_input("Permanent impact (gamma)", value=2.5e-7, format="%.2e")
    price = st.number_input("Arrival price ($)", value=50.0, step=1.0)
    st.header("Risk aversion")
    lam_exp = st.slider("lambda (log10 scale)", -9, -4, -6)
    lam = 0.0 if lam_exp <= -9 else 10.0**lam_exp
    st.write(f"lam = {lam:.2e}" if lam > 0 else "lam = 0 (TWAP)")

params = ExecutionParams(
    total_shares=shares, n_intervals=n_intervals, sigma=sigma, eta=eta, gamma=gamma, arrival_price=price
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Optimal trajectory")
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_trajectories(params, [0.0, lam if lam > 0 else 1e-7], ax=ax)
    st.pyplot(fig)

    holdings = optimal_trajectory(params, lam)
    e_bps = cost_in_bps(params, expected_cost(params, holdings))
    s_bps = cost_in_bps(params, cost_std(params, holdings))
    hl = half_life(params, lam)
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected cost", f"{e_bps:.1f} bps")
    m2.metric("Cost std (risk)", f"{s_bps:.1f} bps")
    m3.metric("Half-life", f"{hl:.1f} intervals")

with col2:
    st.subheader("Efficient frontier")
    frontier_df = efficient_frontier(params)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_frontier(frontier_df, ax=ax2)
    ax2.scatter([s_bps / 1e4 * shares * price], [e_bps / 1e4 * shares * price], color="red", zorder=5, label="current lam")
    st.pyplot(fig2)

st.divider()
st.subheader("Verify the closed form against independent numerical optimization")
if st.button("Run verification"):
    ok, max_diff = closed_form_matches_numerical(params, lam if lam > 0 else 1e-7)
    if ok:
        st.success(f"Closed form matches a from-scratch numerical optimum (max relative diff {max_diff:.2e}).")
    else:
        st.error(f"Mismatch! max relative diff {max_diff:.2e}")
