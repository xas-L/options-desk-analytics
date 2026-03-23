"""Portfolio risk report: join positions to chain, compute IV/Greeks, aggregate, run scenarios.

Usage:
    python scripts/portfolio_risk_report.py --positions data/sample_positions.csv \
        --chain data/sample_chain.csv --outdir out_risk
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pricer.greeks_bs import bs_greeks
from pricer.implied_vol import implied_vol

MULTIPLIER = 100
_FALLBACK_IV = 0.20
_MERGE_KEYS = ["underlying", "expiry", "cp", "K"]


def _enrich_chain(chain: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Add IV and Greeks columns to the chain dataframe."""
    chain = chain.copy()
    if "mid" not in chain.columns:
        chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0

    ivs, deltas, gammas, vegas, thetas = [], [], [], [], []
    for _, row in chain.iterrows():
        S = float(row["spot"])
        r_row = float(row.get("r", r))
        q_row = float(row.get("q", q))
        iv = implied_vol(row["mid"], S, row["K"], row["T"], r_row, row["cp"], q_row)
        sigma = iv if not np.isnan(iv) else _FALLBACK_IV
        g = bs_greeks(S, row["K"], row["T"], r_row, sigma, row["cp"], q_row)
        ivs.append(iv)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        vegas.append(g["vega"])
        thetas.append(g["theta"])

    chain["iv"] = ivs
    chain["delta"] = deltas
    chain["gamma"] = gammas
    chain["vega"] = vegas
    chain["theta"] = thetas
    return chain


def _build_position_greeks(positions: pd.DataFrame, chain: pd.DataFrame) -> pd.DataFrame:
    """Join positions to enriched chain and compute dollar Greeks."""
    merged = positions.merge(chain, on=_MERGE_KEYS, how="left")

    spot = merged["spot"]
    qty = merged["qty"]

    merged["dollar_delta"] = merged["delta"] * qty * MULTIPLIER * spot
    merged["dollar_gamma"] = merged["gamma"] * qty * MULTIPLIER * spot ** 2 / 100.0
    merged["dollar_vega"] = merged["vega"] * qty * MULTIPLIER / 100.0
    merged["dollar_theta"] = merged["theta"] * qty * MULTIPLIER / 365.0
    return merged


def _aggregate_risk(pos_greeks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dollar_cols = ["dollar_delta", "dollar_gamma", "dollar_vega", "dollar_theta"]
    by_underlying = (
        pos_greeks.groupby("underlying")[dollar_cols].sum().reset_index()
    )
    by_expiry = (
        pos_greeks.groupby(["underlying", "expiry"])[dollar_cols].sum().reset_index()
    )
    return {"by_underlying": by_underlying, "by_expiry": by_expiry}


def _scenario_pnl(
    pos_greeks: pd.DataFrame,
    spot_shocks: list[float],
    vol_shocks: list[float],
) -> pd.DataFrame:
    """First-order (delta/vega) scenario PnL grid."""
    rows = []
    for ds in spot_shocks:
        for dv in vol_shocks:
            pnl = (
                (pos_greeks["dollar_delta"] * ds) +
                (pos_greeks["dollar_vega"] * dv)
            ).sum()
            rows.append({
                "spot_shock_pct": round(ds * 100, 1),
                "vol_shock_pts": round(dv * 100, 1),
                "pnl": round(pnl, 2),
            })
    return pd.DataFrame(rows)


def run_report(
    positions_path: str,
    chain_path: str,
    outdir: str,
    r: float = 0.05,
    q: float = 0.0,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    positions = pd.read_csv(positions_path)
    chain = pd.read_csv(chain_path)

    chain = _enrich_chain(chain, r, q)
    pos_greeks = _build_position_greeks(positions, chain)
    pos_greeks.to_csv(os.path.join(outdir, "positions_with_greeks.csv"), index=False)

    agg = _aggregate_risk(pos_greeks)
    agg["by_underlying"].to_csv(os.path.join(outdir, "risk_by_underlying.csv"), index=False)
    agg["by_expiry"].to_csv(os.path.join(outdir, "risk_by_expiry.csv"), index=False)

    spot_shocks = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    vol_shocks = [-0.05, -0.02, 0.0, 0.02, 0.05]
    scen = _scenario_pnl(pos_greeks, spot_shocks, vol_shocks)
    scen.to_csv(os.path.join(outdir, "scenario_pnl.csv"), index=False)

    print(f"Report written to {outdir}/")
    print(f"  positions_with_greeks.csv  ({len(pos_greeks)} rows)")
    print(f"  risk_by_underlying.csv")
    print(f"  risk_by_expiry.csv")
    print(f"  scenario_pnl.csv  ({len(scen)} scenarios)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio risk report for vanilla options.")
    parser.add_argument("--positions", required=True, help="Path to positions CSV")
    parser.add_argument("--chain", required=True, help="Path to chain CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--r", type=float, default=0.05, help="Default risk-free rate")
    parser.add_argument("--q", type=float, default=0.0, help="Default dividend yield")
    args = parser.parse_args()
    run_report(args.positions, args.chain, args.outdir, args.r, args.q)


if __name__ == "__main__":
    main()
