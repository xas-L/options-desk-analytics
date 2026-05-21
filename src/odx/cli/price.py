"""CLI: compute IV and Greeks for each row in an options chain snapshot.

Usage:
    python scripts/chain_to_iv_greeks.py --chain data/sample_chain.csv --out out_chain_iv_greeks.csv
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

_FALLBACK_IV = 0.20


def process_chain(
    chain_path: str,
    out_path: str,
    r: float = 0.05,
    q: float = 0.0,
) -> pd.DataFrame:
    """Compute IV and Greeks for every row in the chain CSV and write results."""
    df = pd.read_csv(chain_path)

    required = {"underlying", "expiry", "cp", "K", "T", "spot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Chain CSV missing required columns: {missing}")

    if "mid" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
        else:
            raise ValueError("Chain CSV must have a 'mid' column or both 'bid' and 'ask'.")

    records = []
    for _, row in df.iterrows():
        S = float(row["spot"])
        r_row = float(row.get("r", r))
        q_row = float(row.get("q", q))

        iv = implied_vol(row["mid"], S, row["K"], row["T"], r_row, row["cp"], q_row)
        sigma_for_greeks = iv if not np.isnan(iv) else _FALLBACK_IV
        greeks = bs_greeks(S, row["K"], row["T"], r_row, sigma_for_greeks, row["cp"], q_row)

        records.append({**row.to_dict(), "iv": iv, **greeks})

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Written {len(out_df)} rows to {out_path}")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IV and Greeks for an options chain.")
    parser.add_argument("--chain", required=True, help="Path to chain CSV")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--r", type=float, default=0.05, help="Default risk-free rate")
    parser.add_argument("--q", type=float, default=0.0, help="Default dividend yield")
    args = parser.parse_args()
    process_chain(args.chain, args.out, args.r, args.q)


if __name__ == "__main__":
    main()
