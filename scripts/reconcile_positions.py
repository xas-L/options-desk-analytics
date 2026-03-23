"""Basic reconciliation: report position rows with no matching line in the chain.

Usage:
    python scripts/reconcile_positions.py --positions data/sample_positions.csv \
        --chain data/sample_chain.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_MERGE_KEYS = ["underlying", "expiry", "cp", "K"]


def reconcile(positions_path: str, chain_path: str) -> pd.DataFrame:
    """Return positions rows with no matching chain line and print a summary."""
    positions = pd.read_csv(positions_path)
    chain = pd.read_csv(chain_path)

    chain_keys = chain[_MERGE_KEYS].drop_duplicates()
    merged = positions.merge(chain_keys, on=_MERGE_KEYS, how="left", indicator=True)
    unmatched = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    total = len(positions)
    n_unmatched = len(unmatched)
    print(f"Positions: {total} rows. Unmatched in chain: {n_unmatched}.")

    if not unmatched.empty:
        print(unmatched.to_string(index=False))

    return unmatched


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile positions against chain.")
    parser.add_argument("--positions", required=True, help="Path to positions CSV")
    parser.add_argument("--chain", required=True, help="Path to chain CSV")
    args = parser.parse_args()
    reconcile(args.positions, args.chain)


if __name__ == "__main__":
    main()
