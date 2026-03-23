"""Per-expiry smile utilities: build IV smile from a chain slice and plot it."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .implied_vol import implied_vol


def build_smile(
    chain: pd.DataFrame,
    expiry: str,
    S: float,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """Compute IV per strike for a single expiry slice.

    chain must have columns: expiry, cp, K, T and either mid or bid+ask.
    Returns a dataframe with K, cp, T, mid, iv columns sorted by K.
    """
    slice_df = chain[chain["expiry"] == expiry].copy()
    if slice_df.empty:
        raise ValueError(f"No rows found for expiry {expiry}")

    if "mid" not in slice_df.columns:
        slice_df["mid"] = (slice_df["bid"] + slice_df["ask"]) / 2.0

    ivs = []
    for _, row in slice_df.iterrows():
        r_row = row.get("r", r)
        q_row = row.get("q", q)
        iv = implied_vol(row["mid"], S, row["K"], row["T"], r_row, row["cp"], q_row)
        ivs.append(iv)

    slice_df["iv"] = ivs
    return (
        slice_df[["K", "cp", "T", "mid", "iv"]]
        .sort_values("K")
        .reset_index(drop=True)
    )


def plot_smile(smile_df: pd.DataFrame, expiry: str, S: float) -> None:
    """Plot implied vol vs strike for a given expiry, with calls and puts overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for cp, grp in smile_df.groupby("cp"):
        label = "Call" if cp in ("call", "c") else "Put"
        valid = grp.dropna(subset=["iv"])
        ax.plot(valid["K"], valid["iv"] * 100, marker="o", linestyle="-", label=label)

    ax.axvline(S, linestyle="--", color="grey", alpha=0.6, label="Spot")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(f"IV Smile  |  expiry {expiry}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
