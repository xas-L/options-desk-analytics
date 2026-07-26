"""Forward curve bootstrap from put-call parity.

Put-Call Parity Derivation
--------------------------
For European options on an asset paying continuous dividend yield q:

    C - P = S * e^{-qT} - K * e^{-rT}

Rearranging for the forward price F = S * e^{(r-q)T}:

    C - P = e^{-rT} * (F - K)
    F - K = e^{rT} * (C - P)
    F = K + e^{rT} * (C - P)

This gives one forward estimate per matched call/put strike pair.
In practice each strike produces a slightly different F due to
bid-ask noise, model imprecision, and early-exercise premia leaking
into American-style quotes. We take the **median** of per-strike
forwards as a robust central estimate, resistant to outliers from
deep ITM/OTM wings where quotes are least reliable.

Reference: Hull, "Options, Futures, and Other Derivatives", Ch. 11.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from odx.logging import get_logger

logger = get_logger(__name__)


def bootstrap_forward(
    chain_slice: pd.DataFrame,
    r: float | None = None,
    T: float | None = None,
) -> float:
    """Estimate the forward price from a single-expiry chain slice.

    For each strike K where both a call and put mid-price exist, computes:
        F_K = K + e^{rT} * (C_mid - P_mid)

    Returns the median of per-strike forward estimates.

    If r or T are not supplied they are read from the first row of
    the slice (the standard chain schema stores them per-row).

    Returns NaN if fewer than 2 matched pairs are available.
    """
    if chain_slice.empty:
        return np.nan

    if r is None:
        r = float(chain_slice.iloc[0]["r"])
    if T is None:
        T = float(chain_slice.iloc[0]["T"])

    # Numerical guard: nonsensical inputs
    if T <= 0:
        return np.nan

    calls = chain_slice[chain_slice["cp"] == "call"][["K", "mid"]].rename(columns={"mid": "C"})
    puts = chain_slice[chain_slice["cp"] == "put"][["K", "mid"]].rename(columns={"mid": "P"})

    matched = calls.merge(puts, on="K", how="inner")
    if len(matched) < 2:
        return np.nan

    discount = np.exp(r * T)
    matched["F"] = matched["K"] + discount * (matched["C"] - matched["P"])

    # Drop obvious outliers (negative forwards) before taking median.
    valid = matched["F"][matched["F"] > 0]
    if valid.empty:
        return np.nan

    return float(valid.median())


def bootstrap_forward_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap forward prices across all expiries in the chain.

    Returns a DataFrame with columns [expiry, T, F].
    """
    records = []
    for expiry, group in df.groupby("expiry"):
        T = float(group.iloc[0]["T"])
        r = float(group.iloc[0]["r"])
        F = bootstrap_forward(group, r=r, T=T)
        records.append({"expiry": expiry, "T": T, "F": F})

    result = pd.DataFrame(records)
    return result.sort_values("T").reset_index(drop=True)
