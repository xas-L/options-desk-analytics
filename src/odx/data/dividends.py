"""Implied dividend yield estimation from the forward curve.

Given the no-arbitrage relationship for the forward of a continuous-dividend-
paying asset:

    F = S * e^{(r - q) * T}

Solving for q:

    q = r - ln(F / S) / T

This is the "implied" dividend yield — the yield the market is pricing into
the option chain. It can differ from the trailing/announced yield due to
expected dividend changes, special dividends, or ex-date timing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_dividend_yield(spot: float, F: float, T: float, r: float) -> float:
    """Estimate continuous implied dividend yield from a single forward point.

    q = r - ln(F / S) / T

    Returns NaN if any input is non-positive or T is zero.
    """
    if spot <= 0 or F <= 0 or T <= 0:
        return np.nan
    return r - np.log(F / spot) / T


def implied_dividend_curve(spot: float, forward_curve: pd.DataFrame) -> pd.DataFrame:
    """Compute implied dividend yield across all expiries.

    Expects forward_curve to have columns [expiry, T, F] (as produced
    by bootstrap_forward_curve). Adds a q_implied column.

    Returns a DataFrame with columns [expiry, T, F, q_implied].
    """
    result = forward_curve.copy()
    result["q_implied"] = result.apply(
        lambda row: implied_dividend_yield(spot, row["F"], row["T"], row.get("r", 0.0)),
        axis=1,
    )
    return result
