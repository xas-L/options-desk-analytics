"""Implied borrow/repo cost estimation from the forward curve.

The total cost of carry embedded in the forward is:

    c = ln(F / S) / T

Under the standard model with risk-free rate r and continuous dividend q:

    F = S * e^{(r - q) * T}   =>   c = r - q

Any residual between the observed cost of carry and the theoretical
(r - q) represents an implied borrow cost (or repo spread):

    b = c - (r - q) = ln(F / S) / T - r + q

Interpretation:
  - b > 0: the stock is "hard to borrow" — forward is elevated relative
    to the dividend-adjusted risk-free carry.
  - b < 0: the stock trades "special" in repo — rare, usually indicates
    measurement noise or a dividend model mismatch.
  - b ~ 0: normal borrow conditions.

This is a key input for short-selling cost estimation and for
correctly pricing options on hard-to-borrow names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_borrow_cost(spot: float, F: float, T: float, r: float, q: float) -> float:
    """Estimate implied borrow/repo cost from a single forward point.

    b = ln(F / S) / T - r + q

    Returns NaN if spot, F, or T are non-positive.
    """
    if spot <= 0 or F <= 0 or T <= 0:
        return np.nan
    return np.log(F / spot) / T - r + q


def implied_borrow_curve(
    spot: float,
    forward_curve: pd.DataFrame,
    q: float = 0.0,
) -> pd.DataFrame:
    """Compute implied borrow cost across all expiries.

    Expects forward_curve to have columns [expiry, T, F] and optionally [r].
    Adds a borrow_cost column.

    Returns a DataFrame with columns [expiry, T, F, borrow_cost].
    """
    result = forward_curve.copy()
    result["borrow_cost"] = result.apply(
        lambda row: implied_borrow_cost(spot, row["F"], row["T"], row.get("r", 0.0), q),
        axis=1,
    )
    return result
