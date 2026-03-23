"""Implied volatility solver using Brent's method."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .vanilla_bs import bs_price

_IV_LOWER = 1e-6
_IV_UPPER = 10.0


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
) -> float:
    """Solve for implied vol given a market mid price. Returns nan if no solution found."""
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type, q) - market_price

    try:
        return float(brentq(objective, _IV_LOWER, _IV_UPPER, xtol=tol))
    except ValueError:
        return float("nan")


def implied_vol_vectorised(
    market_prices: np.ndarray,
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    option_types: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """Vectorised IV calculation over arrays of strikes, expiries and option types."""
    return np.array([
        implied_vol(mp, S, k, t, r, cp, q)
        for mp, k, t, cp in zip(market_prices, K, T, option_types)
    ])
