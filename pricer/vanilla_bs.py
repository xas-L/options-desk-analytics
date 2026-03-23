"""Black-Scholes pricing for vanilla European options with optional dividend yield."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes price for a vanilla European call or put."""
    cp = option_type.strip().lower()
    if cp not in ("call", "put", "c", "p"):
        raise ValueError("option_type must be 'call' or 'put'")
    is_call = cp in ("call", "c")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    if is_call:
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
