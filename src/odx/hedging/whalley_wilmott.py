"""Whalley-Wilmott no-transaction-cost hedging band."""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from odx.pricers.analytic.bs import _d1_d2


def whalley_wilmott_band(
    S: float, K: float, T: float, r: float, sigma: float, 
    gamma_risk_aversion: float, transaction_cost: float, 
    option_type: str = "call", q: float = 0.0
) -> tuple[float, float, float]:
    """Whalley-Wilmott optimal hedging band around Black-Scholes delta.
    
    Returns (lower_bound, bs_delta, upper_bound).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    
    is_call = option_type.strip().lower() in ("call", "c")
    bs_delta = np.exp(-q * T) * norm.cdf(d1) if is_call else -np.exp(-q * T) * norm.cdf(-d1)
    
    # BS Gamma
    bs_gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    # Band width H
    band_width = ((1.5 * transaction_cost * S * bs_gamma**2) / gamma_risk_aversion) ** (1.0 / 3.0)
    
    return float(bs_delta - band_width), float(bs_delta), float(bs_delta + band_width)
