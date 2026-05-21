"""Black-Scholes Greeks for vanilla European options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .vanilla_bs import _d1_d2


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """Delta: sensitivity of price to spot."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if option_type.strip().lower() in ("call", "c"):
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
) -> float:
    """Gamma: second-order sensitivity to spot, same for call and put."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
) -> float:
    """Vega: sensitivity to a 1-unit move in vol (divide by 100 for per-vol-point)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """Theta in price units per year. Divide by 365 for daily decay."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    decay_term = -(S * disc_q * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

    if option_type.strip().lower() in ("call", "c"):
        return decay_term - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    return decay_term + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """Rho: sensitivity to risk-free rate."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if option_type.strip().lower() in ("call", "c"):
        return K * T * disc_r * norm.cdf(d2)
    return -K * T * disc_r * norm.cdf(-d2)


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0,
) -> dict:
    """Return all five BS Greeks as a dict."""
    return {
        "delta": bs_delta(S, K, T, r, sigma, option_type, q),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega": bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, option_type, q),
        "rho": bs_rho(S, K, T, r, sigma, option_type, q),
    }
