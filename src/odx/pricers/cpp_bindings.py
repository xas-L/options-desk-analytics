"""C++ Black-Scholes bindings with pure-Python fallback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Try importing the native C++ module
try:
    import odx.pricers.bs_pricer_cpp as native_bs
    HAS_CPP_PRICER = True
except ImportError:
    HAS_CPP_PRICER = False
    logger.warning(
        "Native C++ pricer 'bs_pricer_cpp' not found. "
        "Falling back to pure-Python implementation. "
        "Run `scripts/build_cpp.ps1` to compile the C++ extension for better performance."
    )
    
# Fallback to pure-Python implementations
from odx.pricers.analytic.bs import bs_price as py_bs_price
from odx.greeks.analytic import (
    bs_delta as py_bs_delta,
    bs_gamma as py_bs_gamma,
    bs_vega as py_bs_vega,
    bs_theta as py_bs_theta,
    bs_rho as py_bs_rho
)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_price(S, K, T, r, sigma, option_type, q)
    return py_bs_price(S, K, T, r, sigma, option_type, q)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_delta(S, K, T, r, sigma, option_type, q)
    return py_bs_delta(S, K, T, r, sigma, option_type, q)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_gamma(S, K, T, r, sigma, q)
    return py_bs_gamma(S, K, T, r, sigma, q)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_vega(S, K, T, r, sigma, q)
    return py_bs_vega(S, K, T, r, sigma, q)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_theta(S, K, T, r, sigma, option_type, q)
    return py_bs_theta(S, K, T, r, sigma, option_type, q)


def bs_rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    if HAS_CPP_PRICER:
        return native_bs.bs_rho(S, K, T, r, sigma, option_type, q)
    return py_bs_rho(S, K, T, r, sigma, option_type, q)
