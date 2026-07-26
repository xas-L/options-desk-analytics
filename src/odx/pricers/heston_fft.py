"""Carr-Madan FFT and Quad Integration for European options under Heston stochastic volatility."""

from __future__ import annotations

import logging
import numpy as np
import scipy.integrate as integrate

# Attempt to load the native C++ implementation
try:
    from odx.pricers.bs_pricer_cpp import heston_fft_price as _heston_fft_price_cpp
    from odx.pricers.bs_pricer_cpp import heston_fft_price_batch as _heston_fft_price_batch_cpp
    HAS_CPP_PRICER = True
except ImportError:
    HAS_CPP_PRICER = False
    logging.warning("Native C++ pricer for Heston FFT not found. Falling back to pure Python Gil-Pelaez quad integration.")


def heston_cf(u: np.ndarray | float | complex, S0: float, r: float, q: float, T: float,
              kappa: float, theta: float, sigma: float, rho: float, V0: float) -> np.ndarray | complex:
    """Characteristic function of log(S_T) in the Heston model."""
    a = kappa - 1j * rho * sigma * u
    d = np.sqrt(a**2 + sigma**2 * (u**2 + 1j * u))
    g = (a - d) / (a + d)

    exp_d = np.exp(-d * T)
    C = (kappa * theta / sigma**2) * ((a - d) * T - 2 * np.log((1 - g * exp_d) / (1 - g)))
    D = (a - d) / sigma**2 * ((1 - exp_d) / (1 - g * exp_d))

    return np.exp(1j * u * (np.log(S0) + (r - q) * T) + C + D * V0)


def _heston_quad_price_py(
    S0: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, V0: float,
    option_type: str = "call"
) -> float:
    """Fallback pure-Python pricer using Gil-Pelaez numerical integration via scipy.integrate.quad."""
    is_call = option_type.strip().lower() in ("call", "c")
    
    def integrand_P1(u: float) -> float:
        num = np.exp(-1j * u * np.log(K)) * heston_cf(u - 1j, S0, r, q, T, kappa, theta, sigma, rho, V0)
        den = 1j * u * heston_cf(-1j, S0, r, q, T, kappa, theta, sigma, rho, V0)
        return float(np.real(num / den))
        
    def integrand_P2(u: float) -> float:
        num = np.exp(-1j * u * np.log(K)) * heston_cf(u, S0, r, q, T, kappa, theta, sigma, rho, V0)
        den = 1j * u
        return float(np.real(num / den))
    
    int1, _ = integrate.quad(integrand_P1, 1e-6, 100, limit=200)
    int2, _ = integrate.quad(integrand_P2, 1e-6, 100, limit=200)
    
    P1 = 0.5 + (1.0 / np.pi) * int1
    P2 = 0.5 + (1.0 / np.pi) * int2
    
    F = S0 * np.exp(-q * T)
    call_price = F * P1 - K * np.exp(-r * T) * P2
    
    if is_call:
        return call_price
    
    return call_price - F + K * np.exp(-r * T)


def heston_fft_price(
    S0: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, V0: float,
    option_type: str = "call",
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
) -> float:
    """Price European option using Heston model.
    
    Uses native C++ Carr-Madan FFT if available, otherwise falls back to 
    pure-Python Gil-Pelaez quad integration (ignoring N, eta, alpha).
    """
    if HAS_CPP_PRICER:
        return _heston_fft_price_cpp(
            S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type, N, eta, alpha
        )
    return _heston_quad_price_py(
        S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type
    )


def heston_fft_price_batch(
    S0: float, K_vec: list[float], T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, V0: float,
    option_type: str = "call",
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
) -> list[float]:
    """Price multiple European options using Heston model.
    
    Uses native C++ Carr-Madan FFT if available, otherwise iterates over 
    the fallback pure-Python Gil-Pelaez pricer.
    """
    if HAS_CPP_PRICER:
        return _heston_fft_price_batch_cpp(
            S0, K_vec, T, r, q, kappa, theta, sigma, rho, V0, option_type, N, eta, alpha
        )
        
    return [
        _heston_quad_price_py(
            S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type
        )
        for K in K_vec
    ]
