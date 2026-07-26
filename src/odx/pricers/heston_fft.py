"""Carr-Madan FFT for European options under Heston stochastic volatility."""

from __future__ import annotations

import numpy as np


def heston_cf(u: np.ndarray, S0: float, r: float, q: float, T: float,
              kappa: float, theta: float, sigma: float, rho: float, V0: float) -> np.ndarray:
    """Characteristic function of log(S_T) in the Heston model."""
    a = kappa - 1j * rho * sigma * u
    d = np.sqrt(a**2 + sigma**2 * (u**2 + 1j * u))
    g = (a - d) / (a + d)

    exp_d = np.exp(-d * T)
    C = (kappa * theta / sigma**2) * ((a - d) * T - 2 * np.log((1 - g * exp_d) / (1 - g)))
    D = (a - d) / sigma**2 * ((1 - exp_d) / (1 - g * exp_d))

    return np.exp(1j * u * (np.log(S0) + (r - q) * T) + C + D * V0)


def heston_fft_price(
    S0: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, V0: float,
    option_type: str = "call",
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
) -> float:
    """Price European option using Carr-Madan FFT.
    
    Damping factor alpha=1.5 is standard to ensure the modified payoff is integrable.
    """
    is_call = option_type.strip().lower() in ("call", "c")

    # Grid for u (integration variable)
    u = np.arange(N) * eta

    # Grid for log-strikes
    lambda_ = 2 * np.pi / (N * eta)
    b = (N * lambda_) / 2
    k_vec = -b + np.arange(N) * lambda_

    # Modified characteristic function
    u_mod = u - (alpha + 1) * 1j
    cf_mod = heston_cf(u_mod, S0, r, q, T, kappa, theta, sigma, rho, V0)

    psi = np.exp(-r * T) * cf_mod / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)

    # Simpson's rule weights
    w = np.ones(N)
    w[1::2] = 4
    w[2::2] = 2
    w[0] = 1
    w[-1] = 1
    w = w / 3

    # FFT input
    x = np.exp(1j * b * u) * psi * eta * w

    # Execute FFT
    y = np.fft.fft(x)

    # Option prices across strikes
    call_prices = np.exp(-alpha * k_vec) / np.pi * np.real(y)

    # Interpolate for specific K
    call_price = np.interp(np.log(K), k_vec, call_prices)

    if is_call:
        return float(call_price)
    
    # Put-call parity for put
    put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return float(put_price)
