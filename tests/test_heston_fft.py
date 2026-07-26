"""Tests for Carr-Madan FFT under Heston model."""

import numpy as np

from odx.pricers.analytic.bs import bs_price
from odx.pricers.heston_fft import heston_fft_price


def test_heston_fft_matches_bs():
    """Heston should recover Black-Scholes when vol of vol is zero and V0 = theta."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    
    # Heston parameters reducing to Black-Scholes
    V0 = 0.04 # 20% vol
    kappa = 1.0
    theta = V0
    sigma = 1e-5 # effectively zero vol of vol
    rho = 0.0
    
    bs_call = bs_price(S0, K, T, r, np.sqrt(V0), option_type="call", q=q)
    heston_call = heston_fft_price(S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type="call")
    
    assert abs(bs_call - heston_call) < 1e-2


def test_heston_fft_put_call_parity():
    S0 = 100.0
    K = 90.0
    T = 1.0
    r = 0.05
    q = 0.0
    
    kappa = 2.0
    theta = 0.04
    sigma = 0.1
    rho = -0.5
    V0 = 0.04
    
    call = heston_fft_price(S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type="call")
    put = heston_fft_price(S0, K, T, r, q, kappa, theta, sigma, rho, V0, option_type="put")
    
    parity = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs(call - put - parity) < 1e-4
