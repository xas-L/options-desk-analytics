"""Tests for native C++ Heston FFT pricer vs pure Python fallback."""

import math
import pytest
import numpy as np

try:
    from odx.pricers.bs_pricer_cpp import heston_fft_price as cpp_heston
    from odx.pricers.bs_pricer_cpp import heston_fft_price_batch as cpp_heston_batch
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from odx.pricers.heston_fft import _heston_quad_price_py, heston_fft_price, heston_fft_price_batch


def test_heston_quad_py_sanity():
    """Test that the pure Python Gil-Pelaez quad integration produces reasonable prices."""
    # Basic parameter set
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    V0 = 0.04
    
    price = _heston_quad_price_py(S0, K, T, r, q, kappa, theta, sigma, rho, V0, "call")
    # Roughly BS price at vol=0.2 (sqrt(0.04)) is ~10.45, Heston should be in the same ballpark
    assert 5.0 < price < 15.0


@pytest.mark.skipif(not HAS_CPP, reason="Native C++ extension not built")
def test_heston_native_vs_fallback_parity():
    """Verify C++ FFT implementation matches pure Python Gil-Pelaez quad integration."""
    S0 = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    kappa = 2.0
    theta = 0.05
    sigma = 0.3
    rho = -0.6
    V0 = 0.04
    
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    
    for K in strikes:
        py_price = _heston_quad_price_py(S0, K, T, r, q, kappa, theta, sigma, rho, V0, "call")
        cpp_price = cpp_heston(S0, K, T, r, q, kappa, theta, sigma, rho, V0, "call", 4096, 0.25, 1.5)
        
        # We use a modest tolerance (1e-3) because FFT interpolation vs Quad integration
        # will naturally have small discretization differences.
        assert math.isclose(py_price, cpp_price, rel_tol=1e-3, abs_tol=1e-3), \
            f"Mismatch at K={K}: py={py_price}, cpp={cpp_price}"
            
    # Test batch pricing
    cpp_batch_prices = cpp_heston_batch(S0, strikes, T, r, q, kappa, theta, sigma, rho, V0, "call", 4096, 0.25, 1.5)
    for K, batch_cpp_price in zip(strikes, cpp_batch_prices):
        py_price = _heston_quad_price_py(S0, K, T, r, q, kappa, theta, sigma, rho, V0, "call")
        assert math.isclose(py_price, batch_cpp_price, rel_tol=1e-3, abs_tol=1e-3)
