"""Tests for native C++ Black-Scholes bindings."""

import numpy as np
import pytest

from odx.pricers import cpp_bindings as cpp
from odx.pricers.analytic import bs as py_bs
from odx.greeks import analytic as py_greeks


def test_has_cpp_pricer():
    """Ensure the native C++ module loaded successfully."""
    assert cpp.HAS_CPP_PRICER, "Native C++ pricer failed to load. Run build_cpp.ps1."


def test_bs_price_matches_python():
    """Check C++ price exactly matches Python reference."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    cpp_call = cpp.bs_price(S, K, T, r, sigma, "call", 0.0)
    py_call = py_bs.bs_price(S, K, T, r, sigma, "call", 0.0)
    np.testing.assert_allclose(cpp_call, py_call, rtol=1e-8)
    
    cpp_put = cpp.bs_price(S, K, T, r, sigma, "put", 0.0)
    py_put = py_bs.bs_price(S, K, T, r, sigma, "put", 0.0)
    np.testing.assert_allclose(cpp_put, py_put, rtol=1e-8)


@pytest.mark.parametrize("S, K, T, r, sigma, q", [
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0),
    (100.0, 90.0, 0.5, 0.02, 0.3, 0.01),
    (100.0, 110.0, 2.0, 0.10, 0.15, 0.05),
])
def test_bs_greeks_match_python(S, K, T, r, sigma, q):
    """Check all C++ Greeks exactly match Python reference."""
    for opt_type in ["call", "put"]:
        # Delta
        np.testing.assert_allclose(
            cpp.bs_delta(S, K, T, r, sigma, opt_type, q),
            py_greeks.bs_delta(S, K, T, r, sigma, opt_type, q),
            rtol=1e-8
        )
        
        # Gamma
        np.testing.assert_allclose(
            cpp.bs_gamma(S, K, T, r, sigma, q),
            py_greeks.bs_gamma(S, K, T, r, sigma, q),
            rtol=1e-8
        )
        
        # Vega
        np.testing.assert_allclose(
            cpp.bs_vega(S, K, T, r, sigma, q),
            py_greeks.bs_vega(S, K, T, r, sigma, q),
            rtol=1e-8
        )
        
        # Theta
        np.testing.assert_allclose(
            cpp.bs_theta(S, K, T, r, sigma, opt_type, q),
            py_greeks.bs_theta(S, K, T, r, sigma, opt_type, q),
            rtol=1e-8
        )
        
        # Rho
        np.testing.assert_allclose(
            cpp.bs_rho(S, K, T, r, sigma, opt_type, q),
            py_greeks.bs_rho(S, K, T, r, sigma, opt_type, q),
            rtol=1e-8
        )
