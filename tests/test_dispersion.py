"""Tests for dispersion strategy."""

import math
from odx.strategies.dispersion import calculate_dispersion_weights, calculate_implied_correlation


def test_dispersion_weights():
    weights = {"AAPL": 0.4, "MSFT": 0.6}
    vegas = {"AAPL": 0.1, "MSFT": 0.2}
    
    # Total weight = 1.0
    # AAPL target vega = 1.0 * 0.4 = 0.4 -> sizes = 0.4 / 0.1 = 4.0
    # MSFT target vega = 1.0 * 0.6 = 0.6 -> sizes = 0.6 / 0.2 = 3.0
    sizes = calculate_dispersion_weights(1.0, vegas, weights, "vega")
    
    assert sizes["INDEX"] == -1.0
    assert math.isclose(sizes["AAPL"], 4.0)
    assert math.isclose(sizes["MSFT"], 3.0)


def test_implied_correlation():
    # If index var == indep var, correlation is 0
    w = [0.5, 0.5]
    v = [0.04, 0.04]  # vol = 0.2
    
    indep_var = (0.5**2 * 0.04) + (0.5**2 * 0.04)  # 0.01 + 0.01 = 0.02
    
    corr_0 = calculate_implied_correlation(0.02, v, w)
    assert math.isclose(corr_0, 0.0, abs_tol=1e-5)
    
    # If correlation is 1.0, index vol = sum(w * vol) = 0.5*0.2 + 0.5*0.2 = 0.2
    # index var = 0.04
    corr_1 = calculate_implied_correlation(0.04, v, w)
    assert math.isclose(corr_1, 1.0, abs_tol=1e-5)
