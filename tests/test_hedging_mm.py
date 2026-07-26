import numpy as np
import pytest

from odx.hedging.whalley_wilmott import whalley_wilmott_band
from odx.mm.avellaneda_stoikov import reservation_price, optimal_quotes


def test_ww_band_scales_with_tc():
    """Verify WW band width scales correctly with transaction cost parameter."""
    # Base parameters
    S, K, T, r, sigma, risk_av = 100.0, 100.0, 1.0, 0.05, 0.2, 1.0
    
    # Low transaction cost
    lower1, delta1, upper1 = whalley_wilmott_band(S, K, T, r, sigma, risk_av, 0.001)
    width1 = upper1 - lower1
    
    # Higher transaction cost
    lower2, delta2, upper2 = whalley_wilmott_band(S, K, T, r, sigma, risk_av, 0.008)
    width2 = upper2 - lower2
    
    assert delta1 == delta2
    # WW band width H is proportional to cost^(1/3). 
    # So 8x cost -> 2x band width
    np.testing.assert_allclose(width2, width1 * 2.0, rtol=1e-5)


def test_as_reservation_price_shifts():
    """Verify Avellaneda-Stoikov reservation price shifts correctly with inventory sign."""
    s = 100.0
    gamma = 0.1
    sigma = 2.0
    T = 1.0
    
    # Neutral inventory
    r0 = reservation_price(s, 0, gamma, sigma, T)
    assert r0 == s
    
    # Long inventory (q > 0) -> lower reservation price to encourage selling
    r_long = reservation_price(s, 10, gamma, sigma, T)
    assert r_long < s
    
    # Short inventory (q < 0) -> higher reservation price to encourage buying
    r_short = reservation_price(s, -10, gamma, sigma, T)
    assert r_short > s
    
    # Symmetry
    assert (s - r_long) == (r_short - s)
