"""Tests for Longstaff-Schwartz Monte Carlo."""

import numpy as np

from odx.pricers.lsm import lsm_american_price


def test_lsm_american_put_positive():
    """Test LSM returns a reasonable positive price for an ITM put."""
    # Deterministic simple paths just to check logic runs
    N_sim = 100
    N_steps = 50
    paths = np.ones((N_sim, N_steps)) * 100.0
    # Add some noise to make paths distinct for regression
    np.random.seed(42)
    paths += np.random.randn(N_sim, N_steps) * 5.0
    
    K = 110.0
    r = 0.05
    T = 1.0
    
    price = lsm_american_price(paths, K, r, T, option_type="put")
    assert price > 0.0
    assert price <= K # American put is bounded by K


def test_lsm_american_call_positive():
    """Test LSM returns a reasonable positive price for an ITM call."""
    N_sim = 100
    N_steps = 50
    paths = np.ones((N_sim, N_steps)) * 100.0
    np.random.seed(42)
    paths += np.random.randn(N_sim, N_steps) * 5.0
    
    K = 90.0
    r = 0.05
    T = 1.0
    
    price = lsm_american_price(paths, K, r, T, option_type="call")
    assert price > 0.0
