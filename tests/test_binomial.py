"""Tests for the binomial tree pricer."""

import pytest

from odx.pricers.analytic.bs import bs_price
from odx.pricers.binomial import crr_price


def test_crr_converges_to_bs_call():
    """Test European call converges to Black-Scholes as steps increase."""
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    q = 0.02

    bs = bs_price(S, K, T, r, sigma, option_type="call", q=q)
    crr = crr_price(S, K, T, r, sigma, n_steps=2000, option_type="call", exercise_style="european", q=q)
    
    assert abs(crr - bs) < 1e-2


def test_crr_converges_to_bs_put():
    """Test European put converges to Black-Scholes as steps increase."""
    S = 100.0
    K = 90.0
    T = 0.5
    r = 0.05
    sigma = 0.30
    q = 0.0

    bs = bs_price(S, K, T, r, sigma, option_type="put", q=q)
    crr = crr_price(S, K, T, r, sigma, n_steps=2000, option_type="put", exercise_style="european", q=q)
    
    assert abs(crr - bs) < 1e-2


def test_american_put_early_exercise():
    """Test American put has a higher price than European put due to early exercise premium."""
    S = 100.0
    K = 110.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    q = 0.0

    european = crr_price(S, K, T, r, sigma, n_steps=200, option_type="put", exercise_style="european", q=q)
    american = crr_price(S, K, T, r, sigma, n_steps=200, option_type="put", exercise_style="american", q=q)
    
    assert american > european + 0.1
