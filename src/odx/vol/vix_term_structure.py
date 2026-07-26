"""VIX-style model-free variance replication from an SSVI surface."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from odx.pricers.analytic.bs import bs_price


def vix_index_variance(
    T: float, F: float, r: float, vol_func: callable
) -> float:
    """Calculate VIX-style variance via continuous model-free replication.
    
    vol_func(K) should return the SSVI implied volatility for strike K.
    """
    if T <= 0:
        return np.nan
        
    # We use F as S and q=r so that the forward price equals F
    # F = S * exp((r-q)T) -> F = S * 1 = S
    
    def integrand(K: float) -> float:
        iv = vol_func(K)
        if K < F:
            # OTM Put
            price = bs_price(F, K, T, r, iv, option_type="put", q=r)
        else:
            # OTM Call
            price = bs_price(F, K, T, r, iv, option_type="call", q=r)
            
        return price / (K**2)
        
    # Use 5 std devs based on ATM vol for integration bounds
    atm_vol = vol_func(F)
    std_dev = atm_vol * np.sqrt(T)
    lower_bound = F * np.exp(-5.0 * std_dev)
    upper_bound = F * np.exp(5.0 * std_dev)
    
    integral, _ = quad(integrand, lower_bound, upper_bound, epsabs=1e-5, epsrel=1e-5)
    
    variance = (2.0 / T) * np.exp(r * T) * integral
    return float(variance)


def build_vix_term_structure(
    expiries: np.ndarray, F_vec: np.ndarray, r_vec: np.ndarray, surface_func: callable
) -> np.ndarray:
    """Build VIX term structure across multiple expiries.
    
    Returns standard VIX points (percentage, e.g. 20.0 for 20%).
    """
    variances = []
    for T, F, r in zip(expiries, F_vec, r_vec):
        vol_func = lambda K: surface_func(T, K)
        var = vix_index_variance(T, F, r, vol_func)
        variances.append(var)
        
    return np.sqrt(np.array(variances)) * 100.0
