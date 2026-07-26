"""Variance and volatility swaps fair value."""

from __future__ import annotations

import numpy as np


def heston_variance_swap(kappa: float, theta: float, V0: float, T: float) -> float:
    """Fair strike of a variance swap under the Heston model."""
    if kappa <= 0 or T <= 0:
        return np.nan
        
    # Expected variance E[1/T \int_0^T V_t dt]
    var_strike = theta + (V0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T)
    return float(var_strike)


def heston_volatility_swap(kappa: float, theta: float, sigma: float, V0: float, T: float) -> float:
    """Fair strike of a volatility swap under Heston using convexity approximation."""
    if kappa <= 0 or T <= 0:
        return np.nan
        
    var_strike = heston_variance_swap(kappa, theta, V0, T)
    
    # Variance of integrated variance
    kT = kappa * T
    var_int_V = (sigma**2 / (kappa**3 * T**2)) * (
        V0 * 2.0 * (1.0 - np.exp(-kT) - kT * np.exp(-kT)) +
        theta * (2.0 * kT - 3.0 + 4.0 * np.exp(-kT) - np.exp(-2.0 * kT))
    )
    
    # Convexity adjustment: E[sqrt(V)] ≈ sqrt(E[V]) - Var(V) / (8 * E[V]^(3/2))
    vol_strike = np.sqrt(var_strike) - var_int_V / (8.0 * var_strike**(1.5))
    return float(vol_strike)
