"""Variance swap pricer via static log-contract replication."""

from __future__ import annotations

import numpy as np
import pandas as pd


def var_swap_fair_strike(
    chain: pd.DataFrame, 
    S0: float, 
    r: float, 
    T: float,
    apply_jump_correction: bool = False
) -> float:
    """Compute fair variance swap strike (annualised) via log-contract replication.
    
    Replicates the log payoff using an integral over OTM puts and calls.
    If apply_jump_correction is True, adds a cubic term approximation for 
    the jump/discrete-monitoring replication error.
    
    Parameters
    ----------
    chain : pd.DataFrame
        Must contain columns 'K', 'cp', and 'mid'.
        Expects a single expiry slice.
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity in years.
    apply_jump_correction : bool
        Whether to apply the third-moment jump correction.
    """
    df = chain.sort_values("K").copy()
    if "mid" not in df.columns and "price" in df.columns:
        df["mid"] = df["price"]
        
    K = df["K"].values
    cp = df["cp"].str.lower().values
    prices = df["mid"].values
    
    if len(K) < 3:
        return np.nan
        
    F = S0 * np.exp(r * T)
    
    # Find at-the-money forward strike
    atm_idx = np.abs(K - F).argmin()
    K_star = K[atm_idx]
    
    # Select OTM options
    is_otm_put = (K <= K_star) & np.isin(cp, ["put", "p"])
    is_otm_call = (K > K_star) & np.isin(cp, ["call", "c"])
    valid_mask = is_otm_put | is_otm_call
    
    # Trapezoidal integration weights (dK)
    dK = np.zeros_like(K, dtype=float)
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    
    # Log-contract replication weights: 1 / K^2
    weights = np.zeros_like(K, dtype=float)
    weights[valid_mask] = dK[valid_mask] / (K[valid_mask]**2)
    
    integral = np.sum(weights * prices)
    
    # Forward contribution term
    fwd_term = (F / K_star - 1.0) - np.log(F / K_star)
    
    # Base continuous variance strike
    var_strike = (2.0 / T) * np.exp(r * T) * integral - (2.0 / T) * fwd_term
    
    if apply_jump_correction:
        # Simple cubic expansion approximation for jump replication error
        # Weighting OTM options by (K/F - 1) * 1/K^2
        jump_weights = weights * (K / F - 1.0)
        jump_integral = np.sum(jump_weights * prices)
        var_strike += (2.0 / T) * np.exp(r * T) * jump_integral
        
    return float(var_strike)
