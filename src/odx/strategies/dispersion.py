"""Index options dispersion strategy logic."""

from __future__ import annotations

import pandas as pd
from typing import List, Dict


def calculate_dispersion_weights(
    index_vega: float,
    constituent_vegas: Dict[str, float],
    constituent_weights: Dict[str, float],
    weighting_scheme: str = "vega"
) -> Dict[str, float]:
    """Calculate trade sizes for a dispersion trade (short index, long constituents).
    
    Args:
        index_vega: Vega of the short index position (should be positive number representing total exposure).
        constituent_vegas: Dict mapping ticker to its 1-unit vega.
        constituent_weights: Dict mapping ticker to its index weight (0.0 to 1.0).
        weighting_scheme: "vega" (vega neutral) or "theta" (theta neutral).
        
    Returns:
        Dict mapping tickers (including 'INDEX') to their trade size (number of options/straddles).
    """
    sizes = {"INDEX": -1.0}  # Sell 1 unit of index exposure
    
    total_idx_weight = sum(constituent_weights.values())
    if total_idx_weight <= 0:
        raise ValueError("Total constituent weight must be > 0.")
        
    if weighting_scheme == "vega":
        # Target vega for each constituent = Index Vega * Constituent Weight
        for ticker, weight in constituent_weights.items():
            if ticker not in constituent_vegas or constituent_vegas[ticker] == 0:
                sizes[ticker] = 0.0
                continue
                
            target_vega = index_vega * (weight / total_idx_weight)
            sizes[ticker] = target_vega / constituent_vegas[ticker]
            
    elif weighting_scheme == "notional":
        # Simplified: just trade proportional to index weight
        for ticker, weight in constituent_weights.items():
            sizes[ticker] = weight / total_idx_weight
            
    else:
        raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        
    return sizes


def calculate_implied_correlation(
    index_var: float,
    constituent_vars: List[float],
    constituent_weights: List[float]
) -> float:
    """Calculate implied correlation from index and constituent variances.
    
    Uses the approximation: 
    Var(Index) = sum(w_i^2 Var_i) + sum_{i != j} w_i w_j rho_ij sqrt(Var_i Var_j)
    Assuming uniform implied correlation rho.
    """
    if not constituent_vars or not constituent_weights:
        return 0.0
        
    w = pd.Series(constituent_weights)
    v = pd.Series(constituent_vars)
    
    # Variance of the weighted sum assuming 0 correlation
    indep_var = (w**2 * v).sum()
    
    # Denominator for uniform correlation
    # sum_{i!=j} w_i w_j vol_i vol_j = (sum w_i vol_i)^2 - sum(w_i^2 vol_i^2)
    vols = v**0.5
    w_vol_sum = (w * vols).sum()
    cross_term = (w_vol_sum**2) - indep_var
    
    if cross_term <= 0:
        return 1.0
        
    implied_corr = (index_var - indep_var) / cross_term
    
    # Bound it to [-1, 1] for safety due to numerical issues
    return float(max(-1.0, min(1.0, implied_corr)))
