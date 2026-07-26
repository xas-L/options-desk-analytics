"""Implied correlation between index and constituent single-name volatilities."""

from __future__ import annotations
import numpy as np


def implied_correlation(
    index_variance: float, constituent_variances: np.ndarray, weights: np.ndarray
) -> float:
    """Calculate implied equicorrelation from index and constituent variances."""
    weighted_var_sum = np.sum((weights**2) * constituent_variances)
    
    cross_term_sum = 0.0
    n = len(weights)
    for i in range(n):
        for j in range(n):
            if i != j:
                cross_term_sum += weights[i] * weights[j] * np.sqrt(constituent_variances[i] * constituent_variances[j])
                
    if cross_term_sum <= 0:
        return np.nan
        
    rho = (index_variance - weighted_var_sum) / cross_term_sum
    return float(np.clip(rho, -1.0, 1.0))
