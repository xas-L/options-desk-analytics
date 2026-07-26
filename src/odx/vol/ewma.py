"""Exponentially Weighted Moving Average (EWMA) volatility model."""

from __future__ import annotations

import numpy as np


def ewma_variance(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """Calculate EWMA variance series from returns."""
    n = len(returns)
    var = np.zeros(n)
    
    if n > 0:
        var[0] = returns[0]**2
        
    for t in range(1, n):
        var[t] = lambda_ * var[t-1] + (1.0 - lambda_) * returns[t-1]**2
        
    return var


def ewma_volatility(returns: np.ndarray, lambda_: float = 0.94, annualization: float = 252.0) -> np.ndarray:
    """Calculate annualized EWMA volatility series."""
    var = ewma_variance(returns, lambda_)
    return np.sqrt(var * annualization)
