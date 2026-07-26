"""Realised variance and bipower variation estimators."""

from __future__ import annotations

import numpy as np


def realised_variance(prices: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised realised variance from a price series (log returns squared)."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return np.nan
        
    log_rets = np.log(prices[1:] / prices[:-1])
    rv = np.sum(log_rets**2) * (ann_factor / len(log_rets))
    return float(rv)


def bipower_variation(prices: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualised bipower variation (jump-robust continuous variance)."""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 3:
        return np.nan
        
    log_rets = np.abs(np.log(prices[1:] / prices[:-1]))
    
    # E[|Z|] = sqrt(2/pi) for standard normal Z, correction factor is pi/2
    bpv = (np.pi / 2.0) * np.sum(log_rets[1:] * log_rets[:-1])
    
    # Annualise using N-1 since there are N-1 adjacent pairs
    return float(bpv * (ann_factor / (len(log_rets) - 1)))
