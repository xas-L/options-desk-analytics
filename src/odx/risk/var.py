"""Value at Risk (VaR) models."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, chi2


def historical_var(pnl: np.ndarray, confidence_level: float = 0.99) -> float:
    """Calculate Historical VaR from a P&L vector.
    
    Returns a positive number representing the loss amount.
    """
    alpha = 1.0 - confidence_level
    return -float(np.quantile(pnl, alpha))


def parametric_var(mean_pnl: float, std_pnl: float, confidence_level: float = 0.99) -> float:
    """Calculate Parametric (Normal) VaR."""
    alpha = 1.0 - confidence_level
    z_score = norm.ppf(alpha)
    return -(mean_pnl + z_score * std_pnl)


def monte_carlo_var(simulated_pnl: np.ndarray, confidence_level: float = 0.99) -> float:
    """Calculate Monte Carlo VaR from simulated P&L."""
    return historical_var(simulated_pnl, confidence_level)
    

def kupiec_pof_test(hits: int, observations: int, confidence_level: float = 0.99) -> tuple[float, float]:
    """Kupiec's Proportion of Failures (POF) test for VaR backtesting.
    
    Returns (LR_statistic, p_value).
    """
    p = 1.0 - confidence_level
    
    if hits == 0:
        lr = -2.0 * np.log((1.0 - p)**observations)
    elif hits >= observations:
        lr = -2.0 * np.log(p**observations)
    else:
        term1 = hits * np.log(p) + (observations - hits) * np.log(1.0 - p)
        term2 = hits * np.log(hits / observations) + (observations - hits) * np.log(1.0 - hits / observations)
        lr = -2.0 * (term1 - term2)
        
    p_value = 1.0 - chi2.cdf(lr, df=1)
    
    return float(lr), float(p_value)
