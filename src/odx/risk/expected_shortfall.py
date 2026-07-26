"""Expected Shortfall (ES) backtesting."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def historical_es(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical Expected Shortfall at confidence level 1 - alpha.
    
    Returns the average loss beyond the historical VaR. 
    Result is positive for a loss.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return np.nan
        
    var = np.percentile(returns, alpha * 100)
    tail_losses = returns[returns <= var]
    
    if len(tail_losses) == 0:
        return np.nan
        
    return float(-np.mean(tail_losses))


def parametric_es(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """Parametric (Normal) Expected Shortfall.
    
    Returns the expected loss beyond the parametric VaR.
    Result is positive for a loss.
    """
    if sigma <= 0:
        return np.nan
        
    # Standard normal PDF at the VaR quantile
    z_alpha = norm.ppf(alpha)
    pdf_val = norm.pdf(z_alpha)
    
    # ES for normal dist: -mu + sigma * pdf(z_alpha) / alpha
    es = -mu + sigma * (pdf_val / alpha)
    return float(es)
