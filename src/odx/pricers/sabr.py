"""Hagan's SABR analytic implied volatility approximation."""

from __future__ import annotations

import numpy as np


def sabr_implied_vol(
    F: float, K: float, T: float, 
    alpha: float, beta: float, rho: float, nu: float
) -> float:
    """Hagan's asymptotic implied log-normal volatility for SABR."""
    if K <= 0 or F <= 0:
        return np.nan
        
    if F == K:
        # ATM formula
        term1 = ((1.0 - beta)**2 / 24.0) * (alpha**2 / F**(2.0 - 2.0*beta))
        term2 = (rho * beta * nu * alpha) / (4.0 * F**(1.0 - beta))
        term3 = ((2.0 - 3.0*rho**2) / 24.0) * nu**2
        return float((alpha / F**(1.0 - beta)) * (1.0 + (term1 + term2 + term3) * T))
        
    f_k = F * K
    log_fk = np.log(F / K)
    
    z = (nu / alpha) * (f_k**((1.0 - beta) / 2.0)) * log_fk
    x_z = np.log((np.sqrt(1.0 - 2.0*rho*z + z**2) + z - rho) / (1.0 - rho))
    
    # Avoid div by zero
    if abs(z) < 1e-8:
        x_z = z
        
    term1 = ((1.0 - beta)**2 / 24.0) * (alpha**2 / (f_k**(1.0 - beta)))
    term2 = (rho * beta * nu * alpha) / (4.0 * (f_k**((1.0 - beta)/2.0)))
    term3 = ((2.0 - 3.0*rho**2) / 24.0) * nu**2
    
    den1 = f_k**((1.0 - beta) / 2.0)
    den2 = 1.0 + ((1.0 - beta)**2 / 24.0) * log_fk**2 + ((1.0 - beta)**4 / 1920.0) * log_fk**4
    
    multiplier = z / x_z if x_z != 0 else 1.0
    
    iv = (alpha / (den1 * den2)) * multiplier * (1.0 + (term1 + term2 + term3) * T)
    return float(iv)
