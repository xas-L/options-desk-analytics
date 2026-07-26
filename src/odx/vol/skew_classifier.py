"""Classify volatility skew shapes from SVI parameters."""

from __future__ import annotations


def classify_svi_skew(a: float, b: float, rho: float, m: float, sigma: float) -> str:
    """Classify the shape of the SVI implied volatility smile.
    
    SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    if b <= 1e-5:
        return "flat"
        
    if abs(rho) < 0.1:
        return "smile"
        
    if rho <= -0.1:
        return "smirk_down"  # Typical equity skew
        
    if rho >= 0.1:
        return "smirk_up"    # Sometimes seen in commodities
        
    return "unknown"
