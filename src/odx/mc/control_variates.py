"""Control variates for Monte Carlo pricing."""

from __future__ import annotations
import numpy as np
from scipy.stats import norm


def geometric_asian_price(
    S0: float, K: float, T: float, r: float, sigma: float, n_steps: int, option_type: str = "call", q: float = 0.0
) -> float:
    """Closed-form price of a discrete geometric Asian option under Black-Scholes."""
    dt = T / n_steps
    is_call = option_type.strip().lower() in ("call", "c")
    
    # Moments of the geometric average
    mu_g = np.log(S0) + (r - q - 0.5 * sigma**2) * dt * (n_steps + 1) / 2.0
    var_g = (sigma**2 * dt / (6.0 * n_steps)) * (n_steps + 1) * (2 * n_steps + 1)
    
    sigma_g = np.sqrt(var_g)
    
    d1 = (mu_g - np.log(K) + 0.5 * var_g) / sigma_g
    d2 = d1 - sigma_g
    
    expected_G = np.exp(mu_g + 0.5 * var_g)
    disc = np.exp(-r * T)
    
    if is_call:
        price = disc * (expected_G * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = disc * (K * norm.cdf(-d2) - expected_G * norm.cdf(-d1))
        
    return float(price)


def apply_control_variate(
    target_payoffs: np.ndarray, control_payoffs: np.ndarray, control_analytic_price: float
) -> tuple[float, float]:
    """Apply standard control variate to Monte Carlo payoffs.
    
    Returns (adjusted_mc_price, standard_error).
    """
    covariance = np.cov(target_payoffs, control_payoffs)[0, 1]
    variance = np.var(control_payoffs)
    
    if variance <= 1e-12:
        c = 0.0
    else:
        c = covariance / variance
        
    adjusted_payoffs = target_payoffs - c * (control_payoffs - control_analytic_price)
    
    price = float(np.mean(adjusted_payoffs))
    se = float(np.std(adjusted_payoffs, ddof=1) / np.sqrt(len(adjusted_payoffs)))
    
    return price, se
