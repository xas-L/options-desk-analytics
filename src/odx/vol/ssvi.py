"""Surface SVI (SSVI) with power-law term structure.

SSVI Total Variance:
w(k, theta_t) = theta_t / 2 * (1 + rho * phi(theta_t) * k + sqrt((phi(theta_t)*k + rho)^2 + (1 - rho^2)))

Power-law ATM variance term structure:
theta(t) = A * t^B

Power-law phi term structure:
phi(theta) = eta / (theta^gamma * (1 + theta)^(1 - gamma))

Arbitrage conditions (Gatheral 2014):
1. No calendar arbitrage: d/dt theta(t) >= 0.
   (Satisfied automatically if A > 0, B > 0).
2. No butterfly arbitrage:
   theta * phi(theta) * (1 + |rho|) <= 4
   theta * phi(theta)^2 * (1 + |rho|) <= 4
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


def ssvi_phi(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Power-law phi function for SSVI."""
    # Avoid division by zero
    theta_safe = np.maximum(theta, 1e-12)
    return eta / (theta_safe**gamma * (1.0 + theta_safe)**(1.0 - gamma))


def ssvi_total_variance(
    k: np.ndarray,
    t: np.ndarray,
    A: float,
    B: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Evaluate SSVI total implied variance w(k, t).
    
    Parameters
    ----------
    k : ndarray
        Log-moneyness k = log(K/F).
    t : ndarray
        Time to maturity in years (must be same shape as k).
    A, B : float
        Parameters for ATM variance theta(t) = A * t^B.
    rho : float
        Correlation parameter.
    eta, gamma : float
        Parameters for phi(theta).
        
    Returns an ndarray of total implied variance w(k, t).
    """
    t_safe = np.maximum(t, 1e-12)
    theta = A * t_safe**B
    phi = ssvi_phi(theta, eta, gamma)
    
    term = phi * k + rho
    return (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(term**2 + (1.0 - rho**2)))


def check_ssvi_arbitrage(
    A: float,
    B: float,
    rho: float,
    eta: float,
    gamma: float,
    t_grid: np.ndarray,
) -> float:
    """Check Gatheral's static arbitrage conditions over a time grid.
    
    Returns a penalty score: 0 if no arbitrage, > 0 if arbitrage is detected.
    """
    penalty = 0.0
    
    # 1. Calendar arbitrage
    if A <= 0 or B <= 0:
        penalty += 1e5
        
    if abs(rho) >= 1.0:
        penalty += 1e5
        
    if eta <= 0:
        penalty += 1e5
        
    if gamma <= 0 or gamma >= 1:
        penalty += 1e5
        
    # 2. Butterfly arbitrage
    t_safe = np.maximum(t_grid, 1e-12)
    theta = A * t_safe**B
    phi = ssvi_phi(theta, eta, gamma)
    
    cond1 = theta * phi * (1.0 + abs(rho)) - 4.0
    cond2 = theta * phi**2 * (1.0 + abs(rho)) - 4.0
    
    viol1 = np.maximum(cond1, 0.0)
    viol2 = np.maximum(cond2, 0.0)
    
    penalty += np.sum(viol1) * 1000.0
    penalty += np.sum(viol2) * 1000.0
    
    return penalty


def fit_ssvi_surface(
    df: pd.DataFrame,
    check_arb: bool = True,
) -> Tuple[np.ndarray, float, dict]:
    """Jointly calibrate SSVI parameters across the entire option chain.
    
    Assumes df has columns: 'K', 'F', 'T', 'iv'
    where 'F' is the forward price and 'iv' is the implied volatility.
    
    Returns an array of fitted parameters [A, B, rho, eta, gamma] & an rmse 
    in total variance space, and optimizer diagnostics.
    """
    # Extract data
    F = df["F"].values
    K = df["K"].values
    T = df["T"].values
    iv = df["iv"].values
    
    # Calculate log-moneyness and total variance
    k = np.log(K / F)
    w_obs = (iv**2) * T
    
    # Grid of unique expiries for arb checks
    t_grid = np.unique(T)
    t_grid = t_grid[t_grid > 0]
    
    def objective(params: np.ndarray) -> float:
        A, B, rho, eta, gamma = params
        
        w_fit = ssvi_total_variance(k, T, A, B, rho, eta, gamma)
        mse = np.mean((w_fit - w_obs)**2)
        
        if check_arb:
            penalty = check_ssvi_arbitrage(A, B, rho, eta, gamma, t_grid)
            return float(mse + penalty)
        
        return float(mse)

    # Bounds for [A, B, rho, eta, gamma]
    bounds = [
        (1e-4, 5.0),       # A
        (0.1, 2.0),        # B
        (-0.999, 0.999),   # rho
        (1e-4, 10.0),      # eta
        (0.001, 0.999),    # gamma
    ]

    result = differential_evolution(objective, bounds=bounds, seed=42)
    params = result.x
    
    # Final RMSE
    w_fit = ssvi_total_variance(k, T, *params)
    rmse = float(np.sqrt(np.mean((w_fit - w_obs)**2)))
    
    info = {
        "optimizer_result": result,
    }
    
    if check_arb:
        penalty = check_ssvi_arbitrage(params[0], params[1], params[2], params[3], params[4], t_grid)
        info["arb_penalty"] = float(penalty)
        info["arbitrage_free"] = bool(penalty == 0.0)
        
        if penalty > 0:
            warnings.warn("SSVI calibration converged but arbitrage conditions are violated.")

    return params, rmse, info
