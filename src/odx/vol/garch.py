"""GARCH(1,1) volatility model via Maximum Likelihood Estimation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def garch_variance(params: np.ndarray, returns: np.ndarray, initial_var: float) -> np.ndarray:
    """Generate GARCH(1,1) variance series given parameters."""
    omega, alpha, beta = params
    n = len(returns)
    var = np.zeros(n)
    if n > 0:
        var[0] = initial_var
    
    for t in range(1, n):
        var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
    return var


def garch_log_likelihood(params: np.ndarray, returns: np.ndarray, initial_var: float) -> float:
    """Calculate negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
        return 1e10
        
    var = garch_variance(params, returns, initial_var)
    
    # Avoid zero or negative variance in log
    if np.any(var <= 0):
        return 1e10
        
    # Negative log-likelihood assuming Gaussian innovations
    nll = 0.5 * np.sum(np.log(var) + returns**2 / var)
    return float(nll)


def fit_garch_11(returns: np.ndarray) -> dict:
    """Calibrate GARCH(1,1) using MLE."""
    returns = np.asarray(returns)
    initial_var = np.var(returns)
    
    # Initial guess [omega, alpha, beta]
    x0 = np.array([initial_var * 0.05, 0.1, 0.85])
    
    bounds = ((1e-8, None), (0.0, 1.0), (0.0, 1.0))
    constraints = {"type": "ineq", "fun": lambda x: 0.999 - (x[1] + x[2])}
    
    res = minimize(
        garch_log_likelihood, x0, args=(returns, initial_var),
        bounds=bounds, constraints=constraints, method="SLSQP"
    )
    
    omega, alpha, beta = res.x
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "long_run_var": float(omega / (1.0 - alpha - beta)),
        "success": bool(res.success)
    }
