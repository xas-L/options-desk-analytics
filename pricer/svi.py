"""SVI (Stochastic Volatility Inspired) raw parametrisation and slice fitter.

Raw SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
where k = log(K/F) and w = sigma_implied^2 * T (total implied variance).

Reference: Gatheral (2004), Zeliade white paper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Raw SVI total implied variance w = sigma_imp^2 * T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Fit SVI to observed (k, w) pairs. Returns (params, rmse).

    params order: [a, b, rho, m, sigma]
    """
    if weights is None:
        weights = np.ones_like(total_variance)

    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sig = params
        if b < 0 or sig <= 0 or abs(rho) >= 1.0:
            return 1e10
        w_fit = svi_total_variance(log_moneyness, a, b, rho, m, sig)
        return float(np.sum(((w_fit - total_variance) * weights) ** 2))

    x0 = np.array([np.mean(total_variance), 0.1, 0.0, 0.0, 0.1])
    bounds = [(-1.0, 1.0), (1e-6, 2.0), (-0.999, 0.999), (-2.0, 2.0), (1e-4, 2.0)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    params = result.x

    w_fit = svi_total_variance(log_moneyness, *params)
    rmse = float(np.sqrt(np.mean((w_fit - total_variance) ** 2)))
    return params, rmse


def svi_iv(
    log_moneyness: np.ndarray,
    params: np.ndarray,
    T: float,
) -> np.ndarray:
    """Convert fitted SVI total variance to annualised implied vol."""
    w = svi_total_variance(log_moneyness, *params)
    w = np.maximum(w, 0.0)
    return np.sqrt(w / T)
