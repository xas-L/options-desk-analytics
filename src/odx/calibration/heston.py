"""Heston parameter calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from odx.pricers.heston_fft import heston_fft_price


def fit_heston_surface(
    chain: pd.DataFrame, S0: float, r: float, q: float
) -> tuple[np.ndarray, float]:
    """Calibrate Heston [kappa, theta, sigma, rho, V0] using least squares.
    
    chain needs columns: 'K', 'T', 'price' (or 'mid'), 'cp'
    """
    df = chain.copy()
    if "mid" not in df.columns and "price" in df.columns:
        df["mid"] = df["price"]
        
    # Initial guess: [kappa, theta, sigma, rho, V0]
    x0 = np.array([2.0, 0.04, 0.3, -0.5, 0.04])
    bounds = (
        [0.01, 0.001, 0.01, -0.99, 0.001],
        [10.0, 1.0, 2.0, 0.99, 1.0]
    )
    
    def residuals(params: np.ndarray) -> np.ndarray:
        kappa, theta, sigma, rho, V0 = params
        res = []
        for _, row in df.iterrows():
            model_price = heston_fft_price(
                S0, row["K"], row["T"], r, q,
                kappa, theta, sigma, rho, V0,
                option_type=row["cp"]
            )
            res.append(model_price - row["mid"])
            
        # Soft penalty for Feller condition (2 * kappa * theta > sigma^2)
        feller = 2.0 * kappa * theta - sigma**2
        if feller < 0:
            res.extend([-feller * 10.0] * 5)
            
        return np.array(res)
        
    result = least_squares(residuals, x0, bounds=bounds, method="trf")
    rmse = float(np.sqrt(np.mean(result.fun**2)))
    return result.x, rmse
