"""Rough Bergomi Monte Carlo path generator and calibration."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class RoughBergomiMC:
    """Monte Carlo path generator for the rough Bergomi model."""
    
    def __init__(
        self,
        H: float,
        eta: float,
        rho: float,
        xi: float,
    ):
        """
        Parameters
        ----------
        H : float
            Hurst parameter (H < 0.5 for rough volatility).
        eta : float
            Volatility of volatility.
        rho : float
            Correlation between spot and variance Brownian motions.
        xi : float
            Forward variance curve level (assumed flat).
        """
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi = xi

    def generate_paths(self, S0: float, T: float, N_steps: int, N_paths: int, seed: int = 42) -> np.ndarray:
        """Generate spot price paths under the rBergomi model using Volterra discretization."""
        np.random.seed(seed)
        dt = T / N_steps
        
        # Base Brownian increments
        Z1 = np.random.normal(size=(N_paths, N_steps))
        Z2_ortho = np.random.normal(size=(N_paths, N_steps))
        Z2 = self.rho * Z1 + np.sqrt(1.0 - self.rho**2) * Z2_ortho
        
        # Volterra process for log-variance (fractional Brownian motion approximation)
        alpha = self.H - 0.5
        j_idx = np.arange(1, N_steps + 1)
        weights = (j_idx - 0.5) ** alpha
        
        # Convolution matrix L
        L = np.zeros((N_steps, N_steps))
        for i in range(N_steps):
            for j in range(i + 1):
                L[i, j] = weights[i - j]
                
        # Vectorized path generation
        W_H_steps = Z1 @ L.T * np.sqrt(2 * self.H * dt**(2 * self.H))
        W_H = np.hstack((np.zeros((N_paths, 1)), W_H_steps))
        
        # Variance paths
        t_grid = np.linspace(0, T, N_steps + 1)
        # v_t = xi * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})
        v = self.xi * np.exp(self.eta * W_H - 0.5 * self.eta**2 * t_grid**(2 * self.H))
        
        # Spot paths
        paths = np.zeros((N_paths, N_steps + 1))
        paths[:, 0] = S0
        
        for i in range(N_steps):
            paths[:, i+1] = paths[:, i] * np.exp(-0.5 * v[:, i] * dt + np.sqrt(v[:, i] * dt) * Z2[:, i])
            
        return paths


def calibrate_rbergomi_smile(
    S0: float, 
    T: float, 
    strikes: np.ndarray, 
    market_prices: np.ndarray, 
    xi: float,
    N_steps: int = 50, 
    N_paths: int = 5000
) -> dict:
    """Calibrate rBergomi parameters (H, eta, rho) to a single expiry smile using MC pricing."""
    
    def objective(params):
        H, eta, rho = params
        if H <= 0 or H >= 0.5 or eta <= 0 or rho <= -0.999 or rho >= 0.999:
            return 1e6
            
        mse = 0.0
        mc = RoughBergomiMC(H, eta, rho, xi)
        # Fixed seed ensures smooth gradient mapping for L-BFGS-B
        paths = mc.generate_paths(S0, T, N_steps, N_paths, seed=42)
        
        for i, K in enumerate(strikes):
            payoff = np.maximum(paths[:, -1] - K, 0.0)
            model_price = np.mean(payoff)
            mse += (model_price - market_prices[i])**2
            
        return mse

    x0 = np.array([0.1, 1.5, -0.7])
    bounds = ((0.01, 0.49), (0.1, 5.0), (-0.99, 0.99))
    
    # Nelder-Mead handles MC noise better than gradient-based methods
    res = minimize(objective, x0, bounds=bounds, method="Nelder-Mead", options={'xatol': 1e-3, 'fatol': 1e-3})
    
    return {
        "H": float(res.x[0]),
        "eta": float(res.x[1]),
        "rho": float(res.x[2]),
        "success": bool(res.success),
        "mse": float(res.fun)
    }
