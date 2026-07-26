"""Andersen's Quadratic-Exponential (QE) scheme for Heston Monte Carlo."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def heston_qe_paths(
    S0: float, V0: float, kappa: float, theta: float, sigma: float, rho: float, 
    r: float, q: float, T: float, N_steps: int, N_paths: int,
    psi_c: float = 1.5
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Heston spot and variance paths using the QE scheme."""
    dt = T / N_steps
    
    S = np.zeros((N_paths, N_steps + 1))
    V = np.zeros((N_paths, N_steps + 1))
    
    S[:, 0] = S0
    V[:, 0] = V0
    
    # Precompute constants
    exp_k = np.exp(-kappa * dt)
    k1 = exp_k
    k2 = (sigma**2 * exp_k / kappa) * (1.0 - exp_k)
    k3 = (theta * sigma**2 / (2.0 * kappa)) * (1.0 - exp_k)**2
    
    gamma1 = 0.5
    gamma2 = 0.5
    
    K0 = -(rho * kappa * theta / sigma) * dt
    K1 = gamma1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = gamma2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = gamma1 * dt * (1.0 - rho**2)
    K4 = gamma2 * dt * (1.0 - rho**2)
    
    log_S = np.log(S0) * np.ones(N_paths)
    
    for t in range(1, N_steps + 1):
        V_prev = V[:, t - 1]
        
        m = theta + (V_prev - theta) * k1
        s2 = V_prev * k2 + k3
        psi = s2 / (m**2 + 1e-12)
        
        UV = np.random.uniform(size=N_paths)
        Z = np.random.normal(size=N_paths)
        
        V_next = np.zeros(N_paths)
        
        # Scheme 1: Non-central chi-square
        mask1 = psi <= psi_c
        if np.any(mask1):
            psi_m = psi[mask1]
            b2 = 2.0 / psi_m - 1.0 + np.sqrt(2.0 / psi_m) * np.sqrt(2.0 / psi_m - 1.0)
            a = m[mask1] / (1.0 + b2)
            b = np.sqrt(b2)
            Z_V = norm.ppf(UV[mask1])
            V_next[mask1] = a * (b + Z_V)**2
            
        # Scheme 2: Exponential
        mask2 = ~mask1
        if np.any(mask2):
            psi_m = psi[mask2]
            p = (psi_m - 1.0) / (psi_m + 1.0)
            beta = (1.0 - p) / m[mask2]
            U2 = UV[mask2]
            V_next[mask2] = np.where(U2 <= p, 0.0, (1.0 / beta) * np.log((1.0 - p) / (1.0 - U2)))
            
        V[:, t] = V_next
        
        # Spot update
        log_S = log_S + (r - q) * dt + K0 + K1 * V_prev + K2 * V_next + np.sqrt(K3 * V_prev + K4 * V_next) * Z
        S[:, t] = np.exp(log_S)
        
    return S, V
