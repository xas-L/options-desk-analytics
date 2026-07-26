"""GPU-accelerated Monte Carlo Rough Bergomi pricer with CPU fallback."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np  # Fallback to numpy
    HAS_GPU = False

from typing import Tuple


def fbm_covariance_matrix(num_steps: int, dt: float, H: float) -> np.ndarray:
    """Construct exact covariance matrix for fractional Brownian motion.
    
    Using exact CPU numpy since Cholesky on CPU is fine for small N.
    """
    times = np.linspace(dt, num_steps * dt, num_steps)
    C = np.zeros((num_steps, num_steps))
    for i in range(num_steps):
        for j in range(num_steps):
            ti = times[i]
            tj = times[j]
            C[i, j] = 0.5 * (ti**(2*H) + tj**(2*H) - abs(ti - tj)**(2*H))
    return C


def mc_rough_bergomi_paths(
    S0: float,
    r: float,
    q: float,
    xi: float,
    eta: float,
    H: float,
    rho: float,
    T: float,
    num_paths: int = 100_000,
    num_steps: int = 100
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Generate Rough Bergomi paths using exact Cholesky for fBm.
    
    Args:
        xi: Initial forward variance curve (assumed flat for simplicity).
        eta: Vol of vol.
        H: Hurst parameter (0 < H < 0.5).
        rho: Correlation between spot and variance brownian motions.
    """
    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    
    # Pre-compute exact Cholesky decomposition of fBm covariance on CPU
    C = fbm_covariance_matrix(num_steps, dt, H)
    L = np.linalg.cholesky(C + 1e-8 * np.eye(num_steps))
    L_gpu = cp.array(L, dtype=cp.float32)
    
    times = cp.linspace(dt, T, num_steps, dtype=cp.float32)
    
    # Draw independent standard normals
    Z1 = cp.random.standard_normal((num_paths, num_steps)).astype(cp.float32)  # For spot
    Z_fbm = cp.random.standard_normal((num_paths, num_steps)).astype(cp.float32) # For variance
    
    # Correlate standard normals before fractional transformation
    # W1 is standard Brownian motion increments for spot
    # W2 is the correlated normal for variance
    W2 = rho * Z1 + cp.sqrt(1.0 - rho**2) * Z_fbm
    
    # Generate fBm paths
    # L_gpu shape is (num_steps, num_steps). W2 shape is (num_paths, num_steps).
    # We want W_H shape (num_paths, num_steps)
    W_H = cp.dot(W2, L_gpu.T)
    
    # Pre-allocate paths
    S = cp.zeros((num_paths, num_steps + 1), dtype=cp.float32)
    V = cp.zeros((num_paths, num_steps + 1), dtype=cp.float32)
    
    S[:, 0] = S0
    V[:, 0] = xi
    
    # Construct variance paths
    # V_t = xi * exp(eta * W_H(t) - 0.5 * eta^2 * t^(2H))
    drift_V = -0.5 * (eta**2) * (times**(2*H))
    V[:, 1:] = xi * cp.exp(eta * W_H + drift_V)
    
    # Construct spot paths using Euler on log-spot
    log_S = cp.zeros(num_paths, dtype=cp.float32) + cp.log(S0)
    for t in range(num_steps):
        v_t = V[:, t]
        log_S += (r - q - 0.5 * v_t) * dt + cp.sqrt(v_t) * sqrt_dt * Z1[:, t]
        S[:, t+1] = cp.exp(log_S)
        
    return S, V


def mc_rough_bergomi_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    xi: float,
    eta: float,
    H: float,
    rho: float,
    T: float,
    option_type: str = "call",
    num_paths: int = 100_000,
    num_steps: int = 100
) -> float:
    """Price European option under Rough Bergomi using GPU-accelerated Monte Carlo."""
    S, _ = mc_rough_bergomi_paths(S0, r, q, xi, eta, H, rho, T, num_paths, num_steps)
    S_T = S[:, -1]
    
    if option_type.lower() in ("call", "c"):
        payoffs = cp.maximum(S_T - K, 0.0)
    else:
        payoffs = cp.maximum(K - S_T, 0.0)
        
    discounted_expected_payoff = cp.exp(-r * T) * cp.mean(payoffs)
    return float(discounted_expected_payoff)
