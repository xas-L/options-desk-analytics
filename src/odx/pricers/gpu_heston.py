"""GPU-accelerated Monte Carlo Heston pricer with CPU fallback."""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = np  # Fallback to numpy
    HAS_GPU = False

from typing import Tuple


def mc_heston_paths(
    S0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    V0: float,
    T: float,
    num_paths: int = 100_000,
    num_steps: int = 100
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Generate Heston paths using Euler-Maruyama with full truncation.
    
    Returns:
        Tuple of (S_paths, V_paths) with shape (num_paths, num_steps + 1).
    """
    dt = T / num_steps
    sqrt_dt = cp.sqrt(dt)
    
    # Pre-allocate arrays
    S = cp.zeros((num_paths, num_steps + 1), dtype=cp.float32)
    V = cp.zeros((num_paths, num_steps + 1), dtype=cp.float32)
    
    S[:, 0] = S0
    V[:, 0] = V0
    
    # Generate correlated brownian motions
    # Z1 is for asset, Z2 is for variance (correlated)
    # Z2 = rho * Z1 + sqrt(1 - rho**2) * Z3
    Z1 = cp.random.standard_normal((num_paths, num_steps)).astype(cp.float32)
    Z3 = cp.random.standard_normal((num_paths, num_steps)).astype(cp.float32)
    Z2 = rho * Z1 + cp.sqrt(1.0 - rho**2) * Z3
    
    for t in range(num_steps):
        S_t = S[:, t]
        V_t = V[:, t]
        
        # Full truncation for negative variance
        V_pos = cp.maximum(V_t, 0.0)
        
        # Asset evolution
        S[:, t+1] = S_t * cp.exp((r - q - 0.5 * V_pos) * dt + cp.sqrt(V_pos) * sqrt_dt * Z1[:, t])
        
        # Variance evolution (Euler-Maruyama)
        V[:, t+1] = V_t + kappa * (theta - V_pos) * dt + sigma * cp.sqrt(V_pos) * sqrt_dt * Z2[:, t]
        
    return S, V


def mc_heston_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    V0: float,
    T: float,
    option_type: str = "call",
    num_paths: int = 100_000,
    num_steps: int = 100
) -> float:
    """Price European option under Heston using GPU-accelerated Monte Carlo."""
    S, _ = mc_heston_paths(S0, r, q, kappa, theta, sigma, rho, V0, T, num_paths, num_steps)
    S_T = S[:, -1]
    
    if option_type.lower() in ("call", "c"):
        payoffs = cp.maximum(S_T - K, 0.0)
    else:
        payoffs = cp.maximum(K - S_T, 0.0)
        
    discounted_expected_payoff = cp.exp(-r * T) * cp.mean(payoffs)
    return float(discounted_expected_payoff)
