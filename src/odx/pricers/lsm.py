"""Longstaff-Schwartz least-squares Monte Carlo for American options."""

from __future__ import annotations

import numpy as np


def lsm_american_price(
    paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: str = "put",
    degree: int = 2
) -> float:
    """Longstaff-Schwartz LSM for American options.
    
    Uses Laguerre polynomials of specified degree as the regression basis 
    on in-the-money paths (standard for LSM).
    
    Parameters
    ----------
    paths : ndarray of shape (N_sim, N_steps)
        Simulated price paths.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    option_type : str
        'call' or 'put'.
    degree : int
        Degree of the Laguerre polynomial basis.
    """
    is_call = option_type.strip().lower() in ("call", "c")
    N_sim, N_steps = paths.shape
    dt = T / (N_steps - 1)
    disc = np.exp(-r * dt)

    if is_call:
        intrinsic = np.maximum(paths - K, 0.0)
    else:
        intrinsic = np.maximum(K - paths, 0.0)

    # Value matrix, tracks realized payoff on each path
    V = np.zeros_like(intrinsic)
    V[:, -1] = intrinsic[:, -1]

    # Step backwards
    for t in range(N_steps - 2, 0, -1):
        itm = intrinsic[:, t] > 0
        if not np.any(itm):
            V[:, t] = V[:, t+1] * disc
            continue
        
        # Discount future realized cash flows
        Y = V[itm, t+1] * disc
        X = paths[itm, t]
        
        # Regression basis: Laguerre polynomials
        coef = np.polynomial.laguerre.lagfit(X, Y, degree)
        continuation = np.polynomial.laguerre.lagval(X, coef)
        
        # Exercise decision
        exercise = intrinsic[itm, t] > continuation
        
        # Update value matrix
        V[:, t] = V[:, t+1] * disc
        V[itm, t] = np.where(exercise, intrinsic[itm, t], V[itm, t])

    # Time 0: average over discounted step 1 values
    return float(np.mean(V[:, 1] * disc))
