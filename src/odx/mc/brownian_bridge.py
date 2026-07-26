"""Brownian bridge path construction."""

from __future__ import annotations
import numpy as np


def construct_brownian_bridge(dt: float, normals: np.ndarray) -> np.ndarray:
    """Construct Brownian paths using the Brownian bridge technique.
    
    normals: array of shape (n_paths, n_steps) containing independent standard normals.
    n_steps must be a power of 2 for this dyadic bisection implementation.
    
    Returns array of shape (n_paths, n_steps + 1) with the initial state W_0 = 0.
    """
    n_paths, n_steps = normals.shape
    W = np.zeros((n_paths, n_steps + 1))
    
    # Total time
    T = n_steps * dt
    
    # Terminal point (driven by the first normal)
    W[:, -1] = np.sqrt(T) * normals[:, 0]
    
    normal_idx = 1
    step = n_steps
    
    # Iterative dyadic bisection
    while step > 1:
        half_step = step // 2
        for i in range(half_step, n_steps, step):
            left = i - half_step
            right = i + half_step
            
            # Midpoint variance
            var = half_step * dt / 2.0
            
            # Bridge interpolation
            mean = 0.5 * (W[:, left] + W[:, right])
            W[:, i] = mean + np.sqrt(var) * normals[:, normal_idx]
            normal_idx += 1
            
        step = half_step
        
    return W
