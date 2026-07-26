"""Sobol sequence generator for quasi-Monte Carlo."""

from __future__ import annotations
import numpy as np
from scipy.stats import qmc, norm


def generate_sobol_normals(n_paths: int, n_steps: int, scramble: bool = True) -> np.ndarray:
    """Generate standard normal variables using a Sobol sequence.
    
    Returns an array of shape (n_paths, n_steps).
    For optimal uniformity, n_paths should be a power of 2.
    """
    sampler = qmc.Sobol(d=n_steps, scramble=scramble)
    
    # Generate uniform samples in (0, 1)
    uniform_samples = sampler.random(n=n_paths)
    
    # Map to standard normal (clip slightly to prevent infinities)
    uniform_samples = np.clip(uniform_samples, 1e-10, 1.0 - 1e-10)
    normal_samples = norm.ppf(uniform_samples)
    
    return normal_samples
