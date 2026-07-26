import numpy as np
import pytest

from odx.vol.svi import fit_svi, svi_total_variance


def test_svi_fit_recovers_synthetic_parameters():
    """Verify SVI calibration recovers known parameters from synthetic data."""
    # Known synthetic parameters
    true_params = np.array([0.04, 0.1, -0.4, 0.1, 0.2])  # [a, b, rho, m, sigma]
    
    # Synthetic log-moneyness grid
    k_grid = np.linspace(-1.0, 1.0, 50)
    
    # Generate synthetic total variance
    w_true = svi_total_variance(k_grid, *true_params)
    
    # Add a tiny bit of noise to avoid perfectly flat gradients in some optimizers, 
    # though differential_evolution is robust.
    np.random.seed(42)
    w_noisy = w_true + np.random.normal(0, 1e-6, size=w_true.shape)
    w_noisy = np.maximum(w_noisy, 1e-6)
    
    # Fit
    fitted_params, rmse, info = fit_svi(
        log_moneyness=k_grid,
        total_variance=w_noisy,
        weights="uniform",
        check_arb=True
    )
    
    # Recover parameters within tolerance
    np.testing.assert_allclose(fitted_params, true_params, rtol=1e-2, atol=1e-2)
    
    # Verify no butterfly arbitrage detected in the fit
    assert info["butterfly_arb_free"] is True
    assert rmse < 1e-4
