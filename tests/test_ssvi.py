import numpy as np
import pandas as pd
import pytest

from odx.vol.ssvi import fit_ssvi_surface, check_ssvi_arbitrage, ssvi_total_variance


def test_ssvi_calibration_no_calendar_arbitrage():
    """Verify SSVI calibration produces a surface with no calendar arbitrage violations."""
    # Create synthetic options chain spanning multiple expiries
    expiries = [0.1, 0.5, 1.0, 2.0]
    strikes = np.linspace(80, 120, 21)
    F = 100.0
    
    # Known SSVI parameters
    true_A = 0.04
    true_B = 1.0
    true_rho = -0.5
    true_eta = 1.0
    true_gamma = 0.5
    
    data = []
    for T in expiries:
        for K in strikes:
            k = np.log(K / F)
            # True total variance
            w = ssvi_total_variance(
                np.array([k]), np.array([T]), 
                true_A, true_B, true_rho, true_eta, true_gamma
            )[0]
            
            iv = np.sqrt(max(w, 1e-8) / T)
            data.append({
                "K": K,
                "F": F,
                "T": T,
                "iv": iv
            })
            
    df = pd.DataFrame(data)
    
    # Fit SSVI
    fitted_params, rmse, info = fit_ssvi_surface(df, check_arb=True)
    
    # Verify the calibration penalty ensures no arbitrage
    assert info["arbitrage_free"]
    assert info["arb_penalty"] == 0.0
    
    A, B, rho, eta, gamma = fitted_params
    
    # Check calendar arbitrage structurally holds (A > 0, B > 0)
    assert A > 0
    assert B > 0
    
    # Check butterfly arbitrage condition explicitly over a grid
    t_grid = np.array(expiries)
    penalty = check_ssvi_arbitrage(A, B, rho, eta, gamma, t_grid)
    assert penalty == 0.0
    
    # Check we recovered the original parameters reasonably well
    np.testing.assert_allclose(
        [A, B, rho, eta, gamma], 
        [true_A, true_B, true_rho, true_eta, true_gamma], 
        rtol=1e-1, atol=1e-2
    )
