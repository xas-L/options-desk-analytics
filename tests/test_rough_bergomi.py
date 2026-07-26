import numpy as np
from odx.vol.rough_bergomi import RoughBergomiMC, calibrate_rbergomi_smile

def test_rough_bergomi_calibration():
    """Test that rough Bergomi calibration recovers parameters from a synthetic smile."""
    S0 = 100.0
    T = 1.0
    xi = 0.04  # Flat forward variance of 20% vol
    
    # Target parameters
    true_H = 0.15
    true_eta = 2.0
    true_rho = -0.6
    
    # Generate synthetic market data
    mc = RoughBergomiMC(true_H, true_eta, true_rho, xi)
    # Use larger path count for data generation to minimize MC noise in target
    paths = mc.generate_paths(S0, T, N_steps=50, N_paths=10000, seed=42)
    
    strikes = np.array([90.0, 100.0, 110.0])
    market_prices = np.zeros(len(strikes))
    
    for i, K in enumerate(strikes):
        market_prices[i] = np.mean(np.maximum(paths[:, -1] - K, 0.0))
        
    # Calibrate back
    # Use smaller path count for calibration speed
    res = calibrate_rbergomi_smile(
        S0, T, strikes, market_prices, xi, 
        N_steps=50, N_paths=10000
    )
    
    assert res["success"] or res["mse"] < 1e-2, "Calibration failed to converge closely."
    
    # Check parameter recovery (allow reasonable tolerance due to MC noise/grid)
    assert np.abs(res["H"] - true_H) < 0.1
    assert np.abs(res["eta"] - true_eta) < 0.5
    assert np.abs(res["rho"] - true_rho) < 0.2
