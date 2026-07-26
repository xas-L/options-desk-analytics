import numpy as np
import pytest

from odx.vol.vix_term_structure import vix_index_variance, build_vix_term_structure


def test_vix_flat_vol():
    """VIX variance for a flat volatility surface should approximately equal the flat vol squared."""
    flat_vol = 0.20
    T = 30.0 / 365.0
    F = 100.0
    r = 0.05
    
    def flat_surface(K):
        return flat_vol
        
    variance = vix_index_variance(T, F, r, flat_surface)
    
    # VIX variance should be very close to flat_vol**2 (0.04)
    # The integration covers the log-contract which perfectly replicates variance for flat vol.
    np.testing.assert_allclose(variance, flat_vol**2, rtol=1e-3)
    

def test_build_vix_term_structure():
    """Test term structure building converts variance to VIX points."""
    flat_vol = 0.20
    expiries = np.array([30.0, 60.0, 90.0]) / 365.0
    F_vec = np.array([100.0, 100.0, 100.0])
    r_vec = np.array([0.05, 0.05, 0.05])
    
    def flat_surface(T, K):
        return flat_vol
        
    vix_curve = build_vix_term_structure(expiries, F_vec, r_vec, flat_surface)
    
    # All points should be ~20.0 VIX points
    np.testing.assert_allclose(vix_curve, np.array([20.0, 20.0, 20.0]), rtol=1e-3)
