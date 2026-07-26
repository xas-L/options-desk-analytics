import numpy as np
import pytest

from odx.mc.sobol import generate_sobol_normals
from odx.mc.brownian_bridge import construct_brownian_bridge
from odx.mc.control_variates import geometric_asian_price, apply_control_variate


def test_sobol_normals_shape():
    normals = generate_sobol_normals(4096, 4)
    assert normals.shape == (4096, 4)
    # Mean should be close to 0
    assert np.abs(np.mean(normals)) < 0.05
    

def test_brownian_bridge():
    n_paths = 100
    n_steps = 4 # power of 2
    dt = 0.25
    normals = np.random.normal(size=(n_paths, n_steps))
    
    W = construct_brownian_bridge(dt, normals)
    assert W.shape == (n_paths, 5)
    assert np.all(W[:, 0] == 0.0)
    # The terminal point should exactly match the first normal scaled by sqrt(T)
    np.testing.assert_allclose(W[:, -1], np.sqrt(1.0) * normals[:, 0])


def test_geometric_asian_call():
    S0, K, T, r, sigma, n_steps = 100.0, 100.0, 1.0, 0.05, 0.2, 50
    price = geometric_asian_price(S0, K, T, r, sigma, n_steps, "call")
    assert price > 0.0
    
    put = geometric_asian_price(S0, K, T, r, sigma, n_steps, "put")
    assert put > 0.0
    

def test_apply_control_variate():
    # Synthetic target and control
    target = np.random.normal(10.0, 2.0, 1000)
    control = target + np.random.normal(0, 0.5, 1000)
    
    true_control_mean = np.mean(control) + 0.1 # slightly shifted analytic mean
    
    adj_mean, adj_se = apply_control_variate(target, control, true_control_mean)
    assert np.isfinite(adj_mean)
    assert adj_se > 0.0
