"""Tests for Markov regime-switching variance model."""

import math
from odx.vol.regime_switching import MarkovRegimeSwitchingVariance


def test_regime_switching_stationary():
    """Test stationary distribution and unconditional variance calculations."""
    model = MarkovRegimeSwitchingVariance(
        var_normal=0.04,
        var_crisis=0.25,
        p_normal_to_crisis=0.01,
        p_crisis_to_normal=0.09
    )
    
    # pi_1 = 0.01 / 0.10 = 0.1
    # pi_0 = 0.09 / 0.10 = 0.9
    pi = model.stationary_distribution()
    assert math.isclose(pi[0], 0.9)
    assert math.isclose(pi[1], 0.1)
    
    # Unconditional variance = 0.9 * 0.04 + 0.1 * 0.25 = 0.036 + 0.025 = 0.061
    u_var = model.unconditional_variance()
    assert math.isclose(u_var, 0.061)


def test_simulate_regime_path():
    """Test basic simulation logic."""
    # Deterministic switching
    model = MarkovRegimeSwitchingVariance(
        var_normal=0.04,
        var_crisis=0.25,
        p_normal_to_crisis=1.0,
        p_crisis_to_normal=1.0
    )
    
    path = model.simulate_regime_path(initial_state=0, num_steps=5)
    # Should alternate 0, 1, 0, 1, 0
    assert list(path) == [0, 1, 0, 1, 0]
    
    var_path = model.simulate_variance_path(initial_state=0, num_steps=5)
    assert list(var_path) == [0.04, 0.25, 0.04, 0.25, 0.04]
