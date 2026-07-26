import numpy as np
import pandas as pd
import pytest

from odx.vol.garch import fit_garch_11
from odx.risk.var import historical_var, kupiec_pof_test
from odx.risk.scenario import ScenarioEngine
from odx.risk.greeks_attribution import explain_scenario_pnl, project_option_pnl
from odx.vol.cone import build_volatility_cone
from odx.vol.ewma import ewma_volatility


def test_garch_parameter_recovery():
    """Test that GARCH MLE bounds parameters properly on a synthetic series."""
    np.random.seed(42)
    # Use scaled values (e.g., percentage returns) for numerical stability in MLE
    omega, alpha, beta = 0.05, 0.1, 0.8
    n = 2000
    returns = np.zeros(n)
    var = np.zeros(n)
    var[0] = omega / (1 - alpha - beta)
    
    for t in range(1, n):
        var[t] = omega + alpha * returns[t-1]**2 + beta * var[t-1]
        returns[t] = np.sqrt(var[t]) * np.random.normal()
        
    res = fit_garch_11(returns)
    assert res["success"]
    assert res["omega"] > 0
    assert 0.0 < res["alpha"] < 0.3
    assert 0.5 < res["beta"] < 0.95


def test_kupiec_pof():
    """Test Kupiec POF backtesting logic."""
    # 99% VaR, expect 1 hit per 100 observations
    lr, p_val = kupiec_pof_test(10, 1000, 0.99)
    # 10 hits is exactly the expected amount, so LR should be 0 and p_value 1
    np.testing.assert_allclose(lr, 0.0, atol=1e-3)
    np.testing.assert_allclose(p_val, 1.0, atol=1e-3)
    
    # 50 hits is way too many for 99% VaR over 1000 obs (expected 10)
    lr2, p_val2 = kupiec_pof_test(50, 1000, 0.99)
    assert lr2 > 10.0
    assert p_val2 < 0.05 # Reject null hypothesis


def test_historical_var():
    # Synthetic P&L array
    pnl = np.linspace(-100, 100, 101) # -100, -98, ..., 0, ..., 100
    # 99% VaR should pick the bottom 1% element
    # Bottom 1% of 101 elements is the 1st index (approx -98)
    var_99 = historical_var(pnl, 0.99)
    assert var_99 > 90.0 # Just ensuring it's on the deep loss tail


def test_scenario_engine():
    engine = ScenarioEngine()
    engine.add_scenario("Crash", {"spot_shift": -0.2})
    
    # Dummy pricer
    def pricer(shocks):
        spot = 100.0
        spot += spot * shocks.get("spot_shift", 0.0)
        return spot - 100.0 # PnL is spot change
        
    res = engine.run_stress_test(pricer)
    assert "Base" in res.index
    assert "Crash" in res.index
    assert res.loc["Base", "PnL"] == 0.0
    assert res.loc["Crash", "PnL"] == -20.0


def test_greeks_attribution():
    # Synthetic option with Delta = 0.5, Gamma = 0.02
    attr = explain_scenario_pnl(
        actual_pnl=5.5,
        delta=0.5, gamma=0.02, vega=0.0, theta=0.0,
        spot_shift=10.0, vol_shift=0.0, dt=0.0
    )
    # Delta PnL = 0.5 * 10 = 5.0
    # Gamma PnL = 0.5 * 0.02 * 100 = 1.0
    # Explained = 6.0
    # Unexplained = 5.5 - 6.0 = -0.5
    assert attr["Delta"] == 5.0
    assert attr["Gamma"] == 1.0
    assert attr["Explained"] == 6.0
    assert attr["Unexplained"] == -0.5


def test_volatility_cone():
    returns = np.random.normal(0, 0.01, 500)
    cone = build_volatility_cone(returns, windows=[10, 20])
    assert "Q_0.5" in cone.columns
    assert cone.shape == (2, 5) # 2 windows, 5 quantiles


def test_ewma_volatility():
    returns = np.random.normal(0, 0.01, 100)
    vol = ewma_volatility(returns)
    assert len(vol) == 100
    assert np.all(vol > 0)
