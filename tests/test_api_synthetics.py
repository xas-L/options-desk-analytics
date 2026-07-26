import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from odx.api.server import app
from odx.strategies.synthetics import scan_synthetic_mispricings

client = TestClient(app)


def test_api_price_option():
    response = client.post(
        "/price",
        json={
            "spot": 100.0,
            "strike": 100.0,
            "expiry": 1.0,
            "r": 0.05,
            "sigma": 0.2,
            "option_type": "call",
            "q": 0.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "price" in data
    assert data["price"] > 10.0  # ATM call should be ~10.45


def test_api_ssvi_surface():
    response = client.post(
        "/volatility/ssvi",
        json={
            "k": [0.0, -0.1, 0.1],
            "t": [1.0, 1.0, 1.0],
            "A": 0.04,
            "B": 1.0,
            "rho": -0.5,
            "eta": 2.0,
            "gamma": 0.5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_variance" in data
    assert "implied_volatility" in data
    assert len(data["total_variance"]) == 3
    assert len(data["implied_volatility"]) == 3


def test_api_portfolio_greeks():
    response = client.post(
        "/greeks/portfolio",
        json={
            "spot": 100.0,
            "r": 0.0,
            "sigma": 0.2,
            "q": 0.0,
            "legs": [
                {"option_type": "call", "strike": 100.0, "expiry": 1.0, "ratio": 1},
                {"option_type": "put", "strike": 100.0, "expiry": 1.0, "ratio": -1}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "net_greeks" in data
    greeks = data["net_greeks"]
    
    # Buy Call, Sell Put ATM at 0 rates/divs should have delta approx 1.0
    np.testing.assert_allclose(greeks["delta"], 1.0, atol=1e-3)
    # Gamma and Vega should offset
    np.testing.assert_allclose(greeks["gamma"], 0.0, atol=1e-3)
    np.testing.assert_allclose(greeks["vega"], 0.0, atol=1e-3)


def test_scan_synthetic_mispricings():
    # Construct a market where the forward is F = 100 * exp(0.05) = 105.127
    calls = pd.DataFrame({
        "strike": [100.0, 105.0],
        "bid": [11.0, 8.0],
        "ask": [11.2, 8.2]
    })
    
    # Make the 100 strike significantly mispriced
    # C - P = (F - K)*e^{-rT} -> P = C - (F - K)*e^{-rT}
    # F = 105.127, K = 100, F-K = 5.127. Discount = 0.9512
    # P_fair approx 11.1 - 5.127*0.9512 = 11.1 - 4.87 = 6.23
    
    # Let's say puts are trading way too cheap, P_ask = 1.0, P_bid = 0.8
    puts = pd.DataFrame({
        "strike": [100.0, 105.0],
        "bid": [0.8, 6.0],
        "ask": [1.0, 6.2]
    })
    
    mispricings = scan_synthetic_mispricings(calls, puts, spot=100.0, r=0.05, dt=1.0, threshold=1.0)
    
    # Target fwd = 105.127
    # For K=100: syn_long_cost = C_ask - P_bid = 11.2 - 0.8 = 10.4
    # implied fwd long = 100 + 10.4 / 0.9512 = 100 + 10.93 = 110.93
    # buying synthetic is NOT cheaper than actual fwd (110.93 > 105.127) -> no "Buy Synthetic"
    
    # syn_short_revenue = C_bid - P_ask = 11.0 - 1.0 = 10.0
    # implied fwd short = 100 + 10.0 / 0.9512 = 100 + 10.51 = 110.51
    # Selling synthetic yields 110.51, actual forward is 105.127. 110.51 > 105.127 -> "Sell Synthetic" edge
    
    assert not mispricings.empty
    
    # Filter for K=100
    m = mispricings[mispricings["strike"] == 100.0]
    assert not m.empty
    assert "Sell Synthetic" in m["type"].values
