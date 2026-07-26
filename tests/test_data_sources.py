import json
from datetime import date
from unittest import mock

import pandas as pd
import pytest

from odx.data.cboe import CboeDelayedSource
from odx.data.yahoo import YahooFinanceSource

# ---------------------------------------------------------------------------
# Test CboeDelayedSource
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cboe_response() -> dict:
    return {
        "data": {
            "current_price": 150.25,
            "options": [
                {
                    "option": "AAPL270620C00150000",
                    "bid": 5.10,
                    "ask": 5.20,
                    "volume": 100,
                    "open_interest": 500,
                },
                {
                    "option": "AAPL270620P00150000",
                    "bid": 4.90,
                    "ask": 5.00,
                    "volume": 200,
                    "open_interest": 1000,
                },
                # Malformed symbol
                {
                    "option": "AAPL2706",
                    "bid": 1.0,
                    "ask": 1.0,
                }
            ]
        }
    }


def test_cboe_source_parsing(mock_cboe_response: dict) -> None:
    source = CboeDelayedSource(default_risk_free_rate=0.04, default_dividend_yield=0.01)
    
    with mock.patch.object(source, "_fetch_json", return_value=mock_cboe_response):
        spot = source.get_spot("AAPL")
        assert spot == 150.25
        
        df = source.get_option_chain("AAPL")
        assert len(df) == 2  # The malformed one is skipped
        
        # Verify schema
        expected_cols = ["underlying", "expiry", "cp", "K", "T", "spot", "r", "q", "bid", "ask", "mid", "volume", "openInterest"]
        assert list(df.columns) == expected_cols
        
        # Verify call data
        call = df[df["cp"] == "call"].iloc[0]
        assert call["expiry"] == "2027-06-20"
        assert call["K"] == 150.0
        assert call["spot"] == 150.25
        assert call["bid"] == 5.10
        assert call["ask"] == 5.20
        assert call["mid"] == 5.15
        assert call["volume"] == 100
        assert call["openInterest"] == 500
        assert call["r"] == 0.04
        assert call["q"] == 0.01
        
        # Verify put data
        put = df[df["cp"] == "put"].iloc[0]
        assert put["K"] == 150.0
        assert put["bid"] == 4.90


def test_cboe_source_expiry_filtering(mock_cboe_response: dict) -> None:
    source = CboeDelayedSource()
    with mock.patch.object(source, "_fetch_json", return_value=mock_cboe_response):
        # Should find it
        df1 = source.get_option_chain("AAPL", expiries=["2027-06-20"])
        assert len(df1) == 2
        
        # Should filter everything out
        df2 = source.get_option_chain("AAPL", expiries=[date(2027, 7, 18)])
        assert len(df2) == 0

# ---------------------------------------------------------------------------
# Test YahooFinanceSource
# ---------------------------------------------------------------------------

class DummyYahooOptionChain:
    def __init__(self):
        self.calls = pd.DataFrame([
            {"strike": 150.0, "bid": 5.10, "ask": 5.20, "volume": 100, "openInterest": 500}
        ])
        self.puts = pd.DataFrame([
            {"strike": 150.0, "bid": 4.90, "ask": 5.00, "volume": 200, "openInterest": 1000}
        ])

class DummyYahooTicker:
    def __init__(self, ticker):
        self.ticker = ticker
        self.info = {"dividendYield": 0.015}
        self.options = ("2027-06-20", "2027-07-18")
        
    def history(self, period):
        return pd.DataFrame({"Close": [149.0, 150.25]})
        
    def option_chain(self, expiry):
        return DummyYahooOptionChain()


def test_yahoo_source_parsing() -> None:
    with mock.patch("odx.data.yahoo.yf", mock.MagicMock()):
        source = YahooFinanceSource(default_risk_free_rate=0.04)
        
        with mock.patch("odx.data.yahoo.yf.Ticker", side_effect=DummyYahooTicker):
            spot = source.get_spot("AAPL")
            assert spot == 150.25
            
            q = source.get_dividend_yield("AAPL")
            assert q == 0.015
            
            # Test chain parsing
            df = source.get_option_chain("AAPL", expiries=["2027-06-20"])
            
            # Should be exactly the same schema
            expected_cols = ["underlying", "expiry", "cp", "K", "T", "spot", "r", "q", "bid", "ask", "mid", "volume", "openInterest"]
            assert list(df.columns) == expected_cols
            
            assert len(df) == 2
            
            call = df[df["cp"] == "call"].iloc[0]
            assert call["expiry"] == "2027-06-20"
        assert call["K"] == 150.0
        assert call["spot"] == 150.25
        assert call["bid"] == 5.10
        assert call["ask"] == 5.20
        assert call["mid"] == 5.15
        assert call["volume"] == 100
        assert call["openInterest"] == 500
        assert call["q"] == 0.015
