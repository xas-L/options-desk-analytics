"""Tests for market data adapters."""

import datetime
from unittest.mock import Mock, patch

import pandas as pd

from odx.data.polygon import PolygonDataSource
from odx.data.tradier import TradierDataSource


@patch("httpx.Client.get")
def test_polygon_adapter(mock_get):
    """Test Polygon.io data adapter with mocked HTTP transport."""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "details": {"ticker": "O:AAPL290119C00150000"},
                "day": {"v": 100},
                "last_quote": {"bid": 1.2, "ask": 1.4},
                "open_interest": 500
            }
        ],
        "next_url": None
    }
    
    mock_spot_resp = Mock()
    mock_spot_resp.status_code = 200
    mock_spot_resp.json.return_value = {
        "ticker": {"lastQuote": {"p": 150.0}}
    }
    
    mock_get.side_effect = [mock_resp, mock_spot_resp]
    
    adapter = PolygonDataSource(api_key="mock_key")
    df = adapter.get_option_chain("AAPL")
    
    import math
    assert len(df) == 1
    assert df.iloc[0]["underlying"] == "AAPL"
    assert df.iloc[0]["cp"] == "call"
    assert df.iloc[0]["K"] == 150.0
    assert df.iloc[0]["spot"] == 150.0
    assert df.iloc[0]["bid"] == 1.2
    assert df.iloc[0]["ask"] == 1.4
    assert math.isclose(df.iloc[0]["mid"], 1.3)
    assert df.iloc[0]["volume"] == 100
    assert df.iloc[0]["openInterest"] == 500


@patch("httpx.Client.get")
def test_tradier_adapter(mock_get):
    """Test Tradier data adapter with mocked HTTP transport."""
    
    mock_exp_resp = Mock()
    mock_exp_resp.status_code = 200
    mock_exp_resp.json.return_value = {
        "expirations": {"date": ["2029-01-19"]}
    }
    
    mock_spot_resp = Mock()
    mock_spot_resp.status_code = 200
    mock_spot_resp.json.return_value = {
        "quotes": {"quote": [{"last": 150.0}]}
    }
    
    mock_chain_resp = Mock()
    mock_chain_resp.status_code = 200
    mock_chain_resp.json.return_value = {
        "options": {
            "option": [
                {
                    "option_type": "call",
                    "strike": 150.0,
                    "bid": 1.2,
                    "ask": 1.4,
                    "volume": 100,
                    "open_interest": 500
                }
            ]
        }
    }
    
    # get_option_chain calls: _get_expiries, get_spot, then chain loop
    mock_get.side_effect = [mock_exp_resp, mock_spot_resp, mock_chain_resp]
    
    adapter = TradierDataSource(api_key="mock_key", sandbox=True)
    df = adapter.get_option_chain("AAPL")
    
    import math
    assert len(df) == 1
    assert df.iloc[0]["underlying"] == "AAPL"
    assert df.iloc[0]["cp"] == "call"
    assert df.iloc[0]["K"] == 150.0
    assert df.iloc[0]["spot"] == 150.0
    assert df.iloc[0]["bid"] == 1.2
    assert df.iloc[0]["ask"] == 1.4
    assert math.isclose(df.iloc[0]["mid"], 1.3)
    assert df.iloc[0]["volume"] == 100
    assert df.iloc[0]["openInterest"] == 500
