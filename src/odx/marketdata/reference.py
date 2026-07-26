"""Static reference data loader for tickers, multipliers, tick sizes."""

from __future__ import annotations


class ReferenceData:
    """Static reference data repository."""
    
    def __init__(self):
        # Mock database
        self.data = {
            "SPX": {"type": "index", "multiplier": 100, "tick_size": 0.05, "ccy": "USD"},
            "AAPL": {"type": "equity", "multiplier": 100, "tick_size": 0.01, "ccy": "USD"},
            "TSLA": {"type": "equity", "multiplier": 100, "tick_size": 0.01, "ccy": "USD"}
        }
        
    def get_ticker_info(self, ticker: str) -> dict:
        """Fetch static info for a ticker."""
        if ticker not in self.data:
            raise ValueError(f"Ticker {ticker} not found in reference data.")
        return self.data[ticker]
        
    def get_multiplier(self, ticker: str) -> int:
        return self.get_ticker_info(ticker)["multiplier"]
