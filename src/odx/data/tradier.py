"""Tradier market data adapter."""

import datetime
from typing import Optional, Sequence, Union

import httpx
import pandas as pd

from odx.data.base import MarketDataSource


class TradierDataSource(MarketDataSource):
    """Adapter for Tradier market data."""

    def __init__(self, api_key: str, sandbox: bool = True) -> None:
        self.api_key = api_key
        self.base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def get_spot(self, ticker: str) -> float:
        """Fetch current spot price."""
        url = f"{self.base_url}/markets/quotes"
        params = {"symbols": ticker}
        with httpx.Client() as client:
            resp = client.get(url, params=params, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            quotes = data.get("quotes", {}).get("quote", [])
            if isinstance(quotes, dict):
                quotes = [quotes]
            if quotes:
                return float(quotes[0].get("last", 0.0))
            return 0.0

    def get_dividend_yield(self, ticker: str) -> float:
        return 0.0

    def get_risk_free_rate(self) -> float:
        return 0.05
        
    def _get_expiries(self, ticker: str) -> list[str]:
        """Fetch available expiries."""
        url = f"{self.base_url}/markets/options/expirations"
        params = {"symbol": ticker}
        with httpx.Client() as client:
            resp = client.get(url, params=params, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            exp = data.get("expirations", {}).get("date", [])
            if isinstance(exp, str):
                return [exp]
            return exp

    def get_option_chain(
        self,
        ticker: str,
        expiries: Optional[Sequence[Union[str, datetime.date]]] = None,
    ) -> pd.DataFrame:
        """Fetch option chain from Tradier."""
        if expiries is None:
            available_expiries = self._get_expiries(ticker)
        else:
            available_expiries = [e if isinstance(e, str) else e.strftime("%Y-%m-%d") for e in expiries]
            
        spot = self.get_spot(ticker)
        r = self.get_risk_free_rate()
        q = self.get_dividend_yield(ticker)
        
        records = []
        with httpx.Client() as client:
            for exp in available_expiries:
                url = f"{self.base_url}/markets/options/chains"
                params = {"symbol": ticker, "expiration": exp}
                resp = client.get(url, params=params, headers=self.headers)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                options = data.get("options", {}).get("option", [])
                if isinstance(options, dict):
                    options = [options]
                    
                exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                T = (exp_date - datetime.date.today()).days / 365.25
                if T <= 0:
                    continue
                    
                for opt in options:
                    cp = "call" if opt.get("option_type") == "call" else "put"
                    strike = float(opt.get("strike", 0.0))
                    bid = float(opt.get("bid", 0.0))
                    ask = float(opt.get("ask", 0.0))
                    mid = (bid + ask) / 2.0
                    vol = int(opt.get("volume", 0))
                    oi = int(opt.get("open_interest", 0))
                    
                    records.append({
                        "underlying": ticker,
                        "expiry": exp,
                        "cp": cp,
                        "K": strike,
                        "T": T,
                        "spot": spot,
                        "r": r,
                        "q": q,
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "volume": vol,
                        "openInterest": oi
                    })

        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(columns=[
                "underlying", "expiry", "cp", "K", "T", "spot", "r", "q",
                "bid", "ask", "mid", "volume", "openInterest"
            ])
            
        return df
