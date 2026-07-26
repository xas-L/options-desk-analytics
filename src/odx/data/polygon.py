"""Polygon.io market data adapter."""

import datetime
from typing import Optional, Sequence, Union

import httpx
import pandas as pd

from odx.data.base import MarketDataSource


class PolygonDataSource(MarketDataSource):
    """Adapter for Polygon.io market data."""

    def __init__(self, api_key: str, sandbox: bool = False) -> None:
        self.api_key = api_key
        # Polygon doesn't have a distinct sandbox URL for all endpoints
        self.base_url = "https://api.polygon.io"
        
    def get_spot(self, ticker: str) -> float:
        """Fetch current spot price via Polygon snapshot API."""
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {"apiKey": self.api_key}
        with httpx.Client() as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if "ticker" in data and "lastQuote" in data["ticker"]:
                return float(data["ticker"]["lastQuote"].get("p", 0.0))
            if "ticker" in data and "day" in data["ticker"]:
                return float(data["ticker"]["day"].get("c", 0.0))
            return 0.0

    def get_dividend_yield(self, ticker: str) -> float:
        """Return default 0.0 or fetch if available."""
        return 0.0

    def get_risk_free_rate(self) -> float:
        """Return default 0.05 or fetch if available."""
        return 0.05

    def get_option_chain(
        self,
        ticker: str,
        expiries: Optional[Sequence[Union[str, datetime.date]]] = None,
    ) -> pd.DataFrame:
        """Fetch option chain from Polygon."""
        url = f"{self.base_url}/v3/snapshot/options/{ticker}"
        params = {"apiKey": self.api_key, "limit": 250}
        
        results = []
        with httpx.Client() as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            
            next_url = data.get("next_url")
            while next_url:
                resp = client.get(next_url + f"&apiKey={self.api_key}")
                resp.raise_for_status()
                data = resp.json()
                results.extend(data.get("results", []))
                next_url = data.get("next_url")

        records = []
        spot = self.get_spot(ticker)
        r = self.get_risk_free_rate()
        q = self.get_dividend_yield(ticker)
        
        for item in results:
            details = item.get("details", {})
            sym = details.get("ticker", "")
            if not sym.startswith("O:" + ticker):
                continue
                
            try:
                date_str = sym[len("O:" + ticker):len("O:" + ticker)+6]
                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                exp_date = datetime.date(year, month, day)
                
                cp_char = sym[len("O:" + ticker)+6].lower()
                cp = "call" if cp_char == "c" else "put"
                
                strike = float(sym[len("O:" + ticker)+7:]) / 1000.0
            except Exception:
                continue
                
            day_data = item.get("day", {})
            quote = item.get("last_quote", {})
            
            bid = quote.get("bid", 0.0)
            ask = quote.get("ask", 0.0)
            mid = (bid + ask) / 2.0
            vol = day_data.get("v", 0)
            oi = item.get("open_interest", 0)
            
            T = (exp_date - datetime.date.today()).days / 365.25
            if T <= 0:
                continue

            records.append({
                "underlying": ticker,
                "expiry": exp_date.strftime("%Y-%m-%d"),
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
            
        if expiries is not None:
            exp_strs = [e if isinstance(e, str) else e.strftime("%Y-%m-%d") for e in expiries]
            df = df[df["expiry"].isin(exp_strs)]
            
        return df
