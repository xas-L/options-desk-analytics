"""CBOE delayed quotes data source.

Fetches data from the public CBOE delayed options API:
``https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json``

Since this is an unsupported endpoint and doesn't provide risk-free rates
or dividend yields natively, it relies on static defaults.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime
from typing import Any, Sequence

import pandas as pd

from odx.data.base import MarketDataSource
from odx.logging import get_logger

logger = get_logger(__name__)


class CboeDelayedSource(MarketDataSource):
    """Data source for CBOE delayed quotes.

    Attributes
    ----------
    default_risk_free_rate : float
        Rate applied to all extracted chain rows (default: 0.05).
    default_dividend_yield : float
        Yield applied to all extracted chain rows (default: 0.0).
    """

    BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_{ticker}.json"

    def __init__(
        self,
        default_risk_free_rate: float = 0.05,
        default_dividend_yield: float = 0.0,
    ) -> None:
        self.default_risk_free_rate = default_risk_free_rate
        self.default_dividend_yield = default_dividend_yield
        # Header to pretend to be a real browser, otherwise CBOE returns 403.
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }

    def _fetch_json(self, ticker: str) -> dict[str, Any]:
        url = self.BASE_URL.format(ticker=ticker.upper())
        req = urllib.request.Request(url, headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to fetch CBOE data for %s: %s", ticker, exc)
            return {}

    def get_spot(self, ticker: str) -> float:
        data = self._fetch_json(ticker)
        data_block = data.get("data", {})
        spot = data_block.get("current_price")
        if spot is None:
            logger.warning("No spot price found in CBOE payload for %s", ticker)
            return 0.0
        return float(spot)

    def get_dividend_yield(self, ticker: str) -> float:
        # CBOE payload doesn't easily expose this, fallback to default.
        return self.default_dividend_yield

    def get_risk_free_rate(self) -> float:
        return self.default_risk_free_rate

    def get_option_chain(
        self,
        ticker: str,
        expiries: Sequence[str | date] | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        data = self._fetch_json(ticker)
        data_block = data.get("data", {})

        spot = float(data_block.get("current_price", 0.0))
        if spot <= 0.0:
            logger.warning("Cannot construct chain for %s without valid spot price.", ticker)
            return pd.DataFrame()

        options_list = data_block.get("options", [])
        if not options_list:
            logger.warning("No options found in CBOE payload for %s.", ticker)
            return pd.DataFrame()

        requested_expiries: set[str] | None = None
        if expiries is not None:
            requested_expiries = {
                e.isoformat() if isinstance(e, date) else e for e in expiries
            }

        rows = []
        r = self.default_risk_free_rate
        q = self.default_dividend_yield

        for opt in options_list:
            # Example option symbol: AAPL250620C00150000
            symbol = opt.get("option", "")
            if len(symbol) < 15:
                continue

            try:
                # The date is characters after the ticker. Since ticker length varies,
                # we search backwards. The structure is TTTT YYMMDD C KKKKKKKK
                # It always ends with 1 letter (C/P) and 8 digits (strike).
                suffix = symbol[-15:]
                date_str = suffix[:6]  # YYMMDD
                cp_char = suffix[6:7]  # C or P
                strike_str = suffix[7:]  # 8 digits

                year = 2000 + int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                exp_date = date(year, month, day)
                expiry_iso = exp_date.isoformat()

                if requested_expiries and expiry_iso not in requested_expiries:
                    continue

                T = max((exp_date - date.today()).days / 365.0, 0.0)
                if T <= 0.0:
                    continue

                cp = "call" if cp_char == "C" else "put"
                K = float(strike_str) / 1000.0

                bid = float(opt.get("bid", 0.0))
                ask = float(opt.get("ask", 0.0))
                volume = int(opt.get("volume", 0))
                open_interest = int(opt.get("open_interest", 0))

                mid = (bid + ask) / 2.0

                rows.append({
                    "underlying": ticker,
                    "expiry": expiry_iso,
                    "cp": cp,
                    "K": K,
                    "T": T,
                    "spot": spot,
                    "r": r,
                    "q": q,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "volume": volume,
                    "openInterest": open_interest,
                })
            except Exception:
                pass  # Skip malformed symbols

        if not rows:
            return pd.DataFrame()

        chain = pd.DataFrame(rows)
        # Ensure column order
        cols = ["underlying", "expiry", "cp", "K", "T", "spot", "r", "q", "bid", "ask", "mid", "volume", "openInterest"]
        chain = chain[cols]
        chain = chain.sort_values(["underlying", "expiry", "cp", "K"]).reset_index(drop=True)
        return chain
