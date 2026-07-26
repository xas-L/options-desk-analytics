"""Yahoo Finance data source implementation.

Fetches live options chains, spot prices, and risk-free rates via yfinance
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import pandas as pd

from odx.data.base import MarketDataSource
from odx.logging import get_logger

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

logger = get_logger(__name__)


class YahooFinanceSource(MarketDataSource):
    """Data source wrapping Yahoo Finance (via yfinance).

    Attributes
    ----------
    default_risk_free_rate - Fallback rate used if the live ^IRX feed is unavailable.
    min_bid - Drop chain rows where the bid is below this value (default: 0.05).
    min_volume - Drop chain rows with daily volume below this (default: 0).
    min_open_interest - Drop chain rows with open interest below this (default: 0).
    """

    def __init__(
        self,
        default_risk_free_rate: float = 0.05,
        min_bid: float = 0.05,
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> None:
        if yf is None:  # pragma: no cover
            raise ImportError(
                "The yfinance package is required to use YahooFinanceSource. "
                "Install it with `pip install yfinance` or `pip install odx[data]`."
            )
        self.default_risk_free_rate = default_risk_free_rate
        self.min_bid = min_bid
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest

    def get_spot(self, ticker: str) -> float:
        try:
            yt = yf.Ticker(ticker.upper())
            hist = yt.history(period="2d")
            if hist.empty:
                logger.warning("No price history for %s.", ticker)
                return 0.0
            return float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("Could not fetch spot for %s: %s", ticker, exc)
            return 0.0

    def get_dividend_yield(self, ticker: str) -> float:
        try:
            yt = yf.Ticker(ticker.upper())
            info = yt.info
            div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
            return float(div_yield)
        except Exception as exc:
            logger.warning("Could not fetch dividend yield for %s: %s", ticker, exc)
            return 0.0

    def get_risk_free_rate(self) -> float:
        try:
            irx = yf.Ticker("^IRX")
            hist = irx.history(period="5d")
            if hist.empty:
                logger.warning("^IRX feed empty, using fallback r=%.4f", self.default_risk_free_rate)
                return self.default_risk_free_rate
            return float(hist["Close"].iloc[-1]) / 100.0
        except Exception as exc:
            logger.warning("Could not fetch ^IRX (%s), using fallback r=%.4f", exc, self.default_risk_free_rate)
            return self.default_risk_free_rate

    def get_option_chain(
        self,
        ticker: str,
        expiries: Sequence[str | date] | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper()
        yt = yf.Ticker(ticker)

        spot = self.get_spot(ticker)
        r = self.get_risk_free_rate()
        q = self.get_dividend_yield(ticker)

        try:
            available_expiries = yt.options
        except Exception as exc:
            logger.warning("Could not fetch expiry list for %s: %s", ticker, exc)
            return pd.DataFrame()

        if not available_expiries:
            logger.warning("No expiries available for %s", ticker)
            return pd.DataFrame()

        # Normalise requested expiries to strings
        if expiries is not None:
            requested = {e.isoformat() if isinstance(e, date) else e for e in expiries}
            expiries_to_fetch = [e for e in available_expiries if e in requested]
            if not expiries_to_fetch:
                logger.warning("None of the requested expiries found for %s", ticker)
                return pd.DataFrame()
        else:
            expiries_to_fetch = list(available_expiries)

        all_frames = []
        for expiry in expiries_to_fetch:
            df = self._fetch_single_expiry(yt, ticker, expiry, spot, r, q)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            return pd.DataFrame()

        chain = pd.concat(all_frames, ignore_index=True)
        chain = chain.drop_duplicates(subset=["underlying", "expiry", "cp", "K"])
        chain = chain.sort_values(["underlying", "expiry", "cp", "K"]).reset_index(drop=True)
        return chain

    def _time_to_expiry(self, expiry_str: str) -> float:
        exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        days = (exp - date.today()).days
        return max(days / 365.0, 0.0)

    def _fetch_single_expiry(
        self,
        yt: "yf.Ticker",
        symbol: str,
        expiry: str,
        spot: float,
        r: float,
        q: float,
    ) -> pd.DataFrame:
        try:
            chain = yt.option_chain(expiry)
        except Exception as exc:
            logger.warning("Could not fetch chain for %s %s: %s", symbol, expiry, exc)
            return pd.DataFrame()

        T = self._time_to_expiry(expiry)
        if T <= 0.0:
            return pd.DataFrame()

        rows = []
        for cp_label, df in [("call", chain.calls), ("put", chain.puts)]:
            if df.empty:
                continue

            df = df.copy()
            df["cp"] = cp_label
            df["underlying"] = symbol
            df["expiry"] = expiry
            df["T"] = T
            df["spot"] = spot
            df["r"] = r
            df["q"] = q

            df = df.rename(columns={"strike": "K"})

            if "bid" not in df.columns or "ask" not in df.columns:
                continue

            df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
            df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
            df["mid"] = (df["bid"] + df["ask"]) / 2.0

            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
            df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0).astype(int)

            df = df[df["bid"] >= self.min_bid]
            df = df[df["volume"] >= self.min_volume]
            df = df[df["openInterest"] >= self.min_open_interest]
            df = df[df["mid"] > 0.0]

            rows.append(df[["underlying", "expiry", "cp", "K", "T", "spot", "r", "q", "bid", "ask", "mid", "volume", "openInterest"]])

        if not rows:
            return pd.DataFrame()

        return pd.concat(rows, ignore_index=True)
