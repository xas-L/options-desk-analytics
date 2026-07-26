"""Base interface for market data sources.

All sources must return DataFrames matching the **Standard Chain Schema**:
- underlying (str): Underlying ticker symbol
- expiry (str): Expiry date in YYYY-MM-DD format
- cp (str): "call" or "put"
- K (float): Strike price
- T (float): Time to expiry in years
- spot (float): Underlying spot price at time of snapshot
- r (float): Risk-free rate
- q (float): Dividend yield
- bid (float): Bid price
- ask (float): Ask price
- mid (float): Mid price (computed as (bid + ask) / 2.0)
- volume (int): Trading volume
- openInterest (int): Open interest

Specific data sources may not provide natively fields like `r` or `q`, in which
case they should fall back to configurable defaults so the output schema remains
consistent for downstream pricers.
"""

from __future__ import annotations

import abc
from datetime import date
from typing import Sequence

import pandas as pd


class MarketDataSource(abc.ABC):
    """Abstract base class for fetching market data into a standard schema."""

    @abc.abstractmethod
    def get_spot(self, ticker: str) -> float:
        """Fetch the current spot price of the underlying.
        """

    @abc.abstractmethod
    def get_dividend_yield(self, ticker: str) -> float:
        """Fetch the continuous annualised dividend yield.
        """

    @abc.abstractmethod
    def get_risk_free_rate(self) -> float:
        """Fetch the current annualised risk-free rate.
        """

    @abc.abstractmethod
    def get_option_chain(
        self,
        ticker: str,
        expiries: Sequence[str | date] | None = None,
    ) -> pd.DataFrame:
        """Fetch the option chain into the standard ODX schema.
        
        If expiries is None, fetches all available expirations.

        """
