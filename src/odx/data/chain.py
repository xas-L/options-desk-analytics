"""ChainSnapshot: a typed wrapper around a standard-schema options chain DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd


# Standard chain schema column names (authoritative list lives in base.py docstring).
CHAIN_COLUMNS = [
    "underlying", "expiry", "cp", "K", "T", "spot",
    "r", "q", "bid", "ask", "mid", "volume", "openInterest",
]


@dataclass
class ChainSnapshot:
    """Immutable snapshot of an options chain at a point in time.

    Wraps a standard-schema DataFrame with typed metadata for convenient
    downstream consumption by the cleaning pipeline, forward bootstrapper,
    and pricers.

    Attributes:
        underlying: Ticker symbol.
        timestamp: When the snapshot was captured.
        spot: Underlying price at snapshot time.
        r: Risk-free rate embedded in the chain.
        q: Dividend yield embedded in the chain.
        chain: DataFrame matching the standard chain schema.
    """

    underlying: str
    timestamp: datetime
    spot: float
    r: float
    q: float
    chain: pd.DataFrame = field(repr=False)

    # -- factories --------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, timestamp: datetime | None = None) -> ChainSnapshot:
        """Build a ChainSnapshot from a standard-schema DataFrame.

        Extracts underlying, spot, r, q from the first row. Caller can
        override the timestamp (defaults to now).
        """
        if df.empty:
            raise ValueError("Cannot build ChainSnapshot from an empty DataFrame.")
        first = df.iloc[0]
        return cls(
            underlying=str(first["underlying"]),
            timestamp=timestamp or datetime.now(),
            spot=float(first["spot"]),
            r=float(first["r"]),
            q=float(first["q"]),
            chain=df.reset_index(drop=True),
        )

    # -- convenience accessors --------------------------------------------

    @property
    def strikes(self) -> np.ndarray:
        """Sorted unique strikes across all expiries."""
        return np.sort(self.chain["K"].unique())

    @property
    def expiries(self) -> list[str]:
        """Sorted unique expiry date strings."""
        return sorted(self.chain["expiry"].unique())

    @property
    def calls(self) -> pd.DataFrame:
        return self.chain[self.chain["cp"] == "call"]

    @property
    def puts(self) -> pd.DataFrame:
        return self.chain[self.chain["cp"] == "put"]

    def expiry_slice(self, expiry: str) -> pd.DataFrame:
        """Return all rows for a single expiry."""
        return self.chain[self.chain["expiry"] == expiry]
