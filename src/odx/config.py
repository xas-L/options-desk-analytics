"""App config with environment variable loading.

Uses pydantic-settings so that every field can be overridden via an
"ODX_"-prefixed environment variable (e.g. "ODX_DEFAULT_RISK_FREE_RATE=0.04").

Example
-------
>>> from odx.config import Config
>>> cfg = Config() # uses defaults / env vars
>>> cfg.default_risk_free_rate
0.05
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Global configuration for ODX.

    Attributes:
    
    default_risk_free_rate : float
        Annualised continuously-compounded risk-free rate used when none is
        supplied by the caller.  Default 0.05 (5 %).
    default_dividend_yield : float
        Annualised continuous dividend yield.  Default 0.0.
    data_dir : Path
        Root directory for market data snapshots and processed files.
    polygon_api_key : str | None
        API key for the Polygon.io data provider (optional).
    tradier_api_key : str | None
        API key for the Tradier data provider (optional).
    log_level : str
        Logging level.  One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """

    model_config = SettingsConfigDict(
        env_prefix="ODX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_risk_free_rate: float = Field(
        default=0.05,
        description="Annualised risk-free rate (continuous compounding).",
    )
    default_dividend_yield: float = Field(
        default=0.0,
        description="Annualised continuous dividend yield.",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for market data.",
    )
    polygon_api_key: Optional[str] = Field(
        default=None,
        description="Polygon.io API key.",
    )
    tradier_api_key: Optional[str] = Field(
        default=None,
        description="Tradier API key.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
