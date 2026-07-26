"""Post-fetch chain cleaning pipeline.

Operates on a standard-schema DataFrame (see base.py) and produces a cleaned
DataFrame plus a flagged DataFrame containing removed rows with reasons.

This is a *separate, composable step* from the per-source liquidity filtering
already built into YahooFinanceSource (min_bid, min_volume, etc.). The source
filters are about ingestion-time noise reduction; this pipeline is about
downstream analytical quality control.
"""

from __future__ import annotations

import pandas as pd

from odx.logging import get_logger

logger = get_logger(__name__)


def clean_chain(
    df: pd.DataFrame,
    *,
    drop_zero_oi: bool = True,
    drop_zero_volume: bool = False,
    flag_crossed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean an options chain DataFrame, returning (clean, flagged).

    Steps applied in order:
      1. Drop expired rows (T <= 0).
      2. Drop crossed markets (bid > ask) if flag_crossed is True.
      3. Drop rows with zero open interest AND zero volume (no liquidity).
      4. Drop rows with mid <= 0 (no meaningful price).
      5. Optionally drop rows with zero OI only (drop_zero_oi).
      6. Optionally drop rows with zero volume only (drop_zero_volume).

    Each removed row is placed in the flagged DataFrame with a
    ``flag_reason`` column for auditability.

    Returns:
        (cleaned, flagged) tuple of DataFrames.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()

    clean = df.copy()
    flagged_parts: list[pd.DataFrame] = []

    def _flag(mask: pd.Series, reason: str) -> None:
        nonlocal clean
        bad = clean[mask].copy()
        if not bad.empty:
            bad["flag_reason"] = reason
            flagged_parts.append(bad)
            logger.info("Cleaning: dropped %d rows — %s", len(bad), reason)
        clean = clean[~mask]

    # 1. Expired
    _flag(clean["T"] <= 0, "expired (T <= 0)")

    # 2. Crossed markets
    if flag_crossed:
        _flag(clean["bid"] > clean["ask"], "crossed market (bid > ask)")

    # 3. No liquidity signal at all
    _flag((clean["openInterest"] == 0) & (clean["volume"] == 0), "zero OI and zero volume")

    # 4. Non-positive mid
    _flag(clean["mid"] <= 0, "non-positive mid price")

    # 5. Optional: zero OI only
    if drop_zero_oi:
        _flag(clean["openInterest"] == 0, "zero open interest")

    # 6. Optional: zero volume only
    if drop_zero_volume:
        _flag(clean["volume"] == 0, "zero volume")

    flagged = pd.concat(flagged_parts, ignore_index=True) if flagged_parts else pd.DataFrame()

    return clean.reset_index(drop=True), flagged
