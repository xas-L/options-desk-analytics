from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def year_fraction(
    start: date | datetime,
    end: date | datetime,
    convention: str = "ACT/365F",
) -> float:
    if isinstance(start, datetime):
        start = start.date()
    if isinstance(end, datetime):
        end = end.date()
    days = (end - start).days
    if convention == "ACT/365F":
        return days / 365.0
    if convention == "ACT/360":
        return days / 360.0
    if convention == "30/360":
        d = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (end.day - start.day)
        return d / 360.0
    raise ValueError(f"unsupported day count convention: {convention}")


def business_days_between(
    start: date,
    end: date,
    holidays: pd.DatetimeIndex | None = None,
) -> int:
    hols = holidays.values.astype("datetime64[D]") if holidays is not None else None
    return int(np.busday_count(np.datetime64(start, "D"), np.datetime64(end, "D"), holidays=hols))


def trading_year_fraction(
    start: date,
    end: date,
    holidays: pd.DatetimeIndex | None = None,
) -> float:
    return business_days_between(start, end, holidays) / TRADING_DAYS_PER_YEAR


def nyse_holidays(start_year: int, end_year: int) -> pd.DatetimeIndex:
    try:
        import pandas_market_calendars as mcal  # type: ignore[import]
    except ImportError:
        return pd.DatetimeIndex([])
    cal = mcal.get_calendar("NYSE")
    schedule = cal.schedule(start_date=f"{start_year}-01-01", end_date=f"{end_year}-12-31")
    all_bdays = pd.bdate_range(start=schedule.index.min(), end=schedule.index.max())
    return all_bdays.difference(schedule.index)