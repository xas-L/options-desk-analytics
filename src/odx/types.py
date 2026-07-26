"""Core type aliases and option enums, plus day-count/date utilities."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd

Moneyness = float
Strike = float
Vol = float
Tenor = Union[str, float]

TRADING_DAYS_PER_YEAR = 252


class OptionType(str, Enum):
    """Call or put."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def parse(cls, value: "OptionType | str") -> "OptionType":
        if isinstance(value, OptionType):
            return value
        normalized = str(value).strip().lower()
        if normalized in ("call", "c"):
            return cls.CALL
        if normalized in ("put", "p"):
            return cls.PUT
        raise ValueError(f"Cannot parse OptionType from: {value!r}")


class ExerciseStyle(str, Enum):
    """European or American exercise."""

    EUROPEAN = "european"
    AMERICAN = "american"

    @classmethod
    def parse(cls, value: "ExerciseStyle | str") -> "ExerciseStyle":
        if isinstance(value, ExerciseStyle):
            return value
        normalized = str(value).strip().lower()
        if normalized in ("european", "eu"):
            return cls.EUROPEAN
        if normalized in ("american", "am"):
            return cls.AMERICAN
        raise ValueError(f"Cannot parse ExerciseStyle from: {value!r}")


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
    raise ValueError(f"Unsupported day count convention: {convention}")


def business_days_between(
    start: date,
    end: date,
    holidays: pd.DatetimeIndex | None = None,
) -> int:
    kwargs = {}
    if holidays is not None:
        kwargs["holidays"] = holidays.values.astype("datetime64[D]")
    return int(np.busday_count(np.datetime64(start, "D"), np.datetime64(end, "D"), **kwargs))


def trading_year_fraction(
    start: date,
    end: date,
    holidays: pd.DatetimeIndex | None = None,
) -> float:
    return business_days_between(start, end, holidays) / TRADING_DAYS_PER_YEAR
