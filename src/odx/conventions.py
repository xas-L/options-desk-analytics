"""Market conventions: day-count enums and tenor string parsing.

Tenor strings follow the standard market format:
  "1D" = 1 calendar day, "2W" = 2 weeks, "3M" = 3 months, "1Y" = 1 year.

Day-count conventions are also accessible via :func:`year_fraction` in
:mod:`odx.types`; this module exposes them as a first-class enum for use
in configuration and serialisation.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum
from typing import Union

from dateutil.relativedelta import relativedelta  # type: ignore[import]


# ---------------------------------------------------------------------------
# Day-count convention enum
# ---------------------------------------------------------------------------


class DayCountConvention(str, Enum):
    """Supported day-count conventions."""

    ACT_365F = "ACT/365F"
    ACT_360 = "ACT/360"
    THIRTY_360 = "30/360"


# ---------------------------------------------------------------------------
# Tenor parsing
# ---------------------------------------------------------------------------

_TENOR_RE = re.compile(r"^(\d+)\s*([DWMY])$", re.IGNORECASE)


def parse_tenor(tenor: str) -> timedelta:
    """Parse a tenor string into a :class:`~datetime.timedelta`.

    Supported units: D (day), W (week), M (month × 30 days), Y (year × 365 days).

    Parameters
    ----------
    tenor : str
        E.g. ``"1D"``, ``"2W"``, ``"3M"``, ``"1Y"``.

    Returns
    -------
    timedelta

    Raises
    ------
    ValueError
        If *tenor* cannot be parsed.

    Examples
    --------
    >>> parse_tenor("3M")
    datetime.timedelta(days=91)
    >>> parse_tenor("1Y")
    datetime.timedelta(days=365)
    """
    m = _TENOR_RE.match(tenor.strip())
    if m is None:
        raise ValueError(
            f"Cannot parse tenor '{tenor}'. Expected format like '1D', '2W', '3M', '1Y'."
        )
    n = int(m.group(1))
    unit = m.group(2).upper()

    if unit == "D":
        return timedelta(days=n)
    if unit == "W":
        return timedelta(weeks=n)
    if unit == "M":
        # Use 30 days per month as a simple convention; for exact month
        # arithmetic use tenor_to_date() with a reference date.
        return timedelta(days=n * 30)
    if unit == "Y":
        return timedelta(days=n * 365)
    raise ValueError(f"Unknown tenor unit '{unit}'.")  # pragma: no cover


def tenor_to_date(
    tenor: Union[str, timedelta],
    reference: date | None = None,
) -> date:
    """Resolve a tenor string (or timedelta) to a calendar date.

    For month / year tenors this uses ``dateutil.relativedelta`` so that
    "1M" from 31 Jan lands on 28 Feb, not 2 Mar.

    Parameters
    ----------
    tenor : str or timedelta
        E.g. ``"3M"`` or ``timedelta(days=90)``.
    reference : date, optional
        Anchor date; defaults to today.

    Returns
    -------
    date
    """
    if reference is None:
        reference = date.today()

    if isinstance(tenor, timedelta):
        return reference + tenor

    m = _TENOR_RE.match(tenor.strip())
    if m is None:
        raise ValueError(f"Cannot parse tenor '{tenor}'.")
    n = int(m.group(1))
    unit = m.group(2).upper()

    if unit == "D":
        return reference + timedelta(days=n)
    if unit == "W":
        return reference + timedelta(weeks=n)
    if unit == "M":
        return reference + relativedelta(months=n)
    if unit == "Y":
        return reference + relativedelta(years=n)
    raise ValueError(f"Unknown tenor unit '{unit}'.")  # pragma: no cover
