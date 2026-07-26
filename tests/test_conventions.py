"""Tests for odx.conventions — day-count enum and tenor parsing."""

from datetime import date, timedelta

import pytest

from odx.conventions import (
    DayCountConvention,
    parse_tenor,
    tenor_to_date,
)


# ---------------------------------------------------------------------------
# DayCountConvention enum
# ---------------------------------------------------------------------------


class TestDayCountConvention:
    def test_values(self) -> None:
        assert DayCountConvention.ACT_365F.value == "ACT/365F"
        assert DayCountConvention.ACT_360.value == "ACT/360"
        assert DayCountConvention.THIRTY_360.value == "30/360"


# ---------------------------------------------------------------------------
# parse_tenor
# ---------------------------------------------------------------------------


class TestParseTenor:
    def test_days(self) -> None:
        assert parse_tenor("1D") == timedelta(days=1)
        assert parse_tenor("30D") == timedelta(days=30)

    def test_weeks(self) -> None:
        assert parse_tenor("2W") == timedelta(weeks=2)

    def test_months(self) -> None:
        # Simplified: 1M = 30 days in timedelta mode
        assert parse_tenor("1M") == timedelta(days=30)
        assert parse_tenor("3M") == timedelta(days=90)

    def test_years(self) -> None:
        assert parse_tenor("1Y") == timedelta(days=365)
        assert parse_tenor("2Y") == timedelta(days=730)

    def test_case_insensitive(self) -> None:
        assert parse_tenor("1d") == timedelta(days=1)
        assert parse_tenor("2w") == timedelta(weeks=2)
        assert parse_tenor("3m") == timedelta(days=90)
        assert parse_tenor("1y") == timedelta(days=365)

    def test_whitespace_stripped(self) -> None:
        assert parse_tenor("  1D  ") == timedelta(days=1)

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_tenor("abc")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_tenor("")


# ---------------------------------------------------------------------------
# tenor_to_date
# ---------------------------------------------------------------------------


class TestTenorToDate:
    def test_days_from_reference(self) -> None:
        ref = date(2024, 6, 15)
        assert tenor_to_date("10D", reference=ref) == date(2024, 6, 25)

    def test_weeks_from_reference(self) -> None:
        ref = date(2024, 6, 15)
        assert tenor_to_date("2W", reference=ref) == date(2024, 6, 29)

    def test_months_end_of_month(self) -> None:
        """dateutil.relativedelta handles end-of-month rollover."""
        ref = date(2024, 1, 31)
        result = tenor_to_date("1M", reference=ref)
        # 1 month from Jan 31 → Feb 29 (2024 is a leap year)
        assert result == date(2024, 2, 29)

    def test_years(self) -> None:
        ref = date(2024, 3, 15)
        assert tenor_to_date("1Y", reference=ref) == date(2025, 3, 15)

    def test_timedelta_passthrough(self) -> None:
        ref = date(2024, 6, 1)
        assert tenor_to_date(timedelta(days=7), reference=ref) == date(2024, 6, 8)

    def test_invalid_tenor_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            tenor_to_date("XYZ", reference=date(2024, 1, 1))
