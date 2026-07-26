"""Tests for odx.types — enums, type parsing, and date utilities."""

from datetime import date

import numpy as np
import pytest

from odx.types import (
    ExerciseStyle,
    OptionType,
    TRADING_DAYS_PER_YEAR,
    business_days_between,
    trading_year_fraction,
    year_fraction,
)



# OptionType enum



class TestOptionType:
    @pytest.mark.parametrize("input_val", ["call", "Call", "CALL", "c", "C"])
    def test_parse_call(self, input_val: str) -> None:
        assert OptionType.parse(input_val) == OptionType.CALL

    @pytest.mark.parametrize("input_val", ["put", "Put", "PUT", "p", "P"])
    def test_parse_put(self, input_val: str) -> None:
        assert OptionType.parse(input_val) == OptionType.PUT

    def test_parse_enum_passthrough(self) -> None:
        assert OptionType.parse(OptionType.CALL) == OptionType.CALL

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            OptionType.parse("future")

    def test_string_value(self) -> None:
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"



# ExerciseStyle enum



class TestExerciseStyle:
    @pytest.mark.parametrize("input_val", ["european", "European", "eu", "EU"])
    def test_parse_european(self, input_val: str) -> None:
        assert ExerciseStyle.parse(input_val) == ExerciseStyle.EUROPEAN

    @pytest.mark.parametrize("input_val", ["american", "American", "am", "AM"])
    def test_parse_american(self, input_val: str) -> None:
        assert ExerciseStyle.parse(input_val) == ExerciseStyle.AMERICAN

    def test_parse_enum_passthrough(self) -> None:
        assert ExerciseStyle.parse(ExerciseStyle.AMERICAN) == ExerciseStyle.AMERICAN

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            ExerciseStyle.parse("bermudan")



# year_fraction



class TestYearFraction:
    def test_act_365f_one_year(self) -> None:
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), "ACT/365F")
        # 2024 is a leap year → 366 days
        assert abs(yf - 366 / 365.0) < 1e-10

    def test_act_360(self) -> None:
        yf = year_fraction(date(2024, 1, 1), date(2024, 7, 1), "ACT/360")
        days = (date(2024, 7, 1) - date(2024, 1, 1)).days  # 182
        assert abs(yf - days / 360.0) < 1e-10

    def test_thirty_360(self) -> None:
        yf = year_fraction(date(2024, 1, 1), date(2024, 7, 1), "30/360")
        # 30/360: 6 months = 180 days / 360 = 0.5
        assert abs(yf - 0.5) < 1e-10

    def test_invalid_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            year_fraction(date(2024, 1, 1), date(2024, 7, 1), "BUS/252")

    def test_negative_year_fraction(self) -> None:
        """end < start should produce a negative year fraction."""
        yf = year_fraction(date(2025, 1, 1), date(2024, 1, 1))
        assert yf < 0



# business_days_between / trading_year_fraction



class TestBusinessDays:
    def test_one_week(self) -> None:
        # Mon 2024-01-08 to Mon 2024-01-15 → 5 business days
        bd = business_days_between(date(2024, 1, 8), date(2024, 1, 15))
        assert bd == 5

    def test_weekend_zero(self) -> None:
        # Sat to Mon → 0 business days within the weekend itself
        bd = business_days_between(date(2024, 1, 6), date(2024, 1, 8))
        assert bd == 0

    def test_trading_year_fraction_nonzero(self) -> None:
        tyf = trading_year_fraction(date(2024, 1, 2), date(2024, 12, 31))
        assert 0.9 < tyf < 1.1  # roughly one trading year
