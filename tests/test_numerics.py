"""Tests for odx.utils.numerics — safe math and finite-difference Greeks."""

import numpy as np
import pytest

from odx.utils.numerics import (
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_vega,
    safe_log,
    safe_sqrt,
)
from odx.pricers.analytic.bs import bs_price
from odx.greeks.analytic import bs_delta, bs_gamma, bs_vega


# ---------------------------------------------------------------------------
# safe_log
# ---------------------------------------------------------------------------


class TestSafeLog:
    def test_positive_values(self) -> None:
        """Normal inputs should return standard log values."""
        result = safe_log(np.array([1.0, np.e, 100.0]))
        np.testing.assert_allclose(result, np.log([1.0, np.e, 100.0]), atol=1e-15)

    def test_zero_returns_large_negative(self) -> None:
        """log(0) is guarded — should return a very large negative, not -inf."""
        result = safe_log(0.0)
        assert np.isfinite(result)
        assert result < -600

    def test_negative_returns_large_negative(self) -> None:
        """Negative inputs should be clamped like zero."""
        result = safe_log(-5.0)
        assert np.isfinite(result)
        assert result < -600

    def test_scalar_input(self) -> None:
        result = safe_log(1.0)
        assert abs(float(result)) < 1e-15


# ---------------------------------------------------------------------------
# safe_sqrt
# ---------------------------------------------------------------------------


class TestSafeSqrt:
    def test_positive_values(self) -> None:
        result = safe_sqrt(np.array([0.0, 1.0, 4.0, 9.0]))
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0], atol=1e-15)

    def test_negative_returns_zero(self) -> None:
        """Negative variance guard — should return 0.0, not NaN."""
        result = safe_sqrt(-1.0)
        assert float(result) == 0.0

    def test_array_with_negatives(self) -> None:
        result = safe_sqrt(np.array([-2.0, 0.0, 4.0]))
        np.testing.assert_allclose(result, [0.0, 0.0, 2.0], atol=1e-15)


# ---------------------------------------------------------------------------
# Finite-difference Greeks vs analytic BS Greeks
# ---------------------------------------------------------------------------


_BASE = dict(K=100.0, T=1.0, r=0.05, sigma=0.20)


def _bs_price_by_spot(S: float, **kwargs: object) -> float:
    """Wrapper for bs_price that accepts S as the first positional kwarg."""
    return bs_price(S=S, option_type="call", **kwargs)


def _bs_price_by_sigma(sigma: float, **kwargs: object) -> float:
    """Wrapper for bs_price that accepts sigma as a kwarg."""
    return bs_price(sigma=sigma, option_type="call", **kwargs)


class TestFiniteDifferenceDelta:
    def test_matches_analytic_delta(self) -> None:
        S = 100.0
        fd_delta = finite_difference_delta(
            _bs_price_by_spot, S=S, h=0.01, **_BASE
        )
        analytic = bs_delta(S=S, option_type="call", **_BASE)
        assert abs(fd_delta - analytic) < 1e-4

    def test_deep_itm(self) -> None:
        S = 150.0
        fd_delta = finite_difference_delta(
            _bs_price_by_spot, S=S, h=0.01, **_BASE
        )
        analytic = bs_delta(S=S, option_type="call", **_BASE)
        assert abs(fd_delta - analytic) < 1e-4


class TestFiniteDifferenceGamma:
    def test_matches_analytic_gamma(self) -> None:
        S = 100.0
        fd_gamma = finite_difference_gamma(
            _bs_price_by_spot, S=S, h=0.01, **_BASE
        )
        analytic = bs_gamma(S=S, **_BASE)
        assert abs(fd_gamma - analytic) < 1e-3


class TestFiniteDifferenceVega:
    def test_matches_analytic_vega(self) -> None:
        sigma = 0.20
        fd_vega = finite_difference_vega(
            _bs_price_by_sigma, sigma=sigma, h=0.001, S=100.0, K=100.0, T=1.0, r=0.05,
        )
        analytic = bs_vega(S=100.0, K=100.0, T=1.0, r=0.05, sigma=sigma)
        assert abs(fd_vega - analytic) < 1e-2
