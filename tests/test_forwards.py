"""Tests for forward curve bootstrap, implied dividends, and implied borrow cost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odx.data.borrow import implied_borrow_cost, implied_borrow_curve
from odx.data.dividends import implied_dividend_curve, implied_dividend_yield
from odx.data.forwards import bootstrap_forward, bootstrap_forward_curve


# Helpers

def _synthetic_chain(
    spot: float = 100.0,
    F: float = 105.0,
    r: float = 0.05,
    T: float = 1.0,
    strikes: list[float] | None = None,
    noise_std: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic chain where call/put mids satisfy put-call parity.

    C - P = e^{-rT} * (F - K)  =>  for a given F, we can set:
      C_mid = max(e^{-rT} * (F - K), 0.01) + some base
      P_mid = C_mid - e^{-rT} * (F - K)
    """
    if strikes is None:
        strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]

    rng = np.random.default_rng(42)
    discount = np.exp(-r * T)
    rows = []

    for K in strikes:
        intrinsic_call = max(discount * (F - K), 0.01)
        # Give calls some time value
        C_mid = intrinsic_call + 5.0
        P_mid = C_mid - discount * (F - K)

        # Add noise if requested
        if noise_std > 0:
            C_mid += rng.normal(0, noise_std)
            P_mid += rng.normal(0, noise_std)

        C_mid = max(C_mid, 0.01)
        P_mid = max(P_mid, 0.01)

        base = {
            "underlying": "TEST",
            "expiry": "2027-06-20",
            "T": T,
            "spot": spot,
            "r": r,
            "q": 0.0,
            "K": K,
            "volume": 100,
            "openInterest": 500,
        }
        rows.append({**base, "cp": "call", "bid": C_mid - 0.05, "ask": C_mid + 0.05, "mid": C_mid})
        rows.append({**base, "cp": "put", "bid": P_mid - 0.05, "ask": P_mid + 0.05, "mid": P_mid})

    return pd.DataFrame(rows)


# Forward bootstrap

class TestBootstrapForward:
    def test_recovers_known_forward(self) -> None:
        """No noise — should recover F exactly."""
        F_true = 105.0
        df = _synthetic_chain(F=F_true, noise_std=0.0)
        F_est = bootstrap_forward(df, r=0.05, T=1.0)
        assert abs(F_est - F_true) < 1e-10

    def test_recovers_forward_with_noise(self) -> None:
        """Small noise — should recover F within tolerance."""
        F_true = 105.0
        df = _synthetic_chain(F=F_true, noise_std=0.05)
        F_est = bootstrap_forward(df, r=0.05, T=1.0)
        assert abs(F_est - F_true) < 0.5

    def test_reads_r_T_from_dataframe(self) -> None:
        """When r and T not passed, reads from DataFrame."""
        F_true = 110.0
        df = _synthetic_chain(F=F_true, r=0.03, T=0.5, noise_std=0.0)
        F_est = bootstrap_forward(df)
        assert abs(F_est - F_true) < 1e-10

    def test_empty_returns_nan(self) -> None:
        assert np.isnan(bootstrap_forward(pd.DataFrame()))

    def test_zero_T_returns_nan(self) -> None:
        df = _synthetic_chain()
        df["T"] = 0.0
        assert np.isnan(bootstrap_forward(df, T=0.0))

    def test_insufficient_pairs_returns_nan(self) -> None:
        """Only one strike — fewer than 2 matched pairs."""
        df = _synthetic_chain(strikes=[100.0])
        assert np.isnan(bootstrap_forward(df))


class TestBootstrapForwardCurve:
    def test_multi_expiry(self) -> None:
        df1 = _synthetic_chain(F=105.0, T=0.5)
        df1["expiry"] = "2027-01-20"
        df1["T"] = 0.5
        df2 = _synthetic_chain(F=110.0, T=1.0)
        df2["expiry"] = "2027-06-20"
        df2["T"] = 1.0
        df = pd.concat([df1, df2], ignore_index=True)

        curve = bootstrap_forward_curve(df)
        assert len(curve) == 2
        assert list(curve.columns) == ["expiry", "T", "F"]
        assert abs(curve.iloc[0]["F"] - 105.0) < 1e-10
        assert abs(curve.iloc[1]["F"] - 110.0) < 1e-10


# Implied dividend yield

class TestImpliedDividendYield:
    def test_round_trip(self) -> None:
        """If F = S * e^{(r-q)T}, we should recover q."""
        S, r, q_true, T = 100.0, 0.05, 0.02, 1.0
        F = S * np.exp((r - q_true) * T)
        q_est = implied_dividend_yield(S, F, T, r)
        assert abs(q_est - q_true) < 1e-12

    def test_zero_spot_returns_nan(self) -> None:
        assert np.isnan(implied_dividend_yield(0.0, 105.0, 1.0, 0.05))

    def test_zero_forward_returns_nan(self) -> None:
        assert np.isnan(implied_dividend_yield(100.0, 0.0, 1.0, 0.05))

    def test_zero_T_returns_nan(self) -> None:
        assert np.isnan(implied_dividend_yield(100.0, 105.0, 0.0, 0.05))


class TestImpliedDividendCurve:
    def test_vectorised(self) -> None:
        S, r, q_true = 100.0, 0.05, 0.02
        fwd_curve = pd.DataFrame({
            "expiry": ["2027-03-20", "2027-06-20"],
            "T": [0.25, 0.5],
            "F": [S * np.exp((r - q_true) * 0.25), S * np.exp((r - q_true) * 0.5)],
            "r": [r, r],
        })
        result = implied_dividend_curve(S, fwd_curve)
        assert "q_implied" in result.columns
        np.testing.assert_allclose(result["q_implied"].values, [q_true, q_true], atol=1e-12)


# Implied borrow cost

class TestImpliedBorrowCost:
    def test_zero_borrow_normal_conditions(self) -> None:
        """When F = S * e^{(r-q)T}, borrow cost should be zero."""
        S, r, q, T = 100.0, 0.05, 0.02, 1.0
        F = S * np.exp((r - q) * T)
        b = implied_borrow_cost(S, F, T, r, q)
        assert abs(b) < 1e-12

    def test_positive_borrow_hard_to_borrow(self) -> None:
        """Forward elevated beyond (r-q) carry => positive borrow cost."""
        S, r, q, T = 100.0, 0.05, 0.02, 1.0
        borrow_spread = 0.03
        F = S * np.exp((r - q + borrow_spread) * T)
        b = implied_borrow_cost(S, F, T, r, q)
        assert abs(b - borrow_spread) < 1e-12

    def test_zero_spot_returns_nan(self) -> None:
        assert np.isnan(implied_borrow_cost(0.0, 105.0, 1.0, 0.05, 0.0))

    def test_zero_T_returns_nan(self) -> None:
        assert np.isnan(implied_borrow_cost(100.0, 105.0, 0.0, 0.05, 0.0))


class TestImpliedBorrowCurve:
    def test_vectorised(self) -> None:
        S, r, q = 100.0, 0.05, 0.02
        borrow = 0.01
        fwd_curve = pd.DataFrame({
            "expiry": ["2027-03-20", "2027-06-20"],
            "T": [0.25, 0.5],
            "F": [
                S * np.exp((r - q + borrow) * 0.25),
                S * np.exp((r - q + borrow) * 0.5),
            ],
            "r": [r, r],
        })
        result = implied_borrow_curve(S, fwd_curve, q=q)
        assert "borrow_cost" in result.columns
        np.testing.assert_allclose(result["borrow_cost"].values, [borrow, borrow], atol=1e-12)
