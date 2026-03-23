"""IV round-trip tests: solve back the vol used to generate a BS price."""

import numpy as np
import pytest

from pricer.vanilla_bs import bs_price
from pricer.implied_vol import implied_vol, implied_vol_vectorised

_TOL = 1e-6


class TestImpliedVol:
    @pytest.mark.parametrize("cp", ["call", "put"])
    @pytest.mark.parametrize("true_iv", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_round_trip(self, cp, true_iv):
        price = bs_price(100.0, 100.0, 1.0, 0.05, true_iv, cp)
        solved = implied_vol(price, 100.0, 100.0, 1.0, 0.05, cp)
        assert abs(solved - true_iv) < _TOL

    def test_round_trip_itm_call(self):
        true_iv = 0.25
        price = bs_price(110.0, 100.0, 0.5, 0.03, true_iv, "call")
        solved = implied_vol(price, 110.0, 100.0, 0.5, 0.03, "call")
        assert abs(solved - true_iv) < _TOL

    def test_round_trip_otm_put_with_dividend(self):
        true_iv = 0.30
        price = bs_price(100.0, 110.0, 0.25, 0.05, true_iv, "put", q=0.02)
        solved = implied_vol(price, 100.0, 110.0, 0.25, 0.05, "put", q=0.02)
        assert abs(solved - true_iv) < _TOL

    def test_no_solution_returns_nan(self):
        iv = implied_vol(0.0, 100.0, 100.0, 1.0, 0.05, "call")
        assert np.isnan(iv)

    def test_price_below_intrinsic_returns_nan(self):
        intrinsic = max(100.0 - 90.0, 0)
        iv = implied_vol(intrinsic - 1.0, 100.0, 90.0, 1.0, 0.05, "call")
        assert np.isnan(iv)


class TestImpliedVolVectorised:
    def test_vectorised_calls(self):
        true_ivs = np.array([0.20, 0.25, 0.30])
        Ks = np.array([95.0, 100.0, 105.0])
        prices = np.array([
            bs_price(100.0, K, 1.0, 0.05, iv, "call")
            for K, iv in zip(Ks, true_ivs)
        ])
        solved = implied_vol_vectorised(
            prices, 100.0, Ks, np.ones(3), 0.05, np.array(["call"] * 3)
        )
        np.testing.assert_allclose(solved, true_ivs, atol=_TOL)

    def test_vectorised_mixed_cp(self):
        true_ivs = np.array([0.22, 0.28])
        Ks = np.array([95.0, 105.0])
        Ts = np.array([0.5, 0.5])
        cps = np.array(["call", "put"])
        prices = np.array([
            bs_price(100.0, K, T, 0.05, iv, cp)
            for K, T, iv, cp in zip(Ks, Ts, true_ivs, cps)
        ])
        solved = implied_vol_vectorised(prices, 100.0, Ks, Ts, 0.05, cps)
        np.testing.assert_allclose(solved, true_ivs, atol=_TOL)
