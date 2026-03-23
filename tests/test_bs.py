"""Black-Scholes sanity checks: known value, put-call parity, bounds."""

import numpy as np
import pytest

from pricer.vanilla_bs import bs_price
from pricer.greeks_bs import bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho

_BASE = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


class TestBsPrice:
    def test_known_atm_call(self):
        # Hull 12e example: S=K=100, T=1, r=0.05, sigma=0.2 ~ 10.45
        price = bs_price(**_BASE, option_type="call")
        assert abs(price - 10.4506) < 0.01

    def test_put_call_parity(self):
        call = bs_price(**_BASE, option_type="call")
        put = bs_price(**_BASE, option_type="put")
        parity = _BASE["S"] - _BASE["K"] * np.exp(-_BASE["r"] * _BASE["T"])
        assert abs(call - put - parity) < 1e-8

    def test_call_positive(self):
        assert bs_price(**_BASE, option_type="call") > 0

    def test_put_positive(self):
        assert bs_price(**_BASE, option_type="put") > 0

    def test_deep_itm_call_approaches_intrinsic(self):
        price = bs_price(S=200.0, K=100.0, T=0.001, r=0.05, sigma=0.20, option_type="call")
        assert price > 99.0

    def test_deep_otm_call_near_zero(self):
        price = bs_price(S=50.0, K=200.0, T=0.25, r=0.05, sigma=0.20, option_type="call")
        assert price < 0.01

    def test_call_with_dividend_yield(self):
        price_no_div = bs_price(**_BASE, option_type="call", q=0.0)
        price_with_div = bs_price(**_BASE, option_type="call", q=0.05)
        assert price_with_div < price_no_div

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            bs_price(**_BASE, option_type="future")


class TestBsGreeks:
    def test_call_delta_in_bounds(self):
        d = bs_delta(**_BASE, option_type="call")
        assert 0.0 < d < 1.0

    def test_put_delta_in_bounds(self):
        d = bs_delta(**_BASE, option_type="put")
        assert -1.0 < d < 0.0

    def test_call_put_delta_relationship(self):
        # delta_call - delta_put = exp(-q*T) = 1 when q=0
        dc = bs_delta(**_BASE, option_type="call")
        dp = bs_delta(**_BASE, option_type="put")
        assert abs(dc - dp - 1.0) < 1e-10

    def test_gamma_positive(self):
        assert bs_gamma(**_BASE) > 0.0

    def test_vega_positive(self):
        assert bs_vega(**_BASE) > 0.0

    def test_call_theta_negative(self):
        assert bs_theta(**_BASE, option_type="call") < 0.0

    def test_call_rho_positive(self):
        assert bs_rho(**_BASE, option_type="call") > 0.0

    def test_put_rho_negative(self):
        assert bs_rho(**_BASE, option_type="put") < 0.0

    def test_numerical_delta_call(self):
        bump = 0.01
        p_up = bs_price(S=_BASE["S"] + bump, **{k: v for k, v in _BASE.items() if k != "S"}, option_type="call")
        p_dn = bs_price(S=_BASE["S"] - bump, **{k: v for k, v in _BASE.items() if k != "S"}, option_type="call")
        num_delta = (p_up - p_dn) / (2 * bump)
        assert abs(num_delta - bs_delta(**_BASE, option_type="call")) < 1e-4

    def test_numerical_gamma(self):
        bump = 0.01
        p_up = bs_price(S=_BASE["S"] + bump, **{k: v for k, v in _BASE.items() if k != "S"}, option_type="call")
        p_0 = bs_price(**_BASE, option_type="call")
        p_dn = bs_price(S=_BASE["S"] - bump, **{k: v for k, v in _BASE.items() if k != "S"}, option_type="call")
        num_gamma = (p_up - 2 * p_0 + p_dn) / bump ** 2
        assert abs(num_gamma - bs_gamma(**_BASE)) < 1e-4
