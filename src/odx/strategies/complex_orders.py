"""Builder for multi-leg orders with net Greeks aggregation."""

from __future__ import annotations
from dataclasses import dataclass

from odx.greeks.analytic import bs_greeks


@dataclass
class Leg:
    option_type: str
    strike: float
    expiry: float
    ratio: int  # positive for long, negative for short
    
    def calc_greeks(self, S: float, r: float, sigma: float, q: float = 0.0) -> dict:
        g = bs_greeks(S, self.strike, self.expiry, r, sigma, self.option_type, q)
        return {k: v * self.ratio for k, v in g.items()}


class ComplexOrder:
    """Represents a multi-leg options order."""
    
    def __init__(self):
        self.legs: list[Leg] = []
        
    def add_leg(self, option_type: str, strike: float, expiry: float, ratio: int):
        self.legs.append(Leg(option_type, strike, expiry, ratio))
        
    def net_greeks(self, S: float, r: float, sigma: float, q: float = 0.0) -> dict:
        """Aggregate net Greeks for the order."""
        net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for leg in self.legs:
            g = leg.calc_greeks(S, r, sigma, q)
            for k in net:
                net[k] += g[k]
        return net
