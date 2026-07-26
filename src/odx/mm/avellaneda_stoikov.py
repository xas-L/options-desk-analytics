"""Avellaneda-Stoikov market making model."""

from __future__ import annotations
import numpy as np


def reservation_price(s: float, q: int, gamma: float, sigma: float, T: float, t: float = 0.0) -> float:
    """Calculate reservation (indifference) price.
    
    Params:
    s - Current mid price.
    q - Current inventory (signed).
    gamma - Risk aversion.
    sigma - Volatility (absolute).
    T - Terminal time.
    t - Current time.
    """
    return s - q * gamma * (sigma**2) * (T - t)


def optimal_quotes(
    s: float, q: int, gamma: float, sigma: float, T: float, k: float, t: float = 0.0
) -> tuple[float, float]:
    """Calculate optimal bid and ask quotes.
    
    Params:
    k - Order book liquidity parameter (kappa).
    
    Returns (bid_price, ask_price).
    """
    res_price = reservation_price(s, q, gamma, sigma, T, t)
    
    # Optimal spread
    spread = gamma * (sigma**2) * (T - t) + (2.0 / gamma) * np.log(1.0 + gamma / k)
    
    # Asymmetric quotes around the reservation price
    bid_price = res_price - spread / 2.0
    ask_price = res_price + spread / 2.0
    
    return float(bid_price), float(ask_price)
