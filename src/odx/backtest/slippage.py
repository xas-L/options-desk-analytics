"""Market impact and slippage models."""

from __future__ import annotations
import numpy as np


def linear_slippage(qty: float, daily_volume: float, volatility: float, coeff: float = 0.1) -> float:
    """Linear market impact model.
    
    Returns slippage as a fraction of price.
    """
    if daily_volume <= 0:
        return 0.0
    return coeff * volatility * (abs(qty) / daily_volume)


def square_root_slippage(qty: float, daily_volume: float, volatility: float, coeff: float = 0.1) -> float:
    """Square-root market impact model.
    
    Returns slippage as a fraction of price.
    """
    if daily_volume <= 0:
        return 0.0
    return coeff * volatility * np.sqrt(abs(qty) / daily_volume)
