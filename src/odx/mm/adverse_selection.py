"""Post-trade markout metrics for market making adverse selection."""

from __future__ import annotations

import numpy as np


def calculate_markouts(
    fill_prices: np.ndarray,
    fill_sides: np.ndarray,
    future_prices: np.ndarray
) -> np.ndarray:
    """Calculate signed markouts representing adverse selection.
    
    A markout measures the change in price from the fill to a future time, 
    signed by trade direction (1 for buy, -1 for sell).
    A negative markout implies adverse selection (the market moved against you).
    
    Params:
    fill_prices - Array of executed prices.
    fill_sides - Array of sides (+1 for bought, -1 for sold).
    future_prices - Array of reference prices (e.g. mid) at t + N seconds.
    """
    fill_prices = np.asarray(fill_prices, dtype=float)
    fill_sides = np.asarray(fill_sides, dtype=float)
    future_prices = np.asarray(future_prices, dtype=float)
    
    markouts = fill_sides * (future_prices - fill_prices)
    return markouts
