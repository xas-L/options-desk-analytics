"""Earnings-aware volatility cone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from odx.vol.cone import build_volatility_cone


def build_earnings_cone(returns: np.ndarray, earnings_dates: list[int], windows: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build volatility cones separated by earnings and non-earnings windows.
    
    earnings_dates: list of indices in returns corresponding to earnings announcements.
    Returns (earnings_cone, non_earnings_cone) DataFrames.
    """
    n = len(returns)
    is_earnings = np.zeros(n, dtype=bool)
    
    for idx in earnings_dates:
        if 0 <= idx < n:
            is_earnings[idx] = True
        if 0 <= idx + 1 < n:
            is_earnings[idx + 1] = True
            
    ret_earnings = returns[is_earnings]
    ret_non_earnings = returns[~is_earnings]
    
    cone_earn = build_volatility_cone(ret_earnings, windows) if len(ret_earnings) > max(windows, default=0) else pd.DataFrame()
    cone_non_earn = build_volatility_cone(ret_non_earnings, windows) if len(ret_non_earnings) > max(windows, default=0) else pd.DataFrame()
    
    return cone_earn, cone_non_earn
