"""Volatility cone construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volatility_cone(returns: np.ndarray, windows: list[int], quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]) -> pd.DataFrame:
    """Build a volatility cone across different historical window lengths.
    
    Returns a DataFrame where index is window lengths and columns are quantiles.
    """
    results = {}
    returns_series = pd.Series(returns)
    
    for w in windows:
        if len(returns) < w:
            continue
            
        # Rolling realized volatility (annualized, 252 days)
        rolling_std = returns_series.rolling(window=w).std() * np.sqrt(252.0)
        rolling_std = rolling_std.dropna()
        
        if len(rolling_std) > 0:
            results[w] = np.quantile(rolling_std, quantiles)
            
    df = pd.DataFrame.from_dict(results, orient="index", columns=[f"Q_{q}" for q in quantiles])
    df.index.name = "Window"
    return df
