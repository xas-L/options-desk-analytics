"""Corsi's HAR-RV (Heterogeneous Autoregressive) model for realized volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_har_rv(rv_series: np.ndarray | pd.Series) -> LinearRegression:
    """Fit a standard HAR-RV model (Daily, Weekly, Monthly lags).
    
    Returns the fitted scikit-learn LinearRegression model.
    """
    rv = np.asarray(rv_series)
    n = len(rv)
    
    if n < 22:
        raise ValueError("Need at least 22 observations for Monthly lag.")
        
    y = rv[22:]
    x_daily = rv[21:-1]
    
    # Weekly (5-day rolling average of lagged RV)
    x_weekly = np.zeros(n - 22)
    for i in range(n - 22):
        x_weekly[i] = np.mean(rv[i + 17 : i + 22])
        
    # Monthly (22-day rolling average of lagged RV)
    x_monthly = np.zeros(n - 22)
    for i in range(n - 22):
        x_monthly[i] = np.mean(rv[i : i + 22])
        
    X = np.column_stack((x_daily, x_weekly, x_monthly))
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model
