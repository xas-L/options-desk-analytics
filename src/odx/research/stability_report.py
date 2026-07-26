"""Stability report generator for rolling calibrations."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def generate_stability_report(wf_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate stability metrics from walk-forward results.
    
    Calculates parameter jump penalties (day-over-day changes) and aggregates out-of-sample RMSE.
    
    Args:
        wf_results: List of dictionaries from walk_forward_calibration.
        
    Returns:
        DataFrame summarizing stability metrics.
    """
    if not wf_results:
        return pd.DataFrame()
        
    # Extract params and dates
    records = []
    for row in wf_results:
        rec = {"date": row["date"], "oos_rmse": row["oos_rmse"]}
        rec.update(row["params"])
        records.append(rec)
        
    df = pd.DataFrame(records)
    
    # Calculate parameter jumps (absolute percentage change)
    param_cols = [c for c in df.columns if c not in ("date", "oos_rmse")]
    for p in param_cols:
        df[f"{p}_jump"] = df[p].diff().abs() / df[p].shift(1).abs().replace(0, np.nan)
        
    # Calculate jump penalty score (average jump across parameters)
    jump_cols = [f"{p}_jump" for p in param_cols]
    df["jump_penalty"] = df[jump_cols].mean(axis=1)
    
    return df
