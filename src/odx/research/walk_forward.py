"""Rolling walk-forward calibration harness."""

from __future__ import annotations

import pandas as pd
from typing import Callable, List, Dict, Any

from odx.logging import get_logger

logger = get_logger(__name__)


def walk_forward_calibration(
    data_slices: List[pd.DataFrame],
    calibrate_fn: Callable[[pd.DataFrame], Dict[str, float]],
    price_fn: Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Run rolling walk-forward calibration.
    
    Args:
        data_slices: List of dataframes, ordered by time (e.g. daily slices).
        calibrate_fn: Function mapping a daily dataframe to calibrated parameters.
        price_fn: Function mapping a daily dataframe and parameters to priced dataframe 
                  (must add 'model_price' column).
                  
    Returns:
        List of dictionaries with 'date', 'params', and 'oos_rmse'.
    """
    logger.info("Starting walk-forward calibration over %d periods.", len(data_slices))
    
    results = []
    
    if len(data_slices) < 2:
        logger.warning("Need at least 2 slices for out-of-sample walk-forward.")
        return results
        
    for i in range(len(data_slices) - 1):
        in_sample = data_slices[i]
        out_sample = data_slices[i + 1]
        
        # We assume data slices have a 'date' column or similar
        current_date = in_sample["date"].iloc[0] if "date" in in_sample else f"Period_{i}"
        logger.debug("Calibrating on %s", current_date)
        
        # 1. Calibrate on in-sample
        try:
            params = calibrate_fn(in_sample)
        except Exception as e:
            logger.error("Calibration failed at %s: %s", current_date, e)
            continue
            
        # 2. Price on out-of-sample
        try:
            oos_priced = price_fn(out_sample, params)
            
            # Calculate RMSE
            if "mid" in oos_priced and "model_price" in oos_priced:
                sq_err = (oos_priced["model_price"] - oos_priced["mid"]) ** 2
                rmse = float(sq_err.mean() ** 0.5)
            else:
                rmse = float("nan")
        except Exception as e:
            logger.error("Out-of-sample pricing failed at %s: %s", current_date, e)
            rmse = float("nan")
            
        results.append({
            "date": current_date,
            "params": params,
            "oos_rmse": rmse
        })
        
    logger.info("Completed walk-forward calibration.")
    return results
