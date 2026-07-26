"""Runner script for walk-forward stability study."""

import sys
import os
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.abspath("src"))

from odx.research.walk_forward import walk_forward_calibration
from odx.research.stability_report import generate_stability_report
from odx.logging import get_logger

logger = get_logger("run_stability_study")


def dummy_calibrate(df: pd.DataFrame) -> Dict[str, float]:
    """Mock calibration returning stationary parameters."""
    return {"sigma": 0.2 + (df["mid"].mean() / 1000.0), "kappa": 1.5}


def dummy_price(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    """Mock pricer using calibrated params."""
    res = df.copy()
    res["model_price"] = res["mid"] * (1.0 + (params["sigma"] - 0.2))
    return res


def main():
    """Execute the stability study."""
    logger.info("Initializing stability study...")
    
    # Generate mock daily slices
    slices = []
    for day in range(10):
        # mock chain of 5 strikes
        df = pd.DataFrame({
            "date": [f"Day_{day}"] * 5,
            "K": [90, 95, 100, 105, 110],
            "mid": [12.0, 8.0, 5.0, 3.0, 1.5]
        })
        # Add noise
        df["mid"] += day * 0.1
        slices.append(df)
        
    results = walk_forward_calibration(slices, dummy_calibrate, dummy_price)
    
    report = generate_stability_report(results)
    
    print("\n=== Walk-Forward Stability Report ===")
    print(report.to_string())
    
if __name__ == "__main__":
    main()
