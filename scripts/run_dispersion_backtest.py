"""Runner script for dispersion strategy backtest."""

import sys
import os
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.abspath("src"))

from odx.strategies.dispersion import calculate_dispersion_weights, calculate_implied_correlation
from odx.logging import get_logger

logger = get_logger("run_dispersion_backtest")


def main():
    """Execute the dispersion backtest."""
    logger.info("Initializing dispersion backtest...")
    
    # Mock index and constituents data
    index_vega = 100.0
    constituent_weights = {"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3}
    constituent_vegas = {"AAPL": 0.5, "MSFT": 0.4, "GOOG": 0.3}
    
    sizes = calculate_dispersion_weights(
        index_vega=index_vega,
        constituent_vegas=constituent_vegas,
        constituent_weights=constituent_weights,
        weighting_scheme="vega"
    )
    
    logger.info("Dispersion trade sizes (vega neutral):")
    for ticker, size in sizes.items():
        logger.info("  %s: %.2f contracts", ticker, size)
        
    # Mock variance data
    index_var = 0.04
    constituent_vars = [0.05, 0.06, 0.055]
    weights_list = [0.4, 0.3, 0.3]
    
    implied_corr = calculate_implied_correlation(index_var, constituent_vars, weights_list)
    logger.info("Implied correlation: %.2f%%", implied_corr * 100.0)
    
if __name__ == "__main__":
    main()
