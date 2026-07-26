"""Scan for ETF vs constituent basket implied volatility mispricings."""

from __future__ import annotations

import numpy as np

from odx.vol.implied_correlation import implied_correlation


class ETFArbScanner:
    """Scanner for dispersion trading and ETF arb opportunities."""
    
    def __init__(self, historical_correlation: float):
        self.hist_rho = historical_correlation
        
    def scan(self, etf_vol: float, constituent_vols: np.ndarray, weights: np.ndarray) -> dict:
        """Scan a single ETF against its basket to find mispricings."""
        etf_var = etf_vol**2
        comp_vars = constituent_vols**2
        
        implied_rho = implied_correlation(etf_var, comp_vars, weights)
        
        # Reconstruct fair ETF vol using historical correlation
        fair_var = np.sum((weights**2) * comp_vars)
        
        n = len(weights)
        cross = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    cross += weights[i] * weights[j] * constituent_vols[i] * constituent_vols[j]
                    
        fair_var += self.hist_rho * cross
        fair_vol = np.sqrt(fair_var)
        
        # If implied > historical, ETF vol is expensive vs basket (short index dispersion)
        # If implied < historical, ETF vol is cheap vs basket (long index dispersion)
        mispricing = etf_vol - fair_vol
        
        return {
            "implied_correlation": implied_rho,
            "fair_etf_vol": fair_vol,
            "market_etf_vol": etf_vol,
            "mispricing_bps": mispricing * 10000,
            "signal": "SELL_ETF_VOL" if mispricing > 0 else "BUY_ETF_VOL"
        }
