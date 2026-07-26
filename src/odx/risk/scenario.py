"""Scenario and stress testing engine."""

from __future__ import annotations
from typing import Callable

import pandas as pd


class ScenarioEngine:
    """Applies defined market shocks to a portfolio.
    
    A scenario is defined as a dict of shocks (e.g., {'spot_shift': -0.10, 'vol_shift': 0.05}).
    """
    
    def __init__(self):
        self.scenarios = {}
        
    def add_scenario(self, name: str, shocks: dict):
        """Register a new scenario by name."""
        self.scenarios[name] = shocks
        
    def run_stress_test(self, portfolio_pricer: Callable[[dict], float]) -> pd.DataFrame:
        """Run all registered scenarios.
        
        portfolio_pricer is a function that takes a 'shocks' dictionary
        and returns the total portfolio NPV.
        """
        results = {}
        
        # Base case (no shocks)
        base_npv = portfolio_pricer({})
        results["Base"] = {"NPV": base_npv, "PnL": 0.0}
        
        for name, shocks in self.scenarios.items():
            scen_npv = portfolio_pricer(shocks)
            results[name] = {"NPV": scen_npv, "PnL": scen_npv - base_npv}
            
        return pd.DataFrame.from_dict(results, orient="index")
