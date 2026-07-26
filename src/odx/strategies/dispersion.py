"""Dispersion trading strategy scaffolding."""

from __future__ import annotations


class DispersionStrategy:
    """Index vs constituents dispersion trading strategy.
    
    Trades the spread between index implied volatility and the 
    weighted sum of constituent implied volatilities.
    """
    
    def __init__(self, index_symbol: str, constituents: list[str], weights: list[float]):
        self.index_symbol = index_symbol
        self.constituents = constituents
        self.weights = weights
        
    def generate_signals(self, market_data: dict) -> dict:
        """Generate trading signals based on implied correlation.
        
        To be implemented fully in Chunk 24.
        """
        raise NotImplementedError("Dispersion strategy logic not yet implemented.")
