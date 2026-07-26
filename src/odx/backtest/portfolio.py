"""Portfolio position and cash tracking."""

from __future__ import annotations
import collections


class Portfolio:
    """Tracks cash, positions, and marks to market."""
    
    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = collections.defaultdict(float)
        self.mark_prices = {}
        
    def on_fill(self, fill_event):
        """Update positions and cash on a trade fill."""
        sym = fill_event.symbol
        qty = fill_event.quantity
        price = fill_event.price
        
        cost = qty * price
        self.positions[sym] += qty
        self.cash -= cost
        
    def mark_to_market(self, market_event):
        """Update current prices from market data."""
        self.mark_prices[market_event.symbol] = getattr(market_event, "price", 0.0)
        
    def total_equity(self) -> float:
        """Calculate total MTM equity."""
        equity = self.cash
        for sym, qty in self.positions.items():
            price = self.mark_prices.get(sym, 0.0)
            equity += qty * price
        return equity
