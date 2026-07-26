"""Event-driven backtest engine."""

from __future__ import annotations
import collections


class BacktestEngine:
    """Core event-driven loop for backtesting."""
    
    def __init__(self, data_handler, execution_handler, portfolio, strategy):
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.strategy = strategy
        self.events = collections.deque()
        self.is_running = False

    def run(self):
        """Main event loop execution."""
        self.is_running = True
        
        while self.is_running:
            if not self.events:
                event = self.data_handler.get_next_event()
                if event is None:
                    self.is_running = False
                    break
                self.events.append(event)
            
            event = self.events.popleft()
            
            if getattr(event, "type", None) == "MARKET":
                self.strategy.on_market_data(event)
                self.portfolio.mark_to_market(event)
            elif getattr(event, "type", None) == "SIGNAL":
                orders = self.strategy.generate_orders(event)
                for order in orders:
                    self.events.append(order)
            elif getattr(event, "type", None) == "ORDER":
                fills = self.execution_handler.execute(event)
                for fill in fills:
                    self.events.append(fill)
            elif getattr(event, "type", None) == "FILL":
                self.portfolio.on_fill(event)
