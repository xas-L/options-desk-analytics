"""Simulated execution and fill generation."""

from __future__ import annotations
import time
from types import SimpleNamespace


class ExecutionHandler:
    """Simulates filling orders against bid/ask with latency."""
    
    def __init__(self, latency_ms: float = 0.0, slippage_model=None):
        self.latency_ms = latency_ms
        self.slippage_model = slippage_model
        
    def execute(self, order_event, market_data=None) -> list:
        """Process an order and return fill events."""
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
            
        if market_data is None:
            return []
            
        sym = getattr(order_event, "symbol", "")
        qty = getattr(order_event, "quantity", 0.0)
        
        # Cross the spread
        if qty > 0:
            base_price = market_data.get("ask", 0.0)
        else:
            base_price = market_data.get("bid", 0.0)
            
        # Apply slippage adversely
        slippage = 0.0
        if self.slippage_model is not None:
            vol = market_data.get("volatility", 0.0)
            vol_mkt = market_data.get("daily_volume", 0.0)
            slippage = self.slippage_model(qty, vol_mkt, vol)
            
        exec_price = base_price * (1 + slippage) if qty > 0 else base_price * (1 - slippage)
        
        fill = SimpleNamespace(
            type="FILL",
            symbol=sym,
            quantity=qty,
            price=exec_price
        )
        return [fill]
