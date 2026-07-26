"""FIFO queue position tracker for limit order simulations."""

from __future__ import annotations


class QueueTracker:
    """Tracks position in a FIFO limit order book queue.
    
    Models the number of shares ahead of our order at a specific price level.
    """

    def __init__(self, price: float, order_size: int, volume_ahead: int) -> None:
        """Initialize tracker.
        
        Args:
            price: Price level of the order.
            order_size: Size of our limit order.
            volume_ahead: Initial volume (shares) ahead of us in the queue.
        """
        self.price = price
        self.order_size = order_size
        self.volume_ahead = volume_ahead
        self.volume_filled = 0
        self.active = True

    def process_trade(self, trade_price: float, trade_size: int) -> int:
        """Process a market execution.
        
        If the trade occurs at our price level, it depletes the volume ahead of us.
        If volume ahead reaches 0, we start getting filled.
        
        Returns:
            Number of our shares filled in this event.
        """
        if not self.active or trade_price != self.price:
            return 0
            
        filled_now = 0
        if trade_size <= self.volume_ahead:
            self.volume_ahead -= trade_size
        else:
            # Trade size exceeds volume ahead, we get a fill
            remaining_trade = trade_size - self.volume_ahead
            self.volume_ahead = 0
            
            fill_amount = min(remaining_trade, self.order_size - self.volume_filled)
            self.volume_filled += fill_amount
            filled_now = fill_amount
            
            if self.volume_filled >= self.order_size:
                self.active = False
                
        return filled_now

    def process_cancellation(self, cancel_price: float, cancel_size: int, prob_ahead: float = 1.0) -> None:
        """Process a cancellation at the order's price level.
        
        Args:
            cancel_price: Price of the cancelled order.
            cancel_size: Size of the cancelled order.
            prob_ahead: Probability that the cancellation occurred ahead of us in the queue.
                        In an exact L3 orderbook this would be deterministic, but in L2/LOBSTER 
                        we can approximate that cancellations happen uniformly across the level.
        """
        if not self.active or cancel_price != self.price or self.volume_ahead == 0:
            return
            
        # Reduce volume ahead probabilistically or proportionally
        # For simplicity, if we assume all cancels happen ahead of us:
        effective_cancel = int(cancel_size * prob_ahead)
        self.volume_ahead = max(0, self.volume_ahead - effective_cancel)

    def is_filled(self) -> bool:
        """Check if the order is completely filled."""
        return self.volume_filled >= self.order_size and not self.active
