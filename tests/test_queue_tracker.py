"""Tests for FIFO queue tracker."""

from odx.mm.queue_tracker import QueueTracker


def test_queue_tracker_execution():
    """Test volume depletion and fills from market executions."""
    # We place an order of 100 shares at price 10.0, with 500 shares ahead of us
    qt = QueueTracker(price=10.0, order_size=100, volume_ahead=500)
    
    # Trade at different price shouldn't affect us
    filled = qt.process_trade(trade_price=10.1, trade_size=200)
    assert filled == 0
    assert qt.volume_ahead == 500
    
    # Trade of 300 shares at our price reduces volume ahead
    filled = qt.process_trade(trade_price=10.0, trade_size=300)
    assert filled == 0
    assert qt.volume_ahead == 200
    
    # Trade of 250 shares at our price depletes remaining 200, fills us for 50
    filled = qt.process_trade(trade_price=10.0, trade_size=250)
    assert filled == 50
    assert qt.volume_ahead == 0
    assert qt.volume_filled == 50
    assert qt.active is True
    
    # Trade of 100 shares fills the remaining 50
    filled = qt.process_trade(trade_price=10.0, trade_size=100)
    assert filled == 50
    assert qt.volume_filled == 100
    assert qt.active is False
    assert qt.is_filled() is True
    
    # Further trades should do nothing
    filled = qt.process_trade(trade_price=10.0, trade_size=100)
    assert filled == 0


def test_queue_tracker_cancellation():
    """Test volume depletion from cancellations."""
    qt = QueueTracker(price=10.0, order_size=100, volume_ahead=500)
    
    # Cancel 100 shares ahead of us
    qt.process_cancellation(cancel_price=10.0, cancel_size=100, prob_ahead=1.0)
    assert qt.volume_ahead == 400
    
    # Cancel at different price
    qt.process_cancellation(cancel_price=10.1, cancel_size=100, prob_ahead=1.0)
    assert qt.volume_ahead == 400
    
    # Probabilistic cancel
    qt.process_cancellation(cancel_price=10.0, cancel_size=200, prob_ahead=0.5)
    assert qt.volume_ahead == 300
    
    # Cancel more than what's ahead
    qt.process_cancellation(cancel_price=10.0, cancel_size=400, prob_ahead=1.0)
    assert qt.volume_ahead == 0
