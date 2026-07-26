"""LOBSTER message and orderbook data loader."""

from __future__ import annotations

import pandas as pd
from typing import Tuple


def load_lobster_data(
    message_path: str, orderbook_path: str, levels: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and parse LOBSTER message and orderbook CSV files.
    
    Args:
        message_path: Path to the LOBSTER messages CSV file.
        orderbook_path: Path to the LOBSTER orderbook CSV file.
        levels: Number of order book levels in the data.
        
    Returns:
        A tuple of (messages_df, orderbook_df).
    """
    # LOBSTER Message Format:
    # 1. Time (seconds after midnight)
    # 2. Event Type (1: Submission, 2: Cancellation, 3: Deletion, 4: Execution Visible, 5: Execution Hidden, 6: Cross, 7: Halt)
    # 3. Order ID
    # 4. Size
    # 5. Price (in cents or ten-thousandths, typically requires normalization but we keep it raw here)
    # 6. Direction (1: Buy, -1: Sell)
    msg_cols = ["time", "event_type", "order_id", "size", "price", "direction"]
    
    # Optional 7th column for MPID in some datasets, but we only strictly need the first 6
    messages = pd.read_csv(message_path, header=None, usecols=range(6), names=msg_cols)
    
    # Orderbook Format:
    # Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, Ask Price 2, Ask Size 2...
    ob_cols = []
    for i in range(1, levels + 1):
        ob_cols.extend([f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"])
        
    orderbook = pd.read_csv(orderbook_path, header=None, usecols=range(len(ob_cols)), names=ob_cols)
    
    # Normalize price (LOBSTER typically multiplies prices by 10,000)
    # We will divide by 10000.0 to convert to standard decimal formatting.
    messages["price"] = messages["price"] / 10000.0
    
    for i in range(1, levels + 1):
        orderbook[f"ask_px_{i}"] = orderbook[f"ask_px_{i}"] / 10000.0
        orderbook[f"bid_px_{i}"] = orderbook[f"bid_px_{i}"] / 10000.0
        
    return messages, orderbook
