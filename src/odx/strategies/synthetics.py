"""Synthetic forward/reversal arbitrage scanner."""

from __future__ import annotations

import pandas as pd
import numpy as np


def scan_synthetic_mispricings(
    calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float, dt: float, threshold: float = 0.05
) -> pd.DataFrame:
    """Scan for mispricings between actual forward and synthetic forward (Call - Put).
    
    calls/puts should have columns: ['strike', 'bid', 'ask']
    """
    merged = pd.merge(calls, puts, on='strike', suffixes=('_c', '_p'))
    
    # Synthetic long: Buy Call (pay ask), Sell Put (receive bid)
    merged['syn_long_cost'] = merged['ask_c'] - merged['bid_p']
    
    # Synthetic short: Sell Call (receive bid), Buy Put (pay ask)
    merged['syn_short_revenue'] = merged['bid_c'] - merged['ask_p']
    
    discount = np.exp(-r * dt)
    
    # Forward parity: (F - K) * e^{-rT} = C - P
    target_fwd = spot / discount
    
    merged['implied_fwd_long'] = merged['strike'] + merged['syn_long_cost'] / discount
    merged['implied_fwd_short'] = merged['strike'] + merged['syn_short_revenue'] / discount
    
    mispricings = []
    
    for _, row in merged.iterrows():
        K = row['strike']
        
        # If buying synthetic forward is cheaper than actual forward
        if target_fwd - row['implied_fwd_long'] > threshold:
            mispricings.append({
                'strike': K,
                'type': 'Buy Synthetic',
                'edge': float(target_fwd - row['implied_fwd_long']),
                'implied_fwd': float(row['implied_fwd_long']),
                'target_fwd': float(target_fwd)
            })
            
        # If selling synthetic forward yields more than actual forward
        if row['implied_fwd_short'] - target_fwd > threshold:
            mispricings.append({
                'strike': K,
                'type': 'Sell Synthetic',
                'edge': float(row['implied_fwd_short'] - target_fwd),
                'implied_fwd': float(row['implied_fwd_short']),
                'target_fwd': float(target_fwd)
            })
            
    return pd.DataFrame(mispricings)
