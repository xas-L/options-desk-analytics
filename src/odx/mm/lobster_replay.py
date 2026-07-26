"""Replay Avellaneda-Stoikov market maker against LOBSTER order flow."""

from __future__ import annotations

import pandas as pd
from typing import Optional, Dict

from odx.mm.avellaneda_stoikov import optimal_quotes
from odx.mm.queue_tracker import QueueTracker
from odx.logging import get_logger

logger = get_logger(__name__)


class LobsterReplayEngine:
    """Engine to replay LOBSTER ticks and simulate a market maker."""

    def __init__(
        self,
        messages: pd.DataFrame,
        orderbook: pd.DataFrame,
        gamma: float = 0.1,
        sigma: float = 0.2,
        k: float = 1.5,
        order_size: int = 100,
        T: float = 1.0
    ) -> None:
        """Initialize engine.
        
        Args:
            messages: DataFrame of LOBSTER messages.
            orderbook: DataFrame of LOBSTER limit order book states.
            gamma: Risk aversion parameter.
            sigma: Absolute volatility parameter.
            k: Liquidity parameter (kappa).
            order_size: Standard quote size for the MM.
            T: Total terminal time for the episode.
        """
        self.messages = messages
        self.orderbook = orderbook
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.order_size = order_size
        self.T = T
        
        self.inventory = 0
        self.cash = 0.0
        self.pnl_history = []
        
        self.bid_tracker: Optional[QueueTracker] = None
        self.ask_tracker: Optional[QueueTracker] = None

    def run(self) -> pd.DataFrame:
        """Run the simulation across all LOBSTER events.
        
        Returns:
            DataFrame containing P&L history over time.
        """
        logger.info("Starting LOBSTER replay engine...")
        
        for i in range(len(self.messages)):
            msg = self.messages.iloc[i]
            ob = self.orderbook.iloc[i]
            
            t = msg["time"]
            # Convert time from seconds to a fraction of T, assuming roughly 23400 seconds in a day (6.5 hours)
            time_fraction = min(t / 23400.0, self.T)
            
            event_type = msg["event_type"]
            price = msg["price"]
            size = msg["size"]
            direction = msg["direction"]  # 1 (buy) or -1 (sell)
            
            best_bid = ob["bid_px_1"]
            best_ask = ob["ask_px_1"]
            mid = (best_bid + best_ask) / 2.0
            
            # 1. Update queue trackers based on market events
            # Event Types: 4, 5 (Execution) -> deplete volume ahead
            # Event Types: 2, 3 (Cancel/Delete) -> probabilistically deplete volume
            if event_type in (4, 5):
                # Execution
                if self.bid_tracker and price == self.bid_tracker.price:
                    fills = self.bid_tracker.process_trade(price, size)
                    if fills > 0:
                        self.inventory += fills
                        self.cash -= fills * price
                if self.ask_tracker and price == self.ask_tracker.price:
                    fills = self.ask_tracker.process_trade(price, size)
                    if fills > 0:
                        self.inventory -= fills
                        self.cash += fills * price
                        
            elif event_type in (2, 3):
                # Cancellation
                if self.bid_tracker and price == self.bid_tracker.price:
                    self.bid_tracker.process_cancellation(price, size, prob_ahead=0.5)
                if self.ask_tracker and price == self.ask_tracker.price:
                    self.ask_tracker.process_cancellation(price, size, prob_ahead=0.5)

            # 2. Recalculate optimal quotes
            opt_bid, opt_ask = optimal_quotes(
                s=mid,
                q=self.inventory,
                gamma=self.gamma,
                sigma=self.sigma,
                T=self.T,
                k=self.k,
                t=time_fraction
            )
            
            # Snap to tick size (e.g. 0.01)
            opt_bid = round(opt_bid * 100) / 100.0
            opt_ask = round(opt_ask * 100) / 100.0
            
            # Prevent crossing the spread completely, but allow joining best bid/ask
            opt_bid = min(opt_bid, best_ask - 0.01)
            opt_ask = max(opt_ask, best_bid + 0.01)
            
            # 3. Place or update orders if price changed or filled
            if not self.bid_tracker or self.bid_tracker.price != opt_bid or not self.bid_tracker.active:
                # Find volume ahead (approximated by LOB depth at this price)
                vol_ahead = 0
                for lvl in range(1, 11):
                    col_px = f"bid_px_{lvl}"
                    col_sz = f"bid_sz_{lvl}"
                    if col_px in ob and ob[col_px] == opt_bid:
                        vol_ahead = ob[col_sz]
                        break
                self.bid_tracker = QueueTracker(price=opt_bid, order_size=self.order_size, volume_ahead=vol_ahead)
                
            if not self.ask_tracker or self.ask_tracker.price != opt_ask or not self.ask_tracker.active:
                vol_ahead = 0
                for lvl in range(1, 11):
                    col_px = f"ask_px_{lvl}"
                    col_sz = f"ask_sz_{lvl}"
                    if col_px in ob and ob[col_px] == opt_ask:
                        vol_ahead = ob[col_sz]
                        break
                self.ask_tracker = QueueTracker(price=opt_ask, order_size=self.order_size, volume_ahead=vol_ahead)
                
            # 4. Record P&L
            mtm_pnl = self.cash + self.inventory * mid
            self.pnl_history.append({
                "time": t,
                "mid": mid,
                "inventory": self.inventory,
                "opt_bid": opt_bid,
                "opt_ask": opt_ask,
                "pnl": mtm_pnl
            })
            
        logger.info("Replay complete. Final P&L: %.2f", self.pnl_history[-1]["pnl"])
        return pd.DataFrame(self.pnl_history)
