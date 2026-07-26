"""Backtest performance metrics."""

from __future__ import annotations
import numpy as np


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, ann_factor: float = 252.0) -> float:
    """Annualised Sharpe ratio."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess_ret = returns - risk_free_rate / ann_factor
    return float(np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(ann_factor))


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0, ann_factor: float = 252.0) -> float:
    """Annualised Sortino ratio."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    excess_ret = returns - risk_free_rate / ann_factor
    downside = excess_ret[excess_ret < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(excess_ret) / np.std(downside) * np.sqrt(ann_factor))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown of an equity curve."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    # Avoid div by zero
    safe_max = np.where(running_max == 0, 1.0, running_max)
    drawdowns = (running_max - eq) / safe_max
    return float(np.max(drawdowns))


def hit_rate(pnl_per_trade: np.ndarray) -> float:
    """Percentage of winning trades."""
    pnl = np.asarray(pnl_per_trade, dtype=float)
    if len(pnl) == 0:
        return 0.0
    return float(np.mean(pnl > 0))


def turnover(trade_volumes: np.ndarray, avg_equity: float) -> float:
    """Portfolio turnover (gross traded value / average equity)."""
    if avg_equity <= 0:
        return 0.0
    total_traded = np.sum(np.abs(trade_volumes))
    return float(total_traded / avg_equity)
