"""Greeks P&L attribution."""

from __future__ import annotations


def attribute_pnl(
    delta: float, 
    gamma: float, 
    vega: float, 
    theta: float,
    dS: float, 
    dVol: float, 
    dt: float
) -> dict[str, float]:
    """Decompose theoretical P&L into Greeks components.
    
    Params:
    dS - Spot move.
    dVol - Volatility move (absolute terms).
    dt - Time step in years.
    
    Returns dict with P&L explained by each Greek.
    """
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * (dS ** 2)
    vega_pnl = vega * dVol
    theta_pnl = theta * dt
    
    total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    
    return {
        "delta_pnl": float(delta_pnl),
        "gamma_pnl": float(gamma_pnl),
        "vega_pnl": float(vega_pnl),
        "theta_pnl": float(theta_pnl),
        "total_explained": float(total)
    }
