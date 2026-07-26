"""Greeks-based P&L attribution and scenario projection."""

from __future__ import annotations

from odx.greeks.analytic import bs_greeks


def greeks_projected_pnl(
    delta: float, gamma: float, vega: float, theta: float,
    spot_shift: float, vol_shift: float, dt: float
) -> float:
    """Project P&L using first and second order Greeks.
    
    spot_shift: Absolute change in spot price (dS).
    vol_shift: Absolute change in implied volatility (dVol).
    dt: Change in time (in years).
    """
    pnl = 0.0
    pnl += delta * spot_shift
    pnl += 0.5 * gamma * (spot_shift**2)
    pnl += vega * vol_shift
    pnl += theta * dt
    return float(pnl)


def explain_scenario_pnl(
    actual_pnl: float,
    delta: float, gamma: float, vega: float, theta: float,
    spot_shift: float, vol_shift: float, dt: float
) -> dict:
    """Explain actual scenario P&L via Greeks components."""
    delta_pnl = delta * spot_shift
    gamma_pnl = 0.5 * gamma * (spot_shift**2)
    vega_pnl = vega * vol_shift
    theta_pnl = theta * dt
    
    explained_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    unexplained_pnl = actual_pnl - explained_pnl
    
    return {
        "Delta": float(delta_pnl),
        "Gamma": float(gamma_pnl),
        "Vega": float(vega_pnl),
        "Theta": float(theta_pnl),
        "Explained": float(explained_pnl),
        "Unexplained": float(unexplained_pnl)
    }


def project_option_pnl(
    S: float, K: float, T: float, r: float, sigma: float,
    spot_shift: float, vol_shift: float, dt: float,
    option_type: str = "call", q: float = 0.0,
) -> float:
    """Calculate Greeks from Black-Scholes and project P&L for a single option."""
    greeks = bs_greeks(S, K, T, r, sigma, option_type, q)
    return greeks_projected_pnl(
        delta=greeks["delta"],
        gamma=greeks["gamma"],
        vega=greeks["vega"],
        theta=greeks["theta"],
        spot_shift=spot_shift,
        vol_shift=vol_shift,
        dt=dt
    )
