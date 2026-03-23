"""SVI (Stochastic Volatility Inspired) raw parametrisation and slice fitter.

Raw SVI: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

where k = log(K/F) is log-moneyness and w = sigma_implied^2 * T is total
implied variance.

References

Gatheral (2004) "A parsimonious arbitrage-free implied volatility
    parametrization with application to the valuation of volatility
    derivatives."
Zeliade Systems (2009) "Quasi-explicit calibration of Gatheral's SVI model."
Roper (2010) "Arbitrage free implied volatility surfaces."

Butterfly no-arbitrage

A vol surface is free of butterfly arbitrage if and only if the risk-neutral
density is non-negative everywhere, which for a single expiry slice translates
to the Durrleman (2005) condition:

    g(k) = (1 - k*w'(k) / (2*w(k)))^2
             - w'(k)^2 / 4 * (1/w(k) + 1/4)
             + w''(k) / 2  >= 0  for all k

where prime denotes d/dk.  For raw SVI the first and second derivatives of w
are available analytically:

    w'(k)  = b * (rho + (k-m) / sqrt((k-m)^2 + sigma^2))
    w''(k) = b * sigma^2 / ((k-m)^2 + sigma^2)^(3/2)

The function ``durrleman_g`` evaluates g(k) analytically on a supplied grid;
``check_butterfly_arb`` returns the minimum value and a flag.

Weighting
-
The original unweighted least-squares fit gives equal influence to every
strike.  Illiquid wings often have wide bid-ask spreads and noisy mid prices,
so they should carry less weight.  Two practical schemes are provided:

  'vega'   — weight proportional to BS vega evaluated at the current
              total variance.  ATM options naturally dominate.
  'spread' — weight proportional to 1 / bid_ask_spread.  Tighter spread
              → higher weight.  Requires bid and ask columns.

A raw weight array can also be supplied directly.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize


# -
# Core SVI formula
# -

def svi_total_variance(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Raw SVI total implied variance w = sigma_imp^2 * T.

    Parameters
    
    k : ndarray
        Log-moneyness k = log(K/F).
    a, b, rho, m, sigma : float
        Raw SVI parameters.  Constraints for a valid slice:
        b >= 0, |rho| < 1, sigma > 0, a + b*sigma*sqrt(1-rho^2) >= 0.

    Returns
    -
    ndarray
        Total variance w(k).
    """
    xi = k - m
    return a + b * (rho * xi + np.sqrt(xi ** 2 + sigma ** 2))


# -
# Analytic derivatives for the Durrleman condition
# -

def _svi_derivatives(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, w', w'') for the raw SVI slice — all evaluated analytically."""
    xi = k - m
    sqrt_term = np.sqrt(xi ** 2 + sigma ** 2)

    w   = a + b * (rho * xi + sqrt_term)
    wp  = b * (rho + xi / sqrt_term)
    wpp = b * sigma ** 2 / sqrt_term ** 3

    return w, wp, wpp


# -
# Durrleman butterfly no-arbitrage condition
# -

def durrleman_g(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate the Durrleman g(k) function on a grid of log-moneyness values.

    Butterfly no-arbitrage requires g(k) >= 0 for all k.

    Parameters
    
    k : ndarray
        Log-moneyness grid on which to evaluate g.
    a, b, rho, m, sigma : float
        Raw SVI parameters.

    Returns
    -
    g : ndarray
        Durrleman condition value at each point in k.  Negative values
        indicate butterfly arbitrage at that moneyness.
    """
    w, wp, wpp = _svi_derivatives(k, a, b, rho, m, sigma)

    # Guard against degenerate w to avoid division by zero
    w_safe = np.maximum(w, 1e-12)

    term1 = (1.0 - k * wp / (2.0 * w_safe)) ** 2
    term2 = wp ** 2 / 4.0 * (1.0 / w_safe + 0.25)
    term3 = wpp / 2.0

    return term1 - term2 + term3


def check_butterfly_arb(
    params: np.ndarray,
    k_min: float = -1.5,
    k_max: float = 1.5,
    n_points: int = 500,
) -> Tuple[float, bool]:
    """Check whether the raw SVI slice satisfies butterfly no-arbitrage.

    Evaluates the Durrleman condition on a fine grid of log-moneyness values
    and returns the minimum g value.  A minimum below zero signals arbitrage.

    Parameters
    
    params : ndarray of length 5
        Raw SVI parameters [a, b, rho, m, sigma].
    k_min, k_max : float
        Log-moneyness range to check (default ±1.5, spanning roughly
        exp(±1.5) ≈ 0.22 to 4.5 in K/F space).
    n_points : int
        Number of grid points (default 500).

    Returns
    -
    g_min : float
        Minimum value of g(k) over the grid.  Negative → butterfly arb.
    is_arbitrage_free : bool
        True if g_min >= 0 everywhere on the grid.
    """
    a, b, rho, m, sigma = params
    k_grid = np.linspace(k_min, k_max, n_points)
    g = durrleman_g(k_grid, a, b, rho, m, sigma)
    g_min = float(g.min())
    return g_min, g_min >= 0.0


# -
# Weighting helpers
# -

def _vega_weights(total_variance: np.ndarray) -> np.ndarray:
    """Weights proportional to approximate ATM vega.

    Under flat vol, vega ~ S * sqrt(T) * phi(d1).  Near the money d1 ≈ 0.5*sqrt(w)
    so phi(d1) ≈ phi(0) is roughly constant; the dominant varying term is
    sqrt(w).  We therefore weight by 1/sqrt(w), which upweights low-variance
    (near-the-money) points relative to the wings.

    Parameters
    
    total_variance : ndarray
        Observed total implied variances w = IV^2 * T.

    Returns
    -
    ndarray
        Non-negative weight vector, normalised to sum to len(total_variance).
    """
    w_safe = np.maximum(total_variance, 1e-8)
    raw = 1.0 / np.sqrt(w_safe)
    return raw / raw.mean()


def _spread_weights(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Weights proportional to 1 / bid-ask spread.

    Wide spread → illiquid → low weight.

    Parameters
    
    bid, ask : ndarray
        Bid and ask prices in the same units as the chain.

    Returns
    -
    ndarray
        Non-negative weight vector, normalised to sum to len(bid).
    """
    spread = np.maximum(ask - bid, 1e-6)   # avoid division by zero
    raw = 1.0 / spread
    return raw / raw.mean()


# -
# Fitter
# -

def fit_svi(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    weights: Optional[Union[np.ndarray, str]] = None,
    bid: Optional[np.ndarray] = None,
    ask: Optional[np.ndarray] = None,
    check_arb: bool = True,
    k_arb_min: float = -1.5,
    k_arb_max: float = 1.5,
) -> Tuple[np.ndarray, float, dict]:
    """Fit raw SVI to observed (k, w) pairs.

    Parameters
    
    log_moneyness : ndarray
        Log-moneyness k = log(K/F) for each option.
    total_variance : ndarray
        Total implied variance w = IV^2 * T for each option.
    weights : ndarray or str, optional
        Controls the weighted least-squares fit:

        * ``None`` or ``'uniform'`` — equal weights (original behaviour).
        * ``'vega'``   — weight by approximate ATM vega (1/sqrt(w)).
          Recommended when no bid/ask data is available.
        * ``'spread'`` — weight by 1 / bid-ask spread.  Requires ``bid``
          and ``ask`` to be supplied.
        * ndarray    — explicit weight vector of length len(log_moneyness).

    bid, ask : ndarray, optional
        Required when ``weights='spread'``.
    check_arb : bool
        If True (default), evaluate the Durrleman butterfly condition after
        fitting and include the result in the returned diagnostics dict.
    k_arb_min, k_arb_max : float
        Log-moneyness range for the Durrleman check.

    Returns
    -
    params : ndarray of length 5
        Fitted [a, b, rho, m, sigma].
    rmse : float
        Root-mean-squared error between fitted and observed total variance.
    info : dict
        Diagnostics including optimizer result, weighting scheme used, and
        (if check_arb=True) butterfly arbitrage status.

        Keys:
          'weight_scheme'   : str describing the weighting used.
          'optimizer_result': scipy OptimizeResult.
          'g_min'           : float, minimum Durrleman g (if check_arb).
          'butterfly_arb_free': bool (if check_arb).
          'arb_warning'     : str, human-readable warning if arb detected.
    """
    k = np.asarray(log_moneyness, dtype=float)
    w_obs = np.asarray(total_variance, dtype=float)

    if len(k) != len(w_obs):
        raise ValueError("log_moneyness and total_variance must have the same length.")
    if np.any(w_obs < 0):
        raise ValueError("total_variance contains negative values.")

    # 
    # Build weight vector
    # 
    if weights is None or (isinstance(weights, str) and weights == "uniform"):
        w_vec = np.ones(len(k))
        weight_scheme = "uniform"

    elif isinstance(weights, str) and weights == "vega":
        w_vec = _vega_weights(w_obs)
        weight_scheme = "vega (1/sqrt(w))"

    elif isinstance(weights, str) and weights == "spread":
        if bid is None or ask is None:
            raise ValueError("bid and ask must be supplied when weights='spread'.")
        b_arr = np.asarray(bid, dtype=float)
        a_arr = np.asarray(ask, dtype=float)
        if len(b_arr) != len(k) or len(a_arr) != len(k):
            raise ValueError("bid and ask must have the same length as log_moneyness.")
        w_vec = _spread_weights(b_arr, a_arr)
        weight_scheme = "spread (1/bid_ask_spread)"

    elif isinstance(weights, np.ndarray):
        if len(weights) != len(k):
            raise ValueError("weights array length must match log_moneyness length.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        w_vec = weights.astype(float)
        weight_scheme = "user-supplied"

    else:
        raise ValueError(
            "weights must be None, 'uniform', 'vega', 'spread', or an ndarray."
        )

    # 
    # Weighted least-squares objective with parameter constraints
    # 
    def objective(params: np.ndarray) -> float:
        a, b, rho, m, sig = params
        # Soft penalty for constraint violations (scipy bounds handle hard limits,
        # but the soft interior check keeps the optimiser away from singularities)
        if b < 0 or sig <= 0 or abs(rho) >= 1.0:
            return 1e10
        # Non-negativity of total variance at ATM (k=m, tightest point)
        if a + b * sig * np.sqrt(1.0 - rho ** 2) < 0:
            return 1e10
        w_fit = svi_total_variance(k, a, b, rho, m, sig)
        return float(np.sum(((w_fit - w_obs) * w_vec) ** 2))

    # Starting guess: level at mean observed variance, moderate wings
    x0 = np.array([np.mean(w_obs), 0.10, 0.0, 0.0, 0.10])
    bounds = [
        (-1.0,  1.0),    # a  — total variance can be small but not negative at ATM
        (1e-6,  2.0),    # b
        (-0.999, 0.999), # rho
        (-2.0,  2.0),    # m
        (1e-4,  2.0),    # sigma
    ]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    params = result.x

    # RMSE in total-variance space (unweighted, for interpretability)
    w_fit = svi_total_variance(k, *params)
    rmse = float(np.sqrt(np.mean((w_fit - w_obs) ** 2)))

    # 
    # Butterfly no-arbitrage check (Durrleman condition)
    # 
    info: dict = {
        "weight_scheme": weight_scheme,
        "optimizer_result": result,
    }

    if check_arb:
        g_min, arb_free = check_butterfly_arb(params, k_arb_min, k_arb_max)
        info["g_min"] = g_min
        info["butterfly_arb_free"] = arb_free
        if not arb_free:
            info["arb_warning"] = (
                f"Fitted SVI slice violates the Durrleman butterfly no-arbitrage "
                f"condition (min g = {g_min:.6f} < 0).  Consider tightening bounds, "
                f"supplying better weights, or filtering illiquid strikes before fitting."
            )
        else:
            info["arb_warning"] = ""

    return params, rmse, info


# -
# Convenience: convert fitted params to annualised IV
# -

def svi_iv(
    log_moneyness: np.ndarray,
    params: np.ndarray,
    T: float,
) -> np.ndarray:
    """Convert fitted SVI total variance to annualised implied volatility.

    Parameters
    
    log_moneyness : ndarray
        Log-moneyness k = log(K/F).
    params : ndarray of length 5
        Fitted [a, b, rho, m, sigma] from ``fit_svi``.
    T : float
        Time to expiry in years.

    Returns
    -
    ndarray
        Annualised implied vol at each moneyness point.
        Points where total variance is negative are clipped to zero before
        taking the square root (should not occur for a well-fitted slice).
    """
    w = svi_total_variance(log_moneyness, *params)
    w = np.maximum(w, 0.0)
    return np.sqrt(w / T)