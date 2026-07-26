"""FastAPI server for the ODX Analytics platform."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException

from odx.pricers.analytic.bs import bs_price
from odx.vol.ssvi import ssvi_total_variance
from odx.strategies.complex_orders import ComplexOrder
from odx.api.schemas import (
    OptionPricingRequest,
    PricingResponse,
    SSVIVolSurfaceRequest,
    SSVIVolSurfaceResponse,
    PortfolioGreeksRequest,
    PortfolioGreeksResponse,
)

app = FastAPI(
    title="ODX Analytics API",
    description="Pricing, volatility surfaces, and portfolio risk endpoints.",
    version="1.0.0"
)


@app.post("/price", response_model=PricingResponse)
def price_option(req: OptionPricingRequest):
    """Price a European option using Black-Scholes."""
    try:
        price = bs_price(
            S=req.spot,
            K=req.strike,
            T=req.expiry,
            r=req.r,
            sigma=req.sigma,
            option_type=req.option_type,
            q=req.q
        )
        return {"price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/volatility/ssvi", response_model=SSVIVolSurfaceResponse)
def ssvi_surface(req: SSVIVolSurfaceRequest):
    """Evaluate an SSVI volatility surface for given coordinates."""
    try:
        k_arr = np.array(req.k)
        t_arr = np.array(req.t)
        
        if len(k_arr) != len(t_arr):
            raise ValueError("k and t must have the same length")
            
        w = ssvi_total_variance(k_arr, t_arr, req.A, req.B, req.rho, req.eta, req.gamma)
        # Avoid negative variance dynamically due to floating points
        w_safe = np.maximum(w, 0.0)
        iv = np.sqrt(w_safe / np.maximum(t_arr, 1e-12))
        
        return {
            "total_variance": w.tolist(),
            "implied_volatility": iv.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/greeks/portfolio", response_model=PortfolioGreeksResponse)
def portfolio_greeks(req: PortfolioGreeksRequest):
    """Calculate net aggregated Greeks for a complex multi-leg options portfolio."""
    try:
        order = ComplexOrder()
        for leg in req.legs:
            order.add_leg(leg.option_type, leg.strike, leg.expiry, leg.ratio)
            
        net = order.net_greeks(req.spot, req.r, req.sigma, req.q)
        return {"net_greeks": net}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
