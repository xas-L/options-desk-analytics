"""Pydantic request/response schemas for the ODX API."""

from __future__ import annotations

from pydantic import BaseModel


class OptionPricingRequest(BaseModel):
    spot: float
    strike: float
    expiry: float
    r: float
    sigma: float
    option_type: str
    q: float = 0.0


class PricingResponse(BaseModel):
    price: float


class SSVIVolSurfaceRequest(BaseModel):
    k: list[float]
    t: list[float]
    A: float
    B: float
    rho: float
    eta: float
    gamma: float


class SSVIVolSurfaceResponse(BaseModel):
    total_variance: list[float]
    implied_volatility: list[float]


class OrderLeg(BaseModel):
    option_type: str
    strike: float
    expiry: float
    ratio: float


class PortfolioGreeksRequest(BaseModel):
    spot: float
    r: float
    sigma: float
    q: float = 0.0
    legs: list[OrderLeg]


class PortfolioGreeksResponse(BaseModel):
    net_greeks: dict[str, float]
