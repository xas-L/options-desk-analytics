"""Streamlit dashboard for options desk analytics."""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from odx.vol.ssvi import ssvi_total_variance
from odx.strategies.complex_orders import ComplexOrder


st.set_page_config(page_title="ODX Analytics", layout="wide")
st.title("Options Desk Analytics")

st.sidebar.header("Market Inputs")
spot = st.sidebar.number_input("Spot Price", value=100.0)
rate = st.sidebar.number_input("Risk-Free Rate", value=0.05)
div = st.sidebar.number_input("Dividend Yield", value=0.0)
sigma = st.sidebar.number_input("ATM Volatility", value=0.2)

# Tabs
tab1, tab2, tab3 = st.tabs(["Vol Surface", "Portfolio Greeks", "P&L Scenario"])

with tab1:
    st.subheader("SSVI Volatility Surface")
    col1, col2, col3 = st.columns(3)
    with col1:
        rho = st.slider("Correlation (rho)", -0.99, 0.99, -0.5)
    with col2:
        eta = st.slider("Eta", 0.1, 5.0, 2.0)
    with col3:
        gamma = st.slider("Gamma", 0.01, 0.99, 0.5)
        
    k_grid = np.linspace(-0.5, 0.5, 50)
    t_grid = np.array([0.25, 0.5, 1.0, 2.0])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in t_grid:
        w = ssvi_total_variance(k_grid, np.full_like(k_grid, t), 0.04, 1.0, rho, eta, gamma)
        # Avoid negative variance from sliders pushing boundaries wildly
        iv = np.sqrt(np.maximum(w, 0.0) / t)
        ax.plot(k_grid, iv, label=f"T={t}y")
        
    ax.set_xlabel("Log-Moneyness (k = ln(K/F))")
    ax.set_ylabel("Implied Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.subheader("Portfolio Greeks")
    st.write("Current sample portfolio: Long 1 Call @ 100, Short 1 Put @ 100")
    order = ComplexOrder()
    order.add_leg("call", 100.0, 1.0, 1)
    order.add_leg("put", 100.0, 1.0, -1)
    
    greeks = order.net_greeks(spot, rate, sigma, div)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Delta", f"{greeks['delta']:.2f}")
    c2.metric("Net Gamma", f"{greeks['gamma']:.4f}")
    c3.metric("Net Vega", f"{greeks['vega']:.2f}")
    c4.metric("Net Theta", f"{greeks['theta']:.2f}")

with tab3:
    st.subheader("P&L Scenario (Spot Shift)")
    spot_shifts = np.linspace(80, 120, 50)
    pnls = []
    
    # Base portfolio value
    base_val = order.price(spot, rate, sigma, div)
    
    for s in spot_shifts:
        pnls.append(order.price(s, rate, sigma, div) - base_val)
        
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(spot_shifts, pnls, color='green')
    ax2.axvline(x=spot, color='red', linestyle='--', label='Current Spot')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel("Spot Price")
    ax2.set_ylabel("Theoretical P&L")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
