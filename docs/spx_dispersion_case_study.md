# SPX Dispersion Case Study

## Overview
Index dispersion trading is a strategy designed to profit from mispricings between the implied volatility of an index (e.g., S&P 500) and the implied volatilities of its constituents (e.g., AAPL, MSFT, GOOG).

The core mathematical identity relies on the variance of a sum:
$Var(Index) = \sum w_i^2 Var_i + \sum_{i \neq j} w_i w_j \rho_{ij} \sqrt{Var_i Var_j}$

If the market price of index options implies a correlation $\rho$ that is significantly higher than historical realized correlation, a dispersion trader will sell index volatility and buy constituent volatility.

## Strategy Implementation

In `odx.strategies.dispersion`, we implemented the weighting schemes required to execute this trade:

1. **Vega Neutral**: This approach weights the constituent options such that the total vega of the long constituent basket exactly offsets the vega of the short index position.
2. **Implied Correlation Back-Out**: We calculate the implied correlation surface by comparing the S&P 500 variance swap rates against the variance swap rates of the single stocks.

### Trade Mechanics
*   **Entry**: Sell 1 unit of SPX straddles (or variance swap). Buy $w_i$ units of constituent straddles (or variance swaps).
*   **Delta Hedging**: The entire portfolio must be delta-hedged daily to isolate the pure volatility and correlation exposure.
*   **Exit**: Hold to maturity, or unwind when the implied correlation drops back to its historical mean.

## Typical P&L Drivers
- **Realised Correlation**: The primary driver. If stocks move violently but in opposite directions, realised correlation is low, the index doesn't move much, but the individual stocks do. The short index vol makes money, and the long single stock vol makes money.
- **Realised Volatility vs Implied**: If overall market volatility collapses, the vega-neutrality should protect the portfolio, but gamma effects can still dominate.
