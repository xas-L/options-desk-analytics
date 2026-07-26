# Known Limitations

This project is a powerful educational and prototyping framework, but it is not a production-ready institutional trading system. Below is an honest discussion of the current limitations:

## 1. Interest Rates & Dividends
- **Flat Curves**: The models currently assume a flat risk-free rate ($r$) and a continuous dividend yield ($q$). Real production systems require bootstrapping a full overnight index swap (OIS) curve and modeling discrete, lumpy dividend payments (especially for single stocks).

## 2. Borrow Costs (Repo Rate)
- **Ignored Hard-to-Borrow Costs**: For equities, the forward price is driven by $r$, $q$, and the borrow cost. We do not model the term structure of borrow rates, which is critical when trading options on heavily shorted stocks.

## 3. American Options
- **European Focus**: The Fast Fourier Transform (FFT) and closed-form analytic solutions only price European options. While the `mc/` directory can theoretically be extended to use Longstaff-Schwartz for American options, it is currently not implemented. Early exercise premium approximation (e.g., Bjerksund-Stensland) is missing.

## 4. Market Making Queue Dynamics
- **L2 Orderbook Approximation**: The `QueueTracker` assumes our limit orders are placed at the back of the queue and that cancellations happen uniformly. Without Level 3 (order-by-order MBO) data, true deterministic queue position tracking is impossible. Our probabilistic approach is standard for research but has variance in reality.

## 5. Volatility Surface Arbitrage
- **SSVI Calibration Limitations**: While Gatheral's SSVI guarantees no static arbitrage in continuous time and space, fitting a discrete set of market quotes perfectly is often impossible. We minimize the error, but the resulting fitted surface may not exactly match the bid-ask spread of every liquid instrument in the market.
