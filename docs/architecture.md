# System Architecture

The Options Desk Analytics (ODX) system is structured to provide a clean separation between mathematical modelling, market data processing, and user-facing infrastructure. This design ensures that junior quantitative developers can easily test and deploy new models without disrupting core production pipelines.

## Module Layout

The codebase relies on a primary `src/odx/` package containing several distinct submodules:

* **`pricers/`**: Houses the core valuation engines. This includes pure Python implementations alongside a native C++ extension (`bs_pricer_cpp`) for Black-Scholes. Models range from simple analytic approximations (SABR) to complex numerical routines (Heston QE Monte Carlo and FFT).
* **`vol/`**: Contains volatility forecasting and surface fitting algorithms. It manages everything from historical variance estimation (GARCH, EWMA, HAR-RV) to implied volatility parameterisations (SVI, SSVI, Rough Bergomi).
* **`greeks/`**: Provides analytic and finite-difference sensitivities for various pricing models. 
* **`risk/`**: Responsible for portfolio-level analytics. This includes Value at Risk (VaR), Expected Shortfall, scenario stress testing, and Greek-based PnL attribution.
* **`hedging/` & `mm/`**: Contains execution logic. It features the Whalley-Wilmott optimal hedging band calculator and Avellaneda-Stoikov market making inventory models.
* **`backtest/`**: An event-driven engine designed to replay historical market data, simulating order fills, slippage, and portfolio state transitions over time.
* **`api/` & `ui/`**: The deployment layer. It includes a FastAPI application for RESTful integration and a Streamlit dashboard for visual analytics.

## Data Flow

The typical lifecycle of a pricing or risk request follows a strict path to ensure data integrity:

1. **Input Normalisation**: Data enters through the FastAPI layer or Python scripts. `Pydantic` schemas immediately validate the inputs (ensuring positive volatilities, correct expiry formats, and valid option types).
2. **Reference Data Matching**: When working with live tickers, the `marketdata.reference` module maps the instrument to its structural constraints, such as tick size and contract multipliers.
3. **Valuation**: Requests are routed to the appropriate module in `pricers/`. The system attempts to use the high-performance C++ bindings first. Where native binaries are missing, it falls back to the pure Python equivalents and logs a warning.
4. **Aggregation**: For portfolio-level requests, objects like `ComplexOrder` iterate over multiple legs, netting the resulting prices and Greeks into a unified summary.
5. **Output**: Results are serialised back to JSON for the API or visualised via Matplotlib within the Streamlit dashboard.
