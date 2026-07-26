# Options Desk Analytics (ODX)

Options Desk Analytics (ODX) is a modern, quantitative library designed to support options trading desks. It bridges the gap between academic theory and practical, trading infrastructure. 

The system provides robust pricing models, vol surface calibration, risk management tooling, and hedging simulators. It's built in Python with a C++ extension to accelerate some core mathematical operations.

## Key Features

* **Pricing Engines**: Black-Scholes (with C++ acceleration), CRR Binomial, Longstaff-Schwartz LSM for American options, and Heston FFT/Monte Carlo.
* **Volatility Modelling**: Arbitrage-free SVI and SSVI surface calibration, EWMA, GARCH, and rough Bergomi fractional variance generation.
* **Risk & Hedging**: PnL scenario engines, Expected Shortfall, and Avellaneda-Stoikov market making simulators.
* **API & UI**: A fully typed FastAPI server for programmatic access and a Streamlit dashboard for real-time risk visualisation.

## Installation and Quickstart

A virtual environment running Python 3.12 is the way to go.

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/xas-L/options-desk-analytics.git
cd options-desk-analytics
pip install -e .
```

2. (Optional but recommended) Compile the native C++ pricing extension for maximum performance. This requires CMake and a C++ compiler.
```bash
# On Windows, run this via PowerShell:
.\scripts\build_cpp.ps1
```

3. Launch the visual analytics dashboard:
```bash
make run-dashboard
```

## Usage Example

The library is modular and heavily typed. Below is an example of pricing a European call option and retrieving its Greeks.

```python
from odx.pricers.cpp_bindings import bs_price, bs_delta
from odx.strategies.complex_orders import ComplexOrder

# Price a standard European call option
price = bs_price(S=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"Call Price: {price:.4f}")

delta = bs_delta(S=100.0, K=105.0, T=1.0, r=0.05, sigma=0.2, option_type="call")
print(f"Call Delta: {delta:.4f}")

# Construct and evaluate a multi-leg portfolio
order = ComplexOrder()
order.add_leg(option_type="call", strike=100.0, expiry=1.0, ratio=1)
order.add_leg(option_type="call", strike=110.0, expiry=1.0, ratio=-1)

net_greeks = order.net_greeks(spot=100.0, rate=0.05, sigma=0.2, div=0.0)
print("Portfolio Greeks:", net_greeks)
```

## Development Notes

I built this to really understand some parts of what junior quants, systematic traders, and quantitative researchers do on a day-to-day basis. Textbooks often focus entirely on pricing theory, but I wanted to get my hands mucky with the infra that rarely gets covered: data cleaning, event-driven backtesting, risk attribution, and hedging simulators.

Some parts were genuinely difficult. Getting the native C++ extension to build cleanly was a struggle. I had to debug the Windows and MinGW toolchains from scratch rather than relying on a perfectly configured MSVC environment. Another major hurdle was the SVI and SSVI calibration. It's one thing to get a surface close to market data, but forcing the optimiser to remain strictly arb-free under Gatheral's conditions needed a lot of trial and error. It also took some toiling to build out the boring but necessary plumbing (like logging, configuration, and data pipelines) before jumping into the interesting pricing maths.

Despite the friction, it was a really fun project. Debugging all the numerical edge cases (like writing guards for negative variance and building calendar arbitrage penalties) taught me far more about how models behave in action.
