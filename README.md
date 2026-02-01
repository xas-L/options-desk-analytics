
# Options MM Desk Analytics Toolkit (Barrier Monte Carlo + IV/Greeks + Desk Scripts)

A Python options analytics project with two layers:

1) **Barrier option pricing** (European knock-in/knock-out barriers) via **Monte Carlo** under risk-neutral **GBM**, including confidence intervals and monitoring controls.

2) **Listed options desk tooling** for **vanilla European calls and puts**:
   **Black–Scholes pricing**, **Greeks**, **implied volatility**, simple **vol smile** building (optional **SVI** slice fit), plus scripts that turn an **options chain snapshot + positions** into **IV/Greeks, aggregated risk, and scenario PnL**.

If you are looking for “pricing”, this repo prices:
- Barrier options via Monte Carlo (in `pricer/pricer.py`)
- Vanilla European options via Black–Scholes (in `pricer/vanilla_bs.py`)

If you are looking for “analytics”, this repo computes analytics for:
- Vanilla European options (IV, Greeks, smile, risk reports, hedged PnL)

Barrier options are priced, but the IV/Greeks and risk scripts are aimed at **vanilla listed options**, which is closer to how an options market-making desk works day to day.

---

## Features

### Barrier option Monte Carlo pricer
- **Barrier types**
  - Down-and-Out (Call/Put)
  - Up-and-Out (Call/Put)
  - Down-and-In (Call/Put)
  - Up-and-In (Call/Put)
- **Underlying model**: Geometric Brownian Motion under the risk-neutral measure
- **Monitoring**
  - Discrete monitoring (barrier checked each time step)
  - Continuous monitoring approximation via a continuity correction
- **Variance reduction**: optional antithetic variates
- **Statistical output**: price, confidence interval, Monte Carlo standard error, barrier hit rate, runtime
- **Analysis tools**
  - Convergence analysis vs number of simulations
  - Sensitivity analysis over S0, K, B, T, r, sigma
- **CLI**: interactive prompt for pricing and optional analyses

### Vanilla options analytics (European)
- **Black–Scholes pricing** for call/put with optional dividend yield
- **Greeks**: delta, gamma, vega, theta, rho
- **Implied volatility** solver for call/put (market price to IV)
- **Vol smile (MVP)**
  - Build a per-expiry smile from a chain snapshot
  - Optional SVI slice fitting (stretch)

### Desk-style scripts
- **Chain to IV/Greeks**: compute mid, IV and Greeks per option line from a chain snapshot
- **Portfolio risk report**: join positions to chain, compute IV/Greeks per line, aggregate exposures by underlying and expiry, and run spot and vol shock scenario PnL
- **Position reconciliation (optional)**: basic key coverage checks between positions and chain

### Notebooks
- Barrier notebooks (original): GBM simulation, barrier pricing examples, convergence, sensitivity, continuity correction impact
- Desk notebooks:
  - `06_iv_smile.ipynb`: build and plot a simple IV smile from chain data
  - `07_risk_report.ipynb`: run the risk report workflow and inspect outputs
  - `08_delta_hedged_pnl.ipynb`: simulate a discrete delta hedge and plot hedged PnL

### Tests
- `tests/test_bs.py`: Black–Scholes sanity checks (known value, parity, bounds)
- `tests/test_iv.py`: IV round-trip tests, including vectorised inputs

---

## Project structure

````
pricer/
  pricer.py              barrier Monte Carlo engine and interactive CLI
  helper.py              plotting utilities for convergence and sensitivity
  vanilla_bs.py          Black–Scholes pricing for vanilla European options
  greeks_bs.py           Black–Scholes Greeks
  implied_vol.py         implied volatility solver
  vol_smile.py           per-expiry smile utilities
  svi.py                 SVI slice fit (stretch)

scripts/
  chain_to_iv_greeks.py
  portfolio_risk_report.py
  reconcile_positions.py

data/
  sample_chain.csv
  sample_positions.csv

notebooks/
  06_iv_smile.ipynb
  07_risk_report.ipynb
  08_delta_hedged_pnl.ipynb
  (plus existing barrier notebooks)

tests/
  test_bs.py
  test_iv.py
````

---

## Theoretical background

### Barrier options

Barrier options are path-dependent options whose payoff depends on whether the underlying crosses a barrier level (B) during the option’s life.

* **Knock-out** options expire worthless if the barrier is breached.
* **Knock-in** options only become active if the barrier is breached.

### Geometric Brownian Motion under the risk-neutral measure

The underlying (S_t) is modelled as GBM under the risk-neutral measure:
$$[
dS_t = r S_t dt + \sigma S_t dW_t
]$$
Discretised simulation step:
$$[
S_{t+\Delta t} = S_t \exp\left((r - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t},Z\right),
\quad Z \sim \mathcal{N}(0, 1)
]$$
### Monte Carlo pricing

Option value is the discounted expected payoff under the risk-neutral measure:
$$[
V_0 = e^{-rT}\mathbb{E}_Q[\text{Payoff}]
]$$
Monte Carlo estimates this by simulating many paths, applying barrier conditions, averaging payoffs, and reporting estimation error and a confidence interval.

### Black–Scholes, Greeks, and implied volatility (vanilla only)

For vanilla European options, Black–Scholes provides closed-form prices and Greeks under the standard assumptions.
Implied volatility is the volatility parameter that makes the model price match a market mid, found via root-finding.

### Volatility smile and SVI (optional)

A volatility smile describes how implied volatility varies with strike for a given expiry. SVI is a common parametric form used to fit a smooth smile slice.

---

## Setup and installation

### Requirements

* Python 3.8+
* Dependencies in `requirements.txt` (typically includes `numpy`, `scipy`, `pandas`, `matplotlib`)
* `pytest` for running tests

### Install

```bash
git clone https://github.com/xas-L/barrier-option-pricer.git
cd barrier-option-pricer

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Run tests

```bash
pytest -q
```

---

## Usage

### 1) Barrier Monte Carlo pricer (interactive CLI)

```bash
python pricer/pricer.py
```

The CLI prompts for:

* S0, K, B, T, r, sigma
* option type (for example `down_and_out_call`)
* number of simulations and time steps
* monitoring type (`discrete` or `continuous_approx`)
* optional convergence and sensitivity analyses

### 2) Chain to IV and Greeks (vanilla options)

```bash
python scripts/chain_to_iv_greeks.py --help
python scripts/chain_to_iv_greeks.py --chain data/sample_chain.csv --out out_chain_iv_greeks.csv
```

Use `--help` to see the exact arguments supported in your version.

### 3) Portfolio risk report (vanilla options)

```bash
python scripts/portfolio_risk_report.py --help
python scripts/portfolio_risk_report.py --positions data/sample_positions.csv --chain data/sample_chain.csv --outdir out_risk
```

Typical outputs include:

* per-line positions with IV/Greeks
* aggregated risk by underlying and expiry
* scenario PnL from spot and vol shocks

### 4) Notebooks

```bash
jupyter lab
```

Open:

* `notebooks/06_iv_smile.ipynb`
* `notebooks/07_risk_report.ipynb`
* `notebooks/08_delta_hedged_pnl.ipynb`

---

## Data formats

### `data/sample_chain.csv`

Minimum expected columns for the desk scripts:

* `underlying`
* `expiry`
* `cp` (call/put or c/p)
* `K` (strike)
* `T` (time to expiry in years)
* `mid` (or `bid` and `ask`)

Optional (recommended where available):

* `spot`
* `r` (rate)
* `q` (dividend yield)

### `data/sample_positions.csv`

Required columns:

* `underlying`
* `expiry`
* `cp`
* `K`
* `qty`

Notes:

* The portfolio scripts assume a **contract multiplier of 100** when converting Greeks into exposures.

---

## Limitations

* Barrier pricer assumes GBM, constant rates, vol, and European exercise.
* Continuity correction is an approximation intended to reduce discrete-monitoring bias.
* The listed options layer uses Black–Scholes assumptions + is intended for analytics and prototyping, not prod rn.

---

## Roadmap

* Add American pricer
* Add basic CI to run pytest on pushes
* Extend reconciliation and data quality checks for chain and positions inputs
* Improve surface handling across expiries (beyond per-expiry smile)

