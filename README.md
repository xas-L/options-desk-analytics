# Options Market-Making Desk Analytics Toolkit

A Python options analytics project built in two layers.

**Layer 1: Barrier option pricing** via Monte Carlo under risk-neutral GBM. Prices European knock-in and knock-out barriers (calls and puts) with discrete or continuous-approximation monitoring, optional antithetic variates, and full statistical output including confidence intervals and barrier hit rates.

**Layer 2: Vanilla listed options desk tooling.** Black-Scholes pricing, Greeks, implied volatility, per-expiry vol smile with optional SVI slice fitting, and desk-style scripts that turn a live or historical options chain into IV/Greeks, aggregated portfolio risk, and scenario PnL. A live data fetcher pulls real chains directly from Yahoo Finance so the full pipeline runs on actual market data.

The analytics layer is deliberately modelled on what a vanilla options market-making desk uses day to day: chain in, IV and Greeks out, positions joined to chain, risk aggregated by underlying and expiry, spot and vol scenarios run against the book.

---

## Features

### Barrier option Monte Carlo pricer

- Barrier types: down-and-out, up-and-out, down-and-in, up-and-in (call and put)
- GBM simulation under the risk-neutral measure
- Discrete barrier monitoring and continuous-approximation via Broadie-Glasserman-Kou continuity correction
- Optional antithetic variates for variance reduction
- Output: price, confidence interval, Monte Carlo standard error, barrier hit rate, runtime
- Convergence analysis across simulation counts
- Sensitivity analysis over S0, K, B, T, r, sigma
- Interactive CLI

### Vanilla options analytics

- Black-Scholes pricing for European calls and puts with optional continuous dividend yield
- Full Greeks: delta, gamma, vega, theta, rho
- Implied volatility solver (Brent's method) with vectorised wrapper for chain-level computation
- Per-expiry vol smile builder and plotter
- Raw SVI parametrisation and slice fitter

### American options

- Cox-Ross-Rubinstein binomial tree pricer for American and European exercise
- Backward induction with early exercise at each node
- Delta, gamma and theta from the tree
- Optional early exercise boundary tracking

### Desk scripts

- `chain_to_iv_greeks.py`: compute IV and all five Greeks for every row in a chain snapshot
- `portfolio_risk_report.py`: join a positions book to a chain, compute per-line IV/Greeks, aggregate dollar Greeks by underlying and expiry, run a spot-and-vol scenario PnL grid
- `reconcile_positions.py`: flag position rows with no matching line in the chain

### Live data

- `data/fetch_chain.py`: fetch a live options chain from Yahoo Finance, pull spot, risk-free rate (13-week T-bill via ^IRX) and dividend yield automatically, apply liquidity filters, and write output in the project's standard CSV format so it drops straight into any of the desk scripts

### Notebooks

- `01` to `05`: GBM simulation, barrier pricing examples, convergence analysis, sensitivity analysis, continuity correction impact
- `06_iv_smile.ipynb`: build, plot and SVI-fit a vol smile from chain data
- `07_risk_report.ipynb`: end-to-end risk report workflow with scenario PnL
- `08_delta_hedged_pnl.ipynb`: discrete delta hedge simulation, single path and multi-path terminal PnL distribution

### Tests

- `tests/test_bs.py`: Black-Scholes sanity checks covering known values, put-call parity, Greek bounds and numerical Greeks
- `tests/test_iv.py`: IV round-trip tests across strikes, expiries and option types, including vectorised inputs

---

## Project structure

```
pricer/
  __init__.py                  package exports
  barrier_options_pricer.py    Monte Carlo barrier engine
  helper.py                    convergence and sensitivity plotting utilities
  vanilla_bs.py                Black-Scholes pricer for European calls and puts
  greeks_bs.py                 Black-Scholes Greeks: delta, gamma, vega, theta, rho
  implied_vol.py               implied vol solver (Brent) and vectorised wrapper
  vol_smile.py                 per-expiry smile builder and plotter
  svi.py                       raw SVI parametrisation and slice fitter
  crr_american.py              CRR binomial tree pricer for American and European options

scripts/
  chain_to_iv_greeks.py        CLI: chain snapshot to IV and Greeks CSV
  portfolio_risk_report.py     CLI: positions + chain to risk report and scenario PnL
  reconcile_positions.py       CLI: flag unmatched position rows

data/
  fetch_chain.py               fetch a live chain from Yahoo Finance
  sample_chain.csv             synthetic reference chain (AAPL, MSFT, two expiries each)
  sample_positions.csv         synthetic reference book (iron condor, spreads, straddle)

notebooks/
  01_GBM_Simulation_Demo.ipynb
  02_Barrier_Option_Pricing_Examples.ipynb
  03_Convergence_Analysis.ipynb
  04_Sensitivity_Analysis.ipynb
  05_Continuity_Correction_Impact.ipynb
  06_iv_smile.ipynb
  07_risk_report.ipynb
  08_delta_hedged_pnl.ipynb

tests/
  test_bs.py
  test_iv.py
```

---

## Setup and installation

**Requirements:** Python 3.8+, `pytest` for running tests.

```bash
git clone https://github.com/xas-L/barrier-option-pricer.git
cd barrier-option-pricer

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

**Run tests:**

```bash
pytest -q
```

---

## Usage

### 1. Barrier Monte Carlo pricer

```bash
python pricer/barrier_options_pricer.py
```

The interactive CLI prompts for S0, K, B, T, r, sigma, option type (e.g. `down_and_out_call`), number of simulations and time steps, monitoring type (`discrete` or `continuous_approx`), and optional convergence and sensitivity analyses.

### 2. Fetch a live chain

```bash
# All available expiries for one or more tickers
python data/fetch_chain.py --tickers AAPL MSFT

# Specific expiries
python data/fetch_chain.py --tickers SPY --expiries 2025-06-20 2025-09-19

# With liquidity filters and custom output path
python data/fetch_chain.py --tickers AAPL --min-bid 0.10 --min-volume 10 --min-oi 100 --out data/aapl_live.csv
```

Spot is pulled from 2-day price history. The risk-free rate is fetched from `^IRX` (13-week T-bill) and falls back to `--r-fallback` if the feed is unavailable. Dividend yield comes from ticker metadata.

The three liquidity filters are:

| Flag | Default | Effect |
|---|---|---|
| `--min-bid` | 0.05 | Drop rows where bid is below this level |
| `--min-volume` | 0 | Drop rows with daily volume below this |
| `--min-oi` | 0 | Drop rows with open interest below this |

### 3. Chain to IV and Greeks

```bash
python scripts/chain_to_iv_greeks.py --chain data/sample_chain.csv --out out/chain_iv_greeks.csv

# On live data
python scripts/chain_to_iv_greeks.py --chain data/live_chain.csv --out out/live_iv_greeks.csv
```

Adds `iv`, `delta`, `gamma`, `vega`, `theta`, `rho` columns to every row in the chain. Where IV cannot be solved (e.g. zero-bid row that passed filters), IV is recorded as `nan` and Greeks fall back to a flat vol estimate.

### 4. Portfolio risk report

```bash
python scripts/portfolio_risk_report.py \
  --positions data/sample_positions.csv \
  --chain data/sample_chain.csv \
  --outdir out/risk

# On live data
python scripts/portfolio_risk_report.py \
  --positions data/sample_positions.csv \
  --chain data/live_chain.csv \
  --outdir out/live_risk
```

Outputs written to `--outdir`:

| File | Contents |
|---|---|
| `positions_with_greeks.csv` | Per-line IV, Greeks and dollar Greeks |
| `risk_by_underlying.csv` | Dollar Greeks aggregated by underlying |
| `risk_by_expiry.csv` | Dollar Greeks aggregated by underlying and expiry |
| `scenario_pnl.csv` | First-order PnL across a 7x5 spot and vol shock grid |

Dollar Greeks use a contract multiplier of 100. Dollar delta is `delta * qty * 100 * spot`. Dollar gamma is scaled per 1% move. Dollar vega is per vol point. Dollar theta is daily.

### 5. Position reconciliation

```bash
python scripts/reconcile_positions.py \
  --positions data/sample_positions.csv \
  --chain data/sample_chain.csv
```

Prints any position rows with no matching `(underlying, expiry, cp, K)` key in the chain.

### 6. Notebooks

```bash
jupyter lab
```

The three desk notebooks (`06`, `07`, `08`) import directly from the `pricer` and `scripts` packages. Swap the chain path from `sample_chain.csv` to `live_chain.csv` to run the full workflow on live data.

---

## Data formats

### Chain CSV

Required columns:

| Column | Description |
|---|---|
| `underlying` | Ticker symbol |
| `expiry` | Expiry date (YYYY-MM-DD) |
| `cp` | `call` or `put` |
| `K` | Strike |
| `T` | Time to expiry in years |
| `spot` | Underlying spot price |
| `mid` | Option mid price (or `bid` and `ask` to compute mid) |

Optional but recommended:

| Column | Description |
|---|---|
| `r` | Risk-free rate (falls back to `--r` CLI argument if absent) |
| `q` | Continuous dividend yield (defaults to 0.0) |

`data/fetch_chain.py` writes all of these columns automatically.

### Positions CSV

Required columns:

| Column | Description |
|---|---|
| `underlying` | Ticker symbol |
| `expiry` | Expiry date (YYYY-MM-DD), must match chain |
| `cp` | `call` or `put` |
| `K` | Strike, must match chain |
| `qty` | Signed quantity in contracts (negative = short) |

---

## Theoretical background

### Barrier options

Barrier options are path-dependent: the payoff depends on whether the underlying crosses a barrier level B during the option's life. Knock-out options expire worthless if the barrier is breached; knock-in options only become active if it is.

### GBM under the risk-neutral measure

The underlying is modelled as GBM:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

Discretised for simulation:

$$S_{t+\Delta t} = S_t \exp\!\left(\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right), \quad Z \sim \mathcal{N}(0,1)$$

### Monte Carlo pricing

$$V_0 = e^{-rT}\,\mathbb{E}_Q\!\left[\text{Payoff}\right]$$

Estimated by simulating $N$ paths, applying barrier conditions to each, averaging discounted payoffs, and reporting a confidence interval via the central limit theorem. Error shrinks as $1/\sqrt{N}$.

### Black-Scholes and Greeks

For vanilla European options, Black-Scholes gives a closed-form price under the assumptions of constant vol, rates, and no jumps. The Greeks are analytic partial derivatives of the price with respect to each input. Implied volatility inverts this: given a market mid, find the volatility that makes the model price match.

### Continuity correction

Discrete barrier monitoring underestimates knock-out probability relative to continuous monitoring. The Broadie-Glasserman-Kou correction adjusts the barrier by $\pm\beta\sigma\sqrt{\Delta t}$ (where $\beta \approx 0.5826$) to approximate the continuous-monitoring price using a discrete simulation.

### SVI

Raw SVI parametrises the total implied variance $w(k) = \sigma_{\text{imp}}^2 T$ as a function of log-moneyness $k = \log(K/F)$:

$$w(k) = a + b\!\left(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\right)$$

Five parameters $(a, b, \rho, m, \sigma)$ are fitted by least squares. SVI is a common parametric form for fitting a smooth smile slice in practice.

### CRR binomial tree

The Cox-Ross-Rubinstein tree discretises the stock price into up and down moves $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$ with risk-neutral probability $p = (e^{(r-q)\Delta t} - d)/(u - d)$. Backward induction computes option values at each node. For American options, the value at each node is $\max(\text{hold value}, \text{intrinsic value})$.

---

## Limitations

- The barrier pricer assumes GBM with constant rates and volatility and European exercise only.
- The continuity correction is an approximation. For accurate continuous-monitoring prices, increase `N_steps` rather than relying on the correction alone.
- Black-Scholes assumes constant vol, no jumps, and continuous trading. It is used here as an analytics and calibration tool, not a production pricing model.
- N.B. Yahoo Finance data can be delayed outside market hours and bid/ask quotes may be stale for illiquid strikes. The fetch script warns when this is likely. For production use a proper market data vendor.
- Dollar Greeks in the risk report are first-order approximations. Scenario PnL uses delta and vega only and does not account for gamma, cross-effects, or term structure.
- The SVI fitter minimises unweighted least squares. In practice you would weight by bid/ask spread or vega to avoid fitting noise in illiquid wings.

---

## Roadmap

- Add GitHub Actions CI to run pytest on push
- Extend reconciliation with data quality checks on chain inputs (duplicate keys, stale quotes, missing spots)
- Improve surface handling across expiries with calendar spread arbitrage checks