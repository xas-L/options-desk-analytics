# Interview Walkthrough Guide

This guide is designed to help you navigate quantitative trading and research interviews using this repository as a showcase of your engineering and modeling capabilities.

## The "Elevator Pitch"
> "I built an end-to-end options desk analytics platform from scratch in Python and C++. It handles market data parsing (including high-frequency LOBSTER data), constructs volatility surfaces using Gatheral's SSVI to ensure absence of arbitrage, and prices derivatives using both native C++ Fast Fourier Transforms (for Heston) and GPU-accelerated Monte Carlo (for Rough Bergomi). I also built out the trading infrastructure to backtest delta-hedging, index dispersion, and market-making strategies."

## Key Discussion Points

### 1. Architectural Decisions
**"Why did you use C++ for some parts and Python for others?"**
*   *Answer:* Python is excellent for data manipulation (`pandas`), API integrations, and rapid strategy prototyping. However, pricing engines inside a calibration loop or a high-frequency market-making replay need raw speed. Writing the Heston characteristic function and Radix-2 FFT in C++ and binding it with `pybind11` gave a 50x speedup while maintaining Python's ease of use.

### 2. Volatility Modeling
**"How do you guarantee your volatility surface is free of arbitrage?"**
*   *Answer:* I used Gatheral's SSVI (Surface Stochastic Volatility Inspired) parameterization. By enforcing the specific bounds on the correlation parameter $\rho$ and the variance angle $\phi$, we guarantee the absence of static arbitrage (both calendar spread arbitrage and butterfly arbitrage), which is crucial because a pricer will return negative probabilities/prices on an arbitragable surface.

### 3. High-Frequency Market Making
**"How did you model market impact and queue position?"**
*   *Answer:* I implemented a FIFO queue tracker to simulate limit order book mechanics using LOBSTER data. Instead of assuming instant fills at the touch, the model tracks the volume ahead of our quotes. Cancellations are modeled probabilistically, and executions deplete the queue deterministically, giving a much more realistic simulation of the Avellaneda-Stoikov market maker's actual fill rates.

### 4. GPU Acceleration
**"Why use CuPy with a NumPy fallback?"**
*   *Answer:* Advanced path-dependent models like Rough Bergomi require simulating Fractional Brownian Motion, which is computationally intense due to the exact Cholesky decomposition of the dense covariance matrix. By writing the Monte Carlo tensor operations using `cupy`, we get massive parallelization on Nvidia GPUs, while the `try/except ImportError` block falling back to `numpy` ensures the code remains fully portable to standard developer laptops without CUDA toolkits.
