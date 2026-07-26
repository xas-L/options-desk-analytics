# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- **Core CLI**: Main entrypoint `python -m odx` with subcommands (`surface`, `cone`, `backtest`, `walk-forward`).
- **Market Making Engine**: LOBSTER orderbook and message data parser.
- **Queue Tracker**: Probabilistic FIFO queue model for order fills in `lobster_replay.py`.
- **GPU Pricers**: `cupy` accelerated implementations for Monte Carlo Heston and Rough Bergomi with transparent NumPy CPU fallbacks.
- **Research Harness**: Walk-forward calibration tester and stability report generator.
- **Strategies**: SPX vs Single-Stock dispersion strategy builder and implied correlation back-out.
- **Volatility**: Two-regime Markov-switching variance model.
- **Documentation**: SPX dispersion case study, interview walkthrough guide, and known limitations overview.

## [0.1.0] - 2026-07-26
### Added
- Initial project structure.
- Pure Python and C++ bindings (`pybind11`) for Black-Scholes and Heston FFT pricing.
- Market data adapters (Tradier, Polygon.io).
- Gatheral SSVI Volatility Surface calibration.
- P&L dashboard via Streamlit.
