"""Fetch a live options chain from Yahoo Finance and write it in the project's
standard CSV format so it drops straight into chain_to_iv_greeks.py and
portfolio_risk_report.py without any changes.

Output columns match data/sample_chain.csv exactly:
    underlying, expiry, cp, K, T, spot, r, q, bid, ask, mid

Usage:
    python data/fetch_chain.py --tickers AAPL MSFT
    python data/fetch_chain.py --tickers SPY --expiries 2025-06-20 2025-09-19
    python data/fetch_chain.py --tickers AAPL --min-bid 0.05 --min-volume 10 --out data/live_chain.csv

Data source: Yahoo Finance via yfinance. Quotes can be stale outside market hours
and bid/ask may be zero for deeply illiquid strikes. Use --min-bid and --min-volume
to filter these out before passing the chain to any pricing script.

Risk-free rate: 13-week T-bill (^IRX) pulled live. Falls back to --r-fallback
if the feed is unavailable.

Dividend yield: from Yahoo ticker metadata. Falls back to 0.0 if not published.

Install dependency: pip install yfinance
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _fetch_risk_free_rate(fallback: float) -> float:
    """Pull the annualised 13-week T-bill rate from Yahoo (^IRX). Returns fallback on failure."""
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if hist.empty:
            print(f"  Warning: ^IRX feed empty, using r={fallback:.4f}")
            return fallback
        rate = float(hist["Close"].iloc[-1]) / 100.0
        print(f"  Risk-free rate (^IRX 13W T-bill): {rate:.4f}")
        return rate
    except Exception as exc:
        print(f"  Warning: could not fetch ^IRX ({exc}), using r={fallback:.4f}")
        return fallback


def _fetch_dividend_yield(ticker: yf.Ticker, symbol: str) -> float:
    """Return annualised continuous dividend yield from Yahoo metadata."""
    try:
        info = ticker.info
        div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0
        return float(div_yield)
    except Exception:
        print(f"  Warning: could not fetch dividend yield for {symbol}, using q=0.0")
        return 0.0


def _time_to_expiry(expiry_str: str) -> float:
    """Calendar days from today to expiry divided by 365. Returns 0 if expiry has passed."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp - date.today()).days
    return max(days / 365.0, 0.0)


def _fetch_single_expiry(
    ticker: yf.Ticker,
    symbol: str,
    expiry: str,
    spot: float,
    r: float,
    q: float,
    min_bid: float,
    min_volume: int,
    min_open_interest: int,
) -> pd.DataFrame:
    """Fetch call and put chains for one expiry and return a tidy dataframe."""
    try:
        chain = ticker.option_chain(expiry)
    except Exception as exc:
        print(f"  Warning: could not fetch chain for {symbol} {expiry} ({exc}), skipping")
        return pd.DataFrame()

    T = _time_to_expiry(expiry)
    if T <= 0.0:
        print(f"  Skipping {expiry}: expiry has passed")
        return pd.DataFrame()

    rows = []
    for cp_label, df in [("call", chain.calls), ("put", chain.puts)]:
        if df.empty:
            continue

        df = df.copy()
        df["cp"] = cp_label
        df["underlying"] = symbol
        df["expiry"] = expiry
        df["T"] = T
        df["spot"] = spot
        df["r"] = r
        df["q"] = q

        df = df.rename(columns={"strike": "K"})

        if "bid" not in df.columns or "ask" not in df.columns:
            print(f"  Warning: bid/ask missing for {symbol} {expiry} {cp_label}, skipping")
            continue

        df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0.0)
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0.0)
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)
        df["openInterest"] = pd.to_numeric(df.get("openInterest", 0), errors="coerce").fillna(0).astype(int)

        df = df[df["bid"] >= min_bid]
        df = df[df["volume"] >= min_volume]
        df = df[df["openInterest"] >= min_open_interest]
        df = df[df["mid"] > 0.0]

        rows.append(df[["underlying", "expiry", "cp", "K", "T", "spot", "r", "q", "bid", "ask", "mid"]])

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def fetch_chain(
    symbols: list[str],
    expiry_filter: Optional[list[str]] = None,
    r_fallback: float = 0.05,
    min_bid: float = 0.05,
    min_volume: int = 0,
    min_open_interest: int = 0,
    out_path: str = "data/live_chain.csv",
) -> pd.DataFrame:
    """Fetch live option chains for one or more symbols and write to CSV.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols, e.g. ['AAPL', 'MSFT']
    expiry_filter : list[str], optional
        Restrict to these expiry dates (YYYY-MM-DD). Fetches all if None.
    r_fallback : float
        Risk-free rate to use if ^IRX is unavailable.
    min_bid : float
        Drop rows where bid is below this level. Removes deeply illiquid strikes.
    min_volume : int
        Drop rows with daily volume below this. 0 keeps everything.
    min_open_interest : int
        Drop rows with open interest below this. 0 keeps everything.
    out_path : str
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        Chain in project-standard column format.
    """
    r = _fetch_risk_free_rate(r_fallback)
    all_frames = []

    for symbol in symbols:
        symbol = symbol.upper()
        print(f"\nFetching {symbol}...")

        ticker = yf.Ticker(symbol)

        try:
            hist = ticker.history(period="2d")
            if hist.empty:
                print(f"  Warning: no price history for {symbol}, skipping")
                continue
            spot = float(hist["Close"].iloc[-1])
            print(f"  Spot: {spot:.2f}")
        except Exception as exc:
            print(f"  Warning: could not fetch spot for {symbol} ({exc}), skipping")
            continue

        q = _fetch_dividend_yield(ticker, symbol)
        print(f"  Dividend yield: {q:.4f}")

        try:
            available_expiries = ticker.options
        except Exception as exc:
            print(f"  Warning: could not fetch expiry list for {symbol} ({exc}), skipping")
            continue

        if not available_expiries:
            print(f"  Warning: no expiries available for {symbol}")
            continue

        expiries_to_fetch = (
            [e for e in available_expiries if e in expiry_filter]
            if expiry_filter
            else list(available_expiries)
        )

        if not expiries_to_fetch:
            print(f"  Warning: none of the requested expiries found for {symbol}")
            print(f"  Available: {list(available_expiries)}")
            continue

        print(f"  Fetching {len(expiries_to_fetch)} expir{'y' if len(expiries_to_fetch) == 1 else 'ies'}: {expiries_to_fetch}")

        for expiry in expiries_to_fetch:
            df = _fetch_single_expiry(
                ticker, symbol, expiry, spot, r, q,
                min_bid, min_volume, min_open_interest,
            )
            if not df.empty:
                all_frames.append(df)
                print(f"    {expiry}: {len(df)} rows after filtering")

    if not all_frames:
        print("\nNo data fetched. Check tickers and expiry dates.")
        return pd.DataFrame()

    chain = pd.concat(all_frames, ignore_index=True)
    chain = chain.drop_duplicates(subset=["underlying", "expiry", "cp", "K"])
    chain = chain.sort_values(["underlying", "expiry", "cp", "K"]).reset_index(drop=True)

    chain.to_csv(out_path, index=False)
    print(f"\nWritten {len(chain)} rows to {out_path}")

    _print_summary(chain)
    return chain


def _print_summary(chain: pd.DataFrame) -> None:
    """Print a brief summary of what was fetched."""
    print("\nSummary:")
    summary = (
        chain.groupby(["underlying", "expiry", "cp"])
        .agg(n_strikes=("K", "count"), min_K=("K", "min"), max_K=("K", "max"))
        .reset_index()
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a live options chain from Yahoo Finance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/fetch_chain.py --tickers AAPL MSFT
  python data/fetch_chain.py --tickers SPY --expiries 2025-06-20 2025-09-19
  python data/fetch_chain.py --tickers AAPL --min-bid 0.05 --min-volume 10 --out data/live_chain.csv
        """,
    )
    parser.add_argument("--tickers", nargs="+", required=True, help="One or more ticker symbols")
    parser.add_argument("--expiries", nargs="+", default=None, metavar="YYYY-MM-DD",
                        help="Restrict to specific expiry dates. Fetches all if omitted.")
    parser.add_argument("--out", default="data/live_chain.csv", help="Output CSV path")
    parser.add_argument("--r-fallback", type=float, default=0.05,
                        help="Risk-free rate to use if ^IRX feed is unavailable (default: 0.05)")
    parser.add_argument("--min-bid", type=float, default=0.05,
                        help="Drop rows where bid is below this value (default: 0.05)")
    parser.add_argument("--min-volume", type=int, default=0,
                        help="Drop rows with daily volume below this (default: 0, keep all)")
    parser.add_argument("--min-oi", type=int, default=0,
                        help="Drop rows with open interest below this (default: 0, keep all)")

    args = parser.parse_args()

    fetch_chain(
        symbols=args.tickers,
        expiry_filter=args.expiries,
        r_fallback=args.r_fallback,
        min_bid=args.min_bid,
        min_volume=args.min_volume,
        min_open_interest=args.min_oi,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
