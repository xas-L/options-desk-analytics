"""Tests for the chain cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from odx.data.chain import ChainSnapshot
from odx.data.cleaning import clean_chain


def _make_row(**overrides) -> dict:
    """Helper to build a single standard-schema row with sensible defaults."""
    row = {
        "underlying": "TEST",
        "expiry": "2027-06-20",
        "cp": "call",
        "K": 100.0,
        "T": 1.0,
        "spot": 100.0,
        "r": 0.05,
        "q": 0.01,
        "bid": 5.0,
        "ask": 5.5,
        "mid": 5.25,
        "volume": 100,
        "openInterest": 500,
    }
    row.update(overrides)
    return row


def _make_chain(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ChainSnapshot

class TestChainSnapshot:
    def test_from_dataframe(self) -> None:
        df = _make_chain([_make_row(), _make_row(K=110.0, cp="put")])
        snap = ChainSnapshot.from_dataframe(df)
        assert snap.underlying == "TEST"
        assert snap.spot == 100.0
        assert snap.r == 0.05
        assert snap.q == 0.01
        assert len(snap.chain) == 2

    def test_strikes(self) -> None:
        df = _make_chain([_make_row(K=100.0), _make_row(K=110.0), _make_row(K=100.0)])
        snap = ChainSnapshot.from_dataframe(df)
        assert list(snap.strikes) == [100.0, 110.0]

    def test_expiries(self) -> None:
        df = _make_chain([
            _make_row(expiry="2027-06-20"),
            _make_row(expiry="2027-09-19"),
        ])
        snap = ChainSnapshot.from_dataframe(df)
        assert snap.expiries == ["2027-06-20", "2027-09-19"]

    def test_calls_puts(self) -> None:
        df = _make_chain([_make_row(cp="call"), _make_row(cp="put")])
        snap = ChainSnapshot.from_dataframe(df)
        assert len(snap.calls) == 1
        assert len(snap.puts) == 1

    def test_expiry_slice(self) -> None:
        df = _make_chain([
            _make_row(expiry="2027-06-20"),
            _make_row(expiry="2027-09-19"),
        ])
        snap = ChainSnapshot.from_dataframe(df)
        sl = snap.expiry_slice("2027-06-20")
        assert len(sl) == 1

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            ChainSnapshot.from_dataframe(pd.DataFrame())


# Cleaning pipeline

class TestCleanChain:
    def test_empty_input(self) -> None:
        clean, flagged = clean_chain(pd.DataFrame())
        assert clean.empty
        assert flagged.empty

    def test_drops_expired(self) -> None:
        df = _make_chain([
            _make_row(T=1.0),
            _make_row(T=0.0),
            _make_row(T=-0.5),
        ])
        clean, flagged = clean_chain(df)
        assert len(clean) == 1
        assert len(flagged) == 2
        assert all("expired" in r for r in flagged["flag_reason"])

    def test_drops_crossed_markets(self) -> None:
        df = _make_chain([
            _make_row(bid=5.0, ask=5.5),
            _make_row(bid=6.0, ask=5.0),  # crossed
        ])
        clean, flagged = clean_chain(df)
        assert len(clean) == 1
        assert len(flagged) == 1
        assert "crossed" in flagged.iloc[0]["flag_reason"]

    def test_drops_zero_oi_and_volume(self) -> None:
        df = _make_chain([
            _make_row(volume=100, openInterest=500),
            _make_row(volume=0, openInterest=0),
        ])
        clean, flagged = clean_chain(df)
        assert len(clean) == 1

    def test_drops_non_positive_mid(self) -> None:
        df = _make_chain([
            _make_row(mid=5.25),
            _make_row(mid=0.0),
            _make_row(mid=-1.0),
        ])
        clean, flagged = clean_chain(df)
        assert len(clean) == 1

    def test_optional_zero_oi_flag(self) -> None:
        df = _make_chain([
            _make_row(openInterest=500, volume=100),
            _make_row(openInterest=0, volume=100),
        ])
        clean_with, _ = clean_chain(df, drop_zero_oi=True)
        clean_without, _ = clean_chain(df, drop_zero_oi=False)
        assert len(clean_with) == 1
        assert len(clean_without) == 2

    def test_optional_zero_volume_flag(self) -> None:
        df = _make_chain([
            _make_row(volume=100, openInterest=500),
            _make_row(volume=0, openInterest=500),
        ])
        clean_with, _ = clean_chain(df, drop_zero_volume=True)
        clean_without, _ = clean_chain(df, drop_zero_volume=False)
        assert len(clean_with) == 1
        assert len(clean_without) == 2

    def test_flag_reason_populated(self) -> None:
        df = _make_chain([
            _make_row(T=0.0),
            _make_row(bid=10.0, ask=5.0),
        ])
        _, flagged = clean_chain(df)
        assert "flag_reason" in flagged.columns
        assert len(flagged) == 2

    def test_all_clean_returns_empty_flagged(self) -> None:
        df = _make_chain([_make_row(), _make_row(K=110.0)])
        clean, flagged = clean_chain(df)
        assert len(clean) == 2
        assert flagged.empty
