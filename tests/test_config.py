"""Tests for odx.config — Config loading and env var override."""

import os

import pytest

from odx.config import Config


class TestConfigDefaults:
    def test_default_risk_free_rate(self) -> None:
        cfg = Config()
        assert cfg.default_risk_free_rate == 0.05

    def test_default_dividend_yield(self) -> None:
        cfg = Config()
        assert cfg.default_dividend_yield == 0.0

    def test_default_log_level(self) -> None:
        cfg = Config()
        assert cfg.log_level == "INFO"

    def test_api_keys_none_by_default(self) -> None:
        cfg = Config()
        assert cfg.polygon_api_key is None
        assert cfg.tradier_api_key is None

    def test_data_dir_default(self) -> None:
        cfg = Config()
        assert str(cfg.data_dir) == "data"


class TestConfigEnvOverride:
    def test_risk_free_rate_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ODX_DEFAULT_RISK_FREE_RATE", "0.04")
        cfg = Config()
        assert cfg.default_risk_free_rate == 0.04

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ODX_LOG_LEVEL", "DEBUG")
        cfg = Config()
        assert cfg.log_level == "DEBUG"

    def test_polygon_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ODX_POLYGON_API_KEY", "test_key_123")
        cfg = Config()
        assert cfg.polygon_api_key == "test_key_123"
