# tests/test_indicators.py
import pytest
import pandas as pd
from data.indicators import add_indicators

class TestIndicators:
    def test_adds_sma_columns(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "sma_20" in df.columns
        assert "sma_50" in df.columns

    def test_adds_rsi(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "rsi" in df.columns
        assert df["rsi"].dropna().between(0, 100).all()

    def test_adds_macd(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_macd_hist_equals_macd_minus_signal(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        valid = df[["macd", "macd_signal", "macd_hist"]].dropna()
        expected_hist = valid["macd"] - valid["macd_signal"]
        assert (abs(valid["macd_hist"] - expected_hist) < 1e-10).all()

    def test_adds_bollinger_bands(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns

    def test_adds_atr(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "atr" in df.columns
        assert (df["atr"].dropna() > 0).all()

    def test_adds_ema(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "ema_12" in df.columns
        assert "ema_26" in df.columns

    def test_does_not_mutate_input(self, sample_ohlcv):
        original_cols = list(sample_ohlcv.columns)
        add_indicators(sample_ohlcv)
        assert list(sample_ohlcv.columns) == original_cols

    def test_adds_stochrsi(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "stochrsi_k" in df.columns
        assert "stochrsi_d" in df.columns
        valid = df["stochrsi_k"].dropna()
        assert len(valid) > 0
        assert valid.between(0, 100).all()

    def test_adds_roc_5(self, sample_ohlcv):
        df = add_indicators(sample_ohlcv)
        assert "roc_5" in df.columns
        valid = df["roc_5"].dropna()
        assert len(valid) > 0
