# tests/test_rsi_mean_reversion.py
import pytest
import pandas as pd
import numpy as np
from strategy.strategies.rsi_mean_reversion import RSIMeanReversion
from strategy.base import Signal


def _make_df(rsi: float, atr: float, close: float = 50000.0, n: int = 25):
    """Build a minimal DataFrame with rsi and atr columns."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": [close] * n,
        "high": [close + 100] * n,
        "low": [close - 100] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
        "rsi": [rsi] * n,
        "atr": [atr] * n,
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestRSIMeanReversion:
    def test_buy_signal_on_oversold(self):
        """RSI below oversold threshold should produce a BUY signal."""
        strat = RSIMeanReversion(oversold=30, overbought=70)
        df = _make_df(rsi=20.0, atr=500.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "BUY"
        assert "oversold" in signals[0].metadata["reason"]

    def test_sell_signal_on_overbought(self):
        """RSI above overbought threshold should produce a SELL signal."""
        strat = RSIMeanReversion(oversold=30, overbought=70)
        df = _make_df(rsi=85.0, atr=500.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "SELL"
        assert "overbought" in signals[0].metadata["reason"]

    def test_no_signal_in_neutral_zone(self):
        """RSI in the neutral zone should produce no signals."""
        strat = RSIMeanReversion(oversold=30, overbought=70)
        df = _make_df(rsi=50.0, atr=500.0)
        signals = strat.signals(df)
        assert signals == []

    def test_empty_without_indicators(self):
        """Missing rsi/atr columns should return empty list."""
        strat = RSIMeanReversion()
        dates = pd.date_range("2025-01-01", periods=25, freq="1h")
        df = pd.DataFrame({
            "open": [50000.0] * 25,
            "high": [50100.0] * 25,
            "low": [49900.0] * 25,
            "close": [50000.0] * 25,
            "volume": [1000.0] * 25,
        }, index=dates)
        df.index.name = "timestamp"
        signals = strat.signals(df)
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        """Fewer than 20 rows should return empty list."""
        strat = RSIMeanReversion()
        df = _make_df(rsi=20.0, atr=500.0, n=10)
        signals = strat.signals(df)
        assert signals == []

    def test_signal_has_stop_and_take_profit(self):
        """BUY signal should have correct ATR-based stop and TP."""
        strat = RSIMeanReversion(atr_stop=2.0, atr_tp=3.0)
        close = 50000.0
        atr = 500.0
        df = _make_df(rsi=20.0, atr=atr, close=close)
        signals = strat.signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.stop_price == pytest.approx(close - atr * 2.0)
        assert sig.take_profit == pytest.approx(close + atr * 3.0)

    def test_sell_signal_stop_and_take_profit(self):
        """SELL signal should have correct ATR-based stop and TP (inverted)."""
        strat = RSIMeanReversion(atr_stop=2.0, atr_tp=3.0)
        close = 50000.0
        atr = 500.0
        df = _make_df(rsi=85.0, atr=atr, close=close)
        signals = strat.signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.stop_price == pytest.approx(close + atr * 2.0)
        assert sig.take_profit == pytest.approx(close - atr * 3.0)

    def test_strength_between_zero_and_one(self):
        """Signal strength should be capped at 1.0 and >= 0."""
        strat = RSIMeanReversion(oversold=30, overbought=70)
        # Extreme oversold
        df = _make_df(rsi=5.0, atr=500.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert 0.0 <= signals[0].strength <= 1.0
        # Extreme overbought
        df = _make_df(rsi=95.0, atr=500.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert 0.0 <= signals[0].strength <= 1.0

    def test_nan_rsi_returns_empty(self):
        """NaN RSI should return empty."""
        strat = RSIMeanReversion()
        df = _make_df(rsi=float("nan"), atr=500.0)
        signals = strat.signals(df)
        assert signals == []

    def test_zero_atr_returns_empty(self):
        """Zero ATR should return empty."""
        strat = RSIMeanReversion()
        df = _make_df(rsi=20.0, atr=0.0)
        signals = strat.signals(df)
        assert signals == []
