# tests/test_macd_rsi_confluence.py
import pytest
import pandas as pd
import numpy as np
from strategy.strategies.macd_rsi_confluence import MACDRSIConfluence
from strategy.base import Signal


def _make_df(rsi: float, macd: float, macd_signal: float,
             prev_macd: float, prev_macd_signal: float,
             atr: float = 500.0, close: float = 50000.0, n: int = 35):
    """Build a DataFrame with MACD, RSI, and ATR columns.
    The second-to-last row gets prev_macd/prev_macd_signal, and
    the last row gets the current values."""
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": [close] * n,
        "high": [close + 100] * n,
        "low": [close - 100] * n,
        "close": [close] * n,
        "volume": [1000.0] * n,
        "rsi": [rsi] * n,
        "macd": [prev_macd] * n,
        "macd_signal": [prev_macd_signal] * n,
        "atr": [atr] * n,
    }, index=dates)
    df.index.name = "timestamp"
    # Set the last row to the "current" values
    df.loc[df.index[-1], "macd"] = macd
    df.loc[df.index[-1], "macd_signal"] = macd_signal
    return df


class TestMACDRSIConfluence:
    def test_returns_list(self):
        """signals() should always return a list."""
        strat = MACDRSIConfluence()
        # MACD crosses up + RSI < 40 => BUY
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0)
        result = strat.signals(df)
        assert isinstance(result, list)

    def test_buy_requires_both_conditions(self):
        """BUY requires MACD cross up AND RSI < rsi_buy_threshold."""
        strat = MACDRSIConfluence(rsi_buy_threshold=40)
        # MACD crosses up + RSI = 30 (below 40) => BUY
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "BUY"

    def test_no_buy_without_macd_cross(self):
        """RSI oversold alone (no MACD cross) should not trigger BUY."""
        strat = MACDRSIConfluence(rsi_buy_threshold=40)
        # No MACD cross (MACD stays above signal)
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=1.0, prev_macd_signal=0.5)
        signals = strat.signals(df)
        assert signals == []

    def test_no_buy_without_low_rsi(self):
        """MACD cross up alone (RSI too high) should not trigger BUY."""
        strat = MACDRSIConfluence(rsi_buy_threshold=40)
        # MACD crosses up but RSI = 55 (above 40)
        df = _make_df(rsi=55.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0)
        signals = strat.signals(df)
        assert signals == []

    def test_sell_requires_both_conditions(self):
        """SELL requires MACD cross down AND RSI > rsi_short_threshold."""
        strat = MACDRSIConfluence(rsi_short_threshold=70)
        # MACD crosses down + RSI = 80 (above 70) => SELL
        df = _make_df(rsi=80.0, macd=-1.0, macd_signal=-0.5,
                       prev_macd=0.5, prev_macd_signal=0.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "SELL"

    def test_no_sell_without_macd_cross_down(self):
        """RSI overbought alone (no MACD cross down) should not trigger SELL."""
        strat = MACDRSIConfluence(rsi_short_threshold=70)
        # No cross (MACD stays below signal)
        df = _make_df(rsi=80.0, macd=-1.0, macd_signal=-0.5,
                       prev_macd=-1.0, prev_macd_signal=-0.5)
        signals = strat.signals(df)
        assert signals == []

    def test_empty_without_indicators(self):
        """Missing required columns should return empty list."""
        strat = MACDRSIConfluence()
        dates = pd.date_range("2025-01-01", periods=35, freq="1h")
        df = pd.DataFrame({
            "open": [50000.0] * 35,
            "high": [50100.0] * 35,
            "low": [49900.0] * 35,
            "close": [50000.0] * 35,
            "volume": [1000.0] * 35,
        }, index=dates)
        df.index.name = "timestamp"
        signals = strat.signals(df)
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        """Fewer than 30 rows should return empty."""
        strat = MACDRSIConfluence()
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0, n=20)
        signals = strat.signals(df)
        assert signals == []

    def test_strength_between_zero_and_one(self):
        """Signal strength should be clamped between 0 and 1."""
        strat = MACDRSIConfluence(rsi_buy_threshold=40)
        # Very low RSI => high strength, but capped at 1.0
        df = _make_df(rsi=5.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert 0.0 <= signals[0].strength <= 1.0

    def test_buy_signal_has_stop_and_tp(self):
        """BUY signal should have ATR-based stop and take-profit."""
        strat = MACDRSIConfluence(atr_stop=2.0, atr_tp=3.0)
        close = 50000.0
        atr = 500.0
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0,
                       atr=atr, close=close)
        signals = strat.signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.stop_price == pytest.approx(close - atr * 2.0)
        assert sig.take_profit == pytest.approx(close + atr * 3.0)

    def test_sell_signal_has_stop_and_tp(self):
        """SELL signal should have inverted ATR-based stop and TP."""
        strat = MACDRSIConfluence(atr_stop=2.0, atr_tp=3.0)
        close = 50000.0
        atr = 500.0
        df = _make_df(rsi=80.0, macd=-1.0, macd_signal=-0.5,
                       prev_macd=0.5, prev_macd_signal=0.0,
                       atr=atr, close=close)
        signals = strat.signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.stop_price == pytest.approx(close + atr * 2.0)
        assert sig.take_profit == pytest.approx(close - atr * 3.0)

    def test_metadata_includes_leverage(self):
        """Signal metadata should include leverage field."""
        strat = MACDRSIConfluence(rsi_buy_threshold=40)
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0)
        signals = strat.signals(df)
        assert len(signals) == 1
        assert "leverage" in signals[0].metadata

    def test_zero_atr_returns_empty(self):
        """Zero ATR should return empty."""
        strat = MACDRSIConfluence()
        df = _make_df(rsi=30.0, macd=1.0, macd_signal=0.5,
                       prev_macd=-0.5, prev_macd_signal=0.0, atr=0.0)
        signals = strat.signals(df)
        assert signals == []
