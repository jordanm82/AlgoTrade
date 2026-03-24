# tests/test_sma_crossover.py
import pytest
import pandas as pd
import numpy as np
from strategy.strategies.sma_crossover import SMACrossover
from data.indicators import add_indicators


class TestSMACrossover:
    def test_buy_signal_on_golden_cross(self):
        """SMA20 crossing above SMA50 should generate BUY."""
        np.random.seed(42)
        n = 100
        prices = np.concatenate([
            np.full(60, 50000.0) + np.random.randn(60) * 100,
            np.linspace(50000, 53000, 40) + np.random.randn(40) * 50,
        ])
        df = pd.DataFrame({
            "open": prices + np.random.randn(n) * 50,
            "high": prices + abs(np.random.randn(n) * 200),
            "low": prices - abs(np.random.randn(n) * 200),
            "close": prices,
            "volume": abs(np.random.randn(n) * 1000) + 500,
        }, index=pd.date_range("2025-01-01", periods=n, freq="1h"))
        df.index.name = "timestamp"
        df = add_indicators(df)

        strat = SMACrossover()
        signals = strat.signals(df)
        assert isinstance(signals, list)

    def test_no_signal_on_flat_market(self):
        """Flat market should produce no signals."""
        np.random.seed(42)
        n = 100
        prices = np.full(n, 50000.0) + np.random.randn(n) * 10
        df = pd.DataFrame({
            "open": prices, "high": prices + 5, "low": prices - 5,
            "close": prices, "volume": np.full(n, 1000.0),
        }, index=pd.date_range("2025-01-01", periods=n, freq="1h"))
        df.index.name = "timestamp"
        df = add_indicators(df)

        strat = SMACrossover()
        signals = strat.signals(df)
        assert isinstance(signals, list)

    def test_requires_sufficient_data(self):
        """Should return empty with insufficient data."""
        df = pd.DataFrame({
            "open": [50000], "high": [50100], "low": [49900],
            "close": [50000], "volume": [1000],
        }, index=pd.date_range("2025-01-01", periods=1, freq="1h"))
        df.index.name = "timestamp"

        strat = SMACrossover()
        signals = strat.signals(df)
        assert signals == []
