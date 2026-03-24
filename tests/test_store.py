# tests/test_store.py
import pytest
import pandas as pd
from data.store import DataStore

class TestDataStore:
    def test_save_and_load_ohlcv(self, sample_ohlcv, tmp_data_dir):
        store = DataStore(tmp_data_dir / "store")
        store.save_ohlcv("BTC-USD", "1h", sample_ohlcv)
        loaded = store.load_ohlcv("BTC-USD", "1h")
        assert len(loaded) == len(sample_ohlcv)
        pd.testing.assert_frame_equal(loaded, sample_ohlcv, check_exact=False, atol=1e-6)

    def test_load_missing_returns_none(self, tmp_data_dir):
        store = DataStore(tmp_data_dir / "store")
        result = store.load_ohlcv("FAKE-USD", "1h")
        assert result is None

    def test_append_trade(self, tmp_data_dir):
        store = DataStore(tmp_data_dir / "store")
        trade = {
            "timestamp": "2025-01-01T00:00:00",
            "symbol": "BTC-USD",
            "side": "BUY",
            "size_usd": 1000.0,
            "price": 50000.0,
            "stop": 48000.0,
            "source": "ma_crossover",
        }
        store.append_trade(trade)
        store.append_trade(trade)
        trades = store.load_trades()
        assert len(trades) == 2

    def test_save_and_load_snapshot(self, tmp_data_dir):
        store = DataStore(tmp_data_dir / "store")
        snapshot = {"positions": [], "signals": [], "portfolio_value": 10000.0}
        store.save_snapshot(snapshot)
        loaded = store.load_latest_snapshot()
        assert loaded["portfolio_value"] == 10000.0
