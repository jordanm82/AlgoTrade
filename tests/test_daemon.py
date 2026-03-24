# tests/test_daemon.py
import pytest
from unittest.mock import patch, MagicMock
from cli.daemon import Daemon


class TestDaemon:
    def test_tick_enforces_stops(self):
        daemon = Daemon.__new__(Daemon)
        daemon.tracker = MagicMock()
        daemon.tracker.check_stops.return_value = [
            {"symbol": "BTC-USD", "side": "BUY", "stop_price": 48000, "size_usd": 1000, "current_price": 47000}
        ]
        daemon.executor = MagicMock()
        daemon.executor.market_sell.return_value = {"success": True}
        daemon.store = MagicMock()
        daemon.risk = MagicMock()
        daemon.risk.is_halted.return_value = False
        daemon._enforce_stops()
        assert daemon.tracker.close.called

    def test_collect_snapshot(self):
        daemon = Daemon.__new__(Daemon)
        daemon.strategies = []
        daemon.fetcher = MagicMock()
        daemon.store = MagicMock()
        daemon.tracker = MagicMock()
        daemon.tracker.open_positions.return_value = []
        daemon.tracker.total_exposure.return_value = 0.0
        daemon.risk = MagicMock()
        daemon.risk.is_halted.return_value = False
        snapshot = daemon._collect_snapshot({"BTC-USD": MagicMock()})
        assert "signals" in snapshot
        assert "positions" in snapshot
