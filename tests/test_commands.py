# tests/test_commands.py
import pytest
import json
from unittest.mock import patch, MagicMock
from cli.commands import CLI


class TestCLI:
    def test_status_returns_json(self):
        cli = CLI.__new__(CLI)
        cli.tracker = MagicMock()
        cli.tracker.open_positions.return_value = [
            {"symbol": "BTC-USD", "side": "BUY", "size_usd": 1000, "unrealized_pnl": 50}
        ]
        cli.tracker.total_exposure.return_value = 1050.0
        cli.tracker.closed_trades.return_value = []
        cli.executor = MagicMock()
        cli.executor.get_balances.return_value = {"USD": 9000.0}
        result = cli.status()
        assert "positions" in result
        assert "balances" in result

    def test_pending_returns_signals(self):
        cli = CLI.__new__(CLI)
        cli.store = MagicMock()
        cli.store.load_latest_snapshot.return_value = {
            "signals": [{"symbol": "BTC-USD", "direction": "BUY", "strength": 0.8}]
        }
        result = cli.pending()
        assert len(result["signals"]) == 1

    def test_pending_returns_empty_when_no_snapshot(self):
        cli = CLI.__new__(CLI)
        cli.store = MagicMock()
        cli.store.load_latest_snapshot.return_value = None
        result = cli.pending()
        assert result["signals"] == []
        assert "message" in result

    def test_snapshot_returns_latest(self):
        cli = CLI.__new__(CLI)
        cli.store = MagicMock()
        cli.store.load_latest_snapshot.return_value = {"timestamp": "2026-03-23", "data": "test"}
        result = cli.snapshot()
        assert result["timestamp"] == "2026-03-23"

    def test_snapshot_returns_empty_dict_when_none(self):
        cli = CLI.__new__(CLI)
        cli.store = MagicMock()
        cli.store.load_latest_snapshot.return_value = None
        result = cli.snapshot()
        assert result == {}

    def test_buy_checks_risk_before_executing(self):
        cli = CLI.__new__(CLI)
        cli.risk = MagicMock()
        cli.risk.check_entry.return_value = (False, "Position too large")
        cli.tracker = MagicMock()
        cli.tracker.open_positions.return_value = []
        cli.executor = MagicMock()
        result = cli.buy("BTC-USD", 5000.0)
        assert result == {"error": "Position too large"}
        cli.executor.market_buy.assert_not_called()

    def test_buy_executes_and_tracks_on_success(self):
        cli = CLI.__new__(CLI)
        cli.risk = MagicMock()
        cli.risk.check_entry.return_value = (True, "OK")
        cli.tracker = MagicMock()
        cli.tracker.open_positions.return_value = []
        cli.executor = MagicMock()
        cli.executor.market_buy.return_value = {"success": True, "order_id": "123"}
        cli.fetcher = MagicMock()
        cli.fetcher.ticker.return_value = {"last": 50000.0}
        cli.store = MagicMock()
        result = cli.buy("BTC-USD", 1000.0)
        assert result["success"] is True
        cli.tracker.open.assert_called_once()
        cli.store.append_trade.assert_called_once()

    def test_sell_delegates_to_executor(self):
        cli = CLI.__new__(CLI)
        cli.executor = MagicMock()
        cli.executor.market_sell.return_value = {"success": True}
        result = cli.sell("BTC-USD", 0.02)
        assert result["success"] is True
        cli.executor.market_sell.assert_called_once_with("BTC-USD", 0.02)

    def test_close_all_closes_every_position(self):
        cli = CLI.__new__(CLI)
        cli.tracker = MagicMock()
        cli.tracker.open_positions.return_value = [
            {"symbol": "BTC-USD", "size_usd": 1000, "current_price": 50000},
            {"symbol": "ETH-USD", "size_usd": 500, "current_price": 3000},
        ]
        cli.executor = MagicMock()
        cli.executor.market_sell.return_value = {"success": True}
        cli.fetcher = MagicMock()
        cli.fetcher.ticker.return_value = {"last": 50000.0}
        cli.tracker.close.return_value = {"pnl_usd": 100}
        results = cli.close_all()
        assert len(results) == 2

    def test_short_checks_risk(self):
        cli = CLI.__new__(CLI)
        cli.risk = MagicMock()
        cli.risk.check_entry.return_value = (False, "Max concurrent positions (3) reached")
        cli.tracker = MagicMock()
        cli.tracker.open_positions.return_value = [1, 2, 3]
        result = cli.short("BTC-PERP-INTX", 2000.0, leverage=2)
        assert "error" in result

    def test_cancel_delegates_to_executor(self):
        cli = CLI.__new__(CLI)
        cli.executor = MagicMock()
        cli.executor.cancel_order.return_value = {"success": True}
        result = cli.cancel("order-abc-123")
        cli.executor.cancel_order.assert_called_once_with("order-abc-123")
        assert result["success"] is True
