import json
from pathlib import Path

import pytest
from exchange import positions as positions_module
from exchange.positions import Position, PositionTracker


@pytest.fixture(autouse=True)
def _isolate_positions_file(tmp_path, monkeypatch):
    """Redirect POSITIONS_FILE to a temp directory so tests don't
    write to the real data/store/ and don't interfere with each other."""
    tmp_file = tmp_path / "positions.json"
    monkeypatch.setattr(positions_module, "POSITIONS_FILE", tmp_file)


class TestPositionTracker:
    def test_open_position(self):
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        positions = tracker.open_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC-USD"
        assert positions[0]["side"] == "BUY"

    def test_close_position(self):
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        closed = tracker.close("BTC-USD", 51000.0)
        assert closed["pnl_usd"] == pytest.approx(20.0, abs=0.1)  # (51000-50000)/50000 * 1000
        assert len(tracker.open_positions()) == 0

    def test_max_concurrent_check(self):
        tracker = PositionTracker(max_concurrent=2)
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.open("ETH-USD", "BUY", 500.0, 3000.0, 2800.0, 3500.0)
        assert tracker.can_open() is False

    def test_update_pnl(self):
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.update_price("BTC-USD", 52000.0)
        pos = tracker.open_positions()[0]
        assert pos["unrealized_pnl"] == pytest.approx(40.0, abs=0.1)

    def test_stop_hit_detection(self):
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.update_price("BTC-USD", 47000.0)
        stops = tracker.check_stops()
        assert len(stops) == 1
        assert stops[0]["symbol"] == "BTC-USD"

    def test_portfolio_value(self):
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.update_price("BTC-USD", 51000.0)
        assert tracker.total_exposure() == pytest.approx(1020.0, abs=1.0)


class TestProfitTaking:
    def test_original_size_usd_preserved(self):
        """original_size_usd should be set on open and not change after reduce."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        pos = tracker._positions["BTC-USD"]
        assert pos.original_size_usd == 1000.0
        pos.reduce_size(25)
        assert pos.original_size_usd == 1000.0
        assert pos.size_usd == pytest.approx(750.0)

    def test_reduce_size(self):
        """reduce_size should correctly reduce size_usd and units."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        pos = tracker._positions["BTC-USD"]
        initial_units = pos.units  # 1000 / 50000 = 0.02
        assert initial_units == pytest.approx(0.02)

        reduction = pos.reduce_size(25)
        assert reduction == pytest.approx(250.0)
        assert pos.size_usd == pytest.approx(750.0)
        assert pos.units == pytest.approx(0.015)

        reduction2 = pos.reduce_size(25)
        # 25% of 750 = 187.5
        assert reduction2 == pytest.approx(187.5)
        assert pos.size_usd == pytest.approx(562.5)

    def test_partial_profit_taking_at_10_pct(self):
        """At +10% gain, should trigger a 25% partial sell."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 100.0, 90.0, 150.0)
        pos = tracker._positions["BTC-USD"]

        # Price at +5% -- no action
        pos.update(105.0)
        actions = pos.check_profit_taking()
        assert len(actions) == 0
        assert pos.tp_10_hit is False

        # Price at +10% -- should trigger TP 10%
        pos.update(110.0)
        actions = pos.check_profit_taking()
        assert len(actions) == 1
        assert actions[0]["action"] == "partial_sell"
        assert actions[0]["pct"] == 25
        assert "TP +10%" in actions[0]["reason"]
        assert pos.tp_10_hit is True

        # Calling again at +10% should NOT re-trigger
        actions = pos.check_profit_taking()
        assert all(a["action"] != "partial_sell" or "TP +10%" not in a.get("reason", "") for a in actions)

    def test_partial_profit_taking_at_20_pct(self):
        """At +20% gain, should trigger TP +20% (and TP +10% if not already hit)."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 100.0, 90.0, 150.0)
        pos = tracker._positions["BTC-USD"]

        # Jump straight to +20% -- should trigger both TP levels
        pos.update(120.0)
        actions = pos.check_profit_taking()
        tp_actions = [a for a in actions if a["action"] == "partial_sell"]
        assert len(tp_actions) == 2
        assert pos.tp_10_hit is True
        assert pos.tp_20_hit is True

    def test_trailing_stop_activation_after_tp(self):
        """Trailing stop should activate after first TP hit."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "BUY", 1000.0, 100.0, 90.0, 150.0)
        pos = tracker._positions["BTC-USD"]

        # Trigger TP at +10%
        pos.update(110.0)
        actions = pos.check_profit_taking()
        assert pos.tp_10_hit is True
        # Trailing stop should now be set (5% below peak of 110)
        assert pos.trailing_stop == pytest.approx(110.0 * 0.95)

        # Price rises further -- trailing stop should move up
        pos.update(120.0)
        actions = pos.check_profit_taking()
        assert pos.peak_price == 120.0
        assert pos.trailing_stop == pytest.approx(120.0 * 0.95)

        # Price drops but not to trailing stop -- no trigger
        pos.update(115.0)
        actions = pos.check_profit_taking()
        trail_actions = [a for a in actions if a["action"] == "trailing_stop"]
        assert len(trail_actions) == 0
        # Trailing stop should still be based on peak of 120
        assert pos.trailing_stop == pytest.approx(120.0 * 0.95)

        # Price drops to trailing stop level
        pos.update(113.0)  # 120 * 0.95 = 114.0, so 113 < 114
        actions = pos.check_profit_taking()
        trail_actions = [a for a in actions if a["action"] == "trailing_stop"]
        assert len(trail_actions) == 1

    def test_short_profit_taking(self):
        """Profit taking should work for short positions."""
        tracker = PositionTracker()
        tracker.open("BTC-USD", "SELL", 1000.0, 100.0, 110.0, 80.0)
        pos = tracker._positions["BTC-USD"]

        # Price drops 10% (gain for short)
        pos.update(90.0)
        actions = pos.check_profit_taking()
        tp_actions = [a for a in actions if a["action"] == "partial_sell"]
        assert len(tp_actions) == 1
        assert pos.tp_10_hit is True

        # Trailing stop should activate for short (5% above peak low)
        assert pos.trailing_stop == pytest.approx(90.0 * 1.05)

    def test_no_action_at_zero_prices(self):
        """No actions when current_price or entry_price is 0."""
        pos = Position(
            symbol="BTC-USD", side="BUY", size_usd=1000.0,
            entry_price=0.0, stop_price=90.0, take_profit=150.0,
        )
        assert pos.check_profit_taking() == []

        pos2 = Position(
            symbol="BTC-USD", side="BUY", size_usd=1000.0,
            entry_price=100.0, stop_price=90.0, take_profit=150.0,
            current_price=0.0,
        )
        assert pos2.check_profit_taking() == []


class TestPositionPersistence:
    """Tests for save_state / load_state round-tripping."""

    def test_save_and_load_basic(self):
        """Positions saved to disk should be fully restored on load."""
        tracker = PositionTracker()
        tracker.open("BTC-USD:bb_grid:long", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.open("ETH-USD:rsi_mr:short", "SELL", 500.0, 3000.0, 3200.0, 2500.0)
        tracker.update_price("BTC-USD:bb_grid:long", 51000.0)

        # Load into a fresh tracker
        tracker2 = PositionTracker()
        tracker2.load_state()
        positions = tracker2.open_positions()
        assert len(positions) == 2
        syms = {p["symbol"] for p in positions}
        assert "BTC-USD:bb_grid:long" in syms
        assert "ETH-USD:rsi_mr:short" in syms

        # Check that numeric fields round-tripped correctly
        btc = next(p for p in positions if "BTC" in p["symbol"])
        assert btc["entry_price"] == 50000.0
        assert btc["size_usd"] == 1000.0

    def test_save_and_load_preserves_tp_fields(self):
        """TP tracking fields (tp_10_hit, trailing_stop, peak_price) persist."""
        tracker = PositionTracker()
        tracker.open("BTC-USD:bb_grid:long", "BUY", 1000.0, 100.0, 90.0, 150.0)
        pos = tracker._positions["BTC-USD:bb_grid:long"]
        pos.update(112.0)
        pos.check_profit_taking()  # triggers tp_10_hit
        assert pos.tp_10_hit is True
        tracker.save_state()

        tracker2 = PositionTracker()
        tracker2.load_state()
        pos2 = tracker2._positions["BTC-USD:bb_grid:long"]
        assert pos2.tp_10_hit is True
        assert pos2.peak_price == pytest.approx(112.0)
        assert pos2.trailing_stop == pytest.approx(112.0 * 0.95)
        assert pos2.original_size_usd == 1000.0

    def test_close_persists_and_clears(self):
        """Closing a position removes it from saved state and adds to closed list."""
        tracker = PositionTracker()
        tracker.open("BTC-USD:bb_grid:long", "BUY", 1000.0, 50000.0, 48000.0, 55000.0)
        tracker.close("BTC-USD:bb_grid:long", 51000.0)

        tracker2 = PositionTracker()
        tracker2.load_state()
        assert len(tracker2.open_positions()) == 0
        assert len(tracker2.closed_trades()) == 1

    def test_load_missing_file_is_noop(self, tmp_path, monkeypatch):
        """Loading when file doesn't exist should silently do nothing."""
        monkeypatch.setattr(positions_module, "POSITIONS_FILE", tmp_path / "nope.json")
        tracker = PositionTracker()
        tracker.load_state()  # should not raise
        assert len(tracker.open_positions()) == 0

    def test_load_corrupt_file_is_noop(self, tmp_path, monkeypatch):
        """Loading a corrupt JSON file should warn but not crash."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json!!!")
        monkeypatch.setattr(positions_module, "POSITIONS_FILE", bad_file)
        tracker = PositionTracker()
        tracker.load_state()  # should not raise
        assert len(tracker.open_positions()) == 0

    def test_closed_list_capped_at_50(self):
        """Saved state should keep at most 50 closed trades."""
        tracker = PositionTracker(max_concurrent=100)
        for i in range(60):
            key = f"TOK{i}-USD:test:long"
            tracker.open(key, "BUY", 100.0, 10.0, 9.0, 12.0)
        for i in range(60):
            key = f"TOK{i}-USD:test:long"
            tracker.close(key, 11.0)

        # Read the saved file directly
        data = json.loads(positions_module.POSITIONS_FILE.read_text())
        assert len(data["closed"]) == 50

    def test_save_state_creates_parent_dirs(self, tmp_path, monkeypatch):
        """save_state should create parent directories if they don't exist."""
        deep_file = tmp_path / "a" / "b" / "positions.json"
        monkeypatch.setattr(positions_module, "POSITIONS_FILE", deep_file)
        tracker = PositionTracker()
        tracker.open("BTC-USD:test:long", "BUY", 500.0, 40000.0, 38000.0, 45000.0)
        assert deep_file.exists()
