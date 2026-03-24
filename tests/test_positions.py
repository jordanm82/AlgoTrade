import pytest
from exchange.positions import Position, PositionTracker


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
