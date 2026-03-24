import pytest
from exchange.positions import PositionTracker


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
