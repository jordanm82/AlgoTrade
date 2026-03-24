import pytest
from risk.manager import RiskManager


class TestRiskManager:
    def test_allows_valid_entry(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.check_entry(size_usd=500.0, leverage=1, current_positions=0)
        assert ok is True

    def test_rejects_oversized_position(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.check_entry(size_usd=1500.0, leverage=1, current_positions=0)
        assert ok is False
        assert "position size" in reason.lower()

    def test_rejects_too_many_concurrent(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.check_entry(size_usd=500.0, leverage=1, current_positions=3)
        assert ok is False
        assert "concurrent" in reason.lower()

    def test_rejects_excessive_leverage(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.check_entry(size_usd=500.0, leverage=10, current_positions=0)
        assert ok is False
        assert "leverage" in reason.lower()

    def test_rejects_below_min_balance(self):
        rm = RiskManager(portfolio_value=50.0)
        ok, reason = rm.check_entry(size_usd=10.0, leverage=1, current_positions=0)
        assert ok is False
        assert "minimum balance" in reason.lower()

    def test_validates_stop_distance(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.validate_stop(entry=50000.0, stop=49000.0, atr=500.0, side="BUY")
        assert ok is True  # (50000-49000)/500 = 2.0 ATR, within 1-3 range

    def test_rejects_stop_too_tight(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.validate_stop(entry=50000.0, stop=49900.0, atr=500.0, side="BUY")
        assert ok is False
        assert "tight" in reason.lower()

    def test_rejects_stop_too_wide(self):
        rm = RiskManager(portfolio_value=10000.0)
        ok, reason = rm.validate_stop(entry=50000.0, stop=47000.0, atr=500.0, side="BUY")
        assert ok is False
        assert "wide" in reason.lower()

    def test_drawdown_halt(self):
        rm = RiskManager(portfolio_value=10000.0)
        rm.record_daily_start(10000.0)
        assert rm.is_halted(current_value=9400.0) is True  # 6% drawdown > 5% limit

    def test_no_halt_within_limit(self):
        rm = RiskManager(portfolio_value=10000.0)
        rm.record_daily_start(10000.0)
        assert rm.is_halted(current_value=9600.0) is False  # 4% drawdown < 5% limit
