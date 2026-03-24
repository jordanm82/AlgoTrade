# tests/test_funding_arb.py
import pytest
from strategy.strategies.funding_arb import FundingArb

class TestFundingArb:
    def test_high_positive_funding_triggers_long_spot(self):
        arb = FundingArb()
        signals = arb.check_funding(funding_rate=0.0005, price=50000, atr=500)
        assert len(signals) == 1
        assert signals[0].direction == "BUY"
        assert "long_spot_short_perp" in signals[0].metadata["arb_type"]

    def test_deeply_negative_funding_triggers_short_spot(self):
        arb = FundingArb()
        signals = arb.check_funding(funding_rate=-0.0003, price=50000, atr=500)
        assert len(signals) == 1
        assert signals[0].direction == "SELL"
        assert "short_spot_long_perp" in signals[0].metadata["arb_type"]

    def test_normal_funding_no_signal(self):
        arb = FundingArb()
        signals = arb.check_funding(funding_rate=0.0001, price=50000, atr=500)
        assert len(signals) == 0

    def test_strength_scales_with_rate(self):
        arb = FundingArb()
        low = arb.check_funding(funding_rate=0.0004, price=50000, atr=500)
        high = arb.check_funding(funding_rate=0.001, price=50000, atr=500)
        assert high[0].strength > low[0].strength

    def test_metadata_includes_annualized(self):
        arb = FundingArb()
        signals = arb.check_funding(funding_rate=0.0005, price=50000, atr=500)
        assert "annualized_pct" in signals[0].metadata
        assert signals[0].metadata["annualized_pct"] > 0
