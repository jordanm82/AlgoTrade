# tests/test_mm_vpin.py
"""Tests for VPIN computation and kill switch logic."""
import pytest


class TestComputeSpotVpin:
    def test_balanced_flow_low_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": 0.05, "buy_ratio": 0.51}
        assert compute_spot_vpin(flow) == pytest.approx(0.05)

    def test_heavy_buying_high_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": 0.7, "buy_ratio": 0.8}
        assert compute_spot_vpin(flow) == pytest.approx(0.7)

    def test_heavy_selling_high_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": -0.6, "buy_ratio": 0.2}
        assert compute_spot_vpin(flow) == pytest.approx(0.6)

    def test_empty_flow_returns_zero(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        assert compute_spot_vpin({}) == 0.0


class TestComputeKalshiObHeuristic:
    def test_no_change_returns_zero(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        assert compute_kalshi_ob_heuristic(100, 100, 100, 100) == 0.0

    def test_one_sided_yes_increase(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        result = compute_kalshi_ob_heuristic(100, 200, 50, 50)
        assert result > 0.5

    def test_symmetric_change(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        result = compute_kalshi_ob_heuristic(100, 150, 100, 150)
        assert result == pytest.approx(0.0)


class TestBlendedVpin:
    def test_blending_weights(self):
        from kalshi_mm.mm_vpin import compute_blended_vpin
        assert compute_blended_vpin(0.4, 0.2) == pytest.approx(0.34)


class TestKillSwitch:
    def test_safe_vpin(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        assert ks.should_go_dark(0.2) is False

    def test_toxic_vpin(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        # VPIN_CAUTION=0.8
        assert ks.should_go_dark(0.85) is True

    def test_volatility_spike(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_price(100.0)
        ks.record_price(100.6)
        assert ks.volatility_spike() is True

    def test_no_volatility_spike(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_price(100.0)
        ks.record_price(100.1)
        assert ks.volatility_spike() is False

    def test_rising_vpin_detection(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_vpin(0.2)
        ks.record_vpin(0.3)
        ks.record_vpin(0.4)
        assert ks.vpin_rising() is True

    def test_stable_vpin_not_rising(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_vpin(0.3)
        ks.record_vpin(0.3)
        ks.record_vpin(0.3)
        assert ks.vpin_rising() is False

    def test_spread_state_safe(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        assert ks.get_spread_state(0.1) == "SAFE"

    def test_spread_state_caution(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        # VPIN_SAFE=0.6, VPIN_CAUTION=0.8
        assert ks.get_spread_state(0.7) == "CAUTION"

    def test_spread_state_toxic(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        assert ks.get_spread_state(0.9) == "TOXIC"
