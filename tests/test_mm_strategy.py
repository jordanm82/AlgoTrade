# tests/test_mm_strategy.py
"""Tests for MM strategy: orderbook parsing, mid calc, spread sizing, quotes."""
import pytest


class TestComputeMidCents:
    def test_basic_mid(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.4800", "10"]], "no_dollars": [["0.5000", "5"]]}}
        assert compute_mid_cents(ob) == 49

    def test_asymmetric_book(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.4000", "20"]], "no_dollars": [["0.5500", "15"]]}}
        assert compute_mid_cents(ob) == 42

    def test_empty_yes_returns_none(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.50", "5"]]}}
        assert compute_mid_cents(ob) is None

    def test_empty_no_returns_none(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.48", "10"]], "no_dollars": []}}
        assert compute_mid_cents(ob) is None

    def test_round_not_truncate_29c(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.2900", "10"]], "no_dollars": [["0.7100", "10"]]}}
        assert compute_mid_cents(ob) == 29

    def test_round_not_truncate_57c(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.5700", "10"]], "no_dollars": [["0.4300", "10"]]}}
        assert compute_mid_cents(ob) == 57

    def test_multi_level_book_uses_best_bid(self):
        """Kalshi sorts ascending — best (highest) bid is last element."""
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {
            "yes_dollars": [["0.0100", "100"], ["0.5000", "50"], ["0.9200", "25"]],
            "no_dollars": [["0.0100", "200"], ["0.0300", "100"], ["0.0600", "50"]],
        }}
        # best_yes_bid=92, best_no_bid=6, yes_ask=100-6=94, mid=(92+94)//2=93
        assert compute_mid_cents(ob) == 93


class TestComputeSpreadCents:
    def test_safe_vpin(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        # VPIN_SAFE=0.6, below it -> 2c spread
        assert compute_spread_cents(0.3) == 2

    def test_caution_low(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        # At VPIN_SAFE boundary -> 3c
        assert compute_spread_cents(0.6) == 3

    def test_caution_high(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        # Near VPIN_CAUTION=0.8 -> 4c
        assert compute_spread_cents(0.79) == 4

    def test_toxic_returns_none(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        # At/above VPIN_CAUTION=0.8 -> go dark
        assert compute_spread_cents(0.85) is None


class TestComputeBidCents:
    def test_normal_bid(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(50, 2) == 49

    def test_wide_spread(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(50, 4) == 48

    def test_boundary_low_returns_none(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(1, 4) is None


class TestComputeAskCents:
    def test_normal_ask(self):
        from kalshi_mm.mm_strategy import compute_ask_cents
        assert compute_ask_cents(48, 3) == 51

    def test_boundary_high_returns_none(self):
        from kalshi_mm.mm_strategy import compute_ask_cents
        assert compute_ask_cents(97, 4) is None


class TestParseOrderbookVolume:
    def test_sums_volumes(self):
        from kalshi_mm.mm_strategy import parse_ob_total_volume
        levels = [["0.48", "10"], ["0.47", "20"], ["0.46", "5"]]
        assert parse_ob_total_volume(levels) == 35


class TestParseObAsDict:
    def test_converts_to_cents_dict(self):
        from kalshi_mm.mm_strategy import parse_ob_as_dict
        levels = [["0.4800", "10.00"], ["0.5000", "20.00"]]
        assert parse_ob_as_dict(levels) == {48: 10, 50: 20}


class TestVolumeConsumed:
    def test_volume_consumed_at_or_above(self):
        from kalshi_mm.mm_strategy import volume_consumed_at_or_above
        prev = {48: 100, 50: 50, 52: 30}
        curr = {48: 100, 50: 30, 52: 10}  # 20 consumed at 50, 20 at 52
        assert volume_consumed_at_or_above(prev, curr, 50) == 40

    def test_no_consumption(self):
        from kalshi_mm.mm_strategy import volume_consumed_at_or_above
        prev = {48: 100, 50: 50}
        curr = {48: 100, 50: 60}  # volume increased, not consumed
        assert volume_consumed_at_or_above(prev, curr, 48) == 0

    def test_level_disappeared(self):
        from kalshi_mm.mm_strategy import volume_consumed_at_or_above
        prev = {50: 30, 52: 20}
        curr = {50: 30}  # 52 level gone = 20 consumed
        assert volume_consumed_at_or_above(prev, curr, 50) == 20

    def test_below_threshold_ignored(self):
        from kalshi_mm.mm_strategy import volume_consumed_at_or_above
        prev = {40: 100, 50: 50}
        curr = {40: 10, 50: 50}  # 90 consumed at 40, but below threshold 45
        assert volume_consumed_at_or_above(prev, curr, 45) == 0

    def test_volume_consumed_at_or_below(self):
        from kalshi_mm.mm_strategy import volume_consumed_at_or_below
        prev = {48: 100, 50: 50, 52: 30}
        curr = {48: 80, 50: 30}  # 20 at 48, 20 at 50, 52 not checked (above 50)
        assert volume_consumed_at_or_below(prev, curr, 50) == 40
