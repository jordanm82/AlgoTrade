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
