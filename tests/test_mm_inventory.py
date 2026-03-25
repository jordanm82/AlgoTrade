# tests/test_mm_inventory.py
"""Tests for MM inventory: fee math, P&L, position sizing."""
import math
import pytest


class TestCalcMakerFeeCents:
    def test_1_contract_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(1, 50) == 1

    def test_5_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(5, 50) == 3

    def test_10_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 50) == 5

    def test_20_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(20, 50) == 9

    def test_10_contracts_at_30c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 30) == 4

    def test_at_extreme_low_price(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 5) == 1

    def test_at_extreme_high_price(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 95) == 1


class TestCalcRoundTripPnl:
    def test_profit_at_3c_spread_10_contracts_50c(self):
        from kalshi_mm.mm_inventory import calc_round_trip_pnl
        pnl = calc_round_trip_pnl(buy_cents=49, sell_cents=52, contracts=10)
        assert pnl == 20

    def test_loss_when_forced_exit(self):
        from kalshi_mm.mm_inventory import calc_round_trip_pnl
        pnl = calc_round_trip_pnl(buy_cents=50, sell_cents=47, contracts=10)
        assert pnl < 0


class TestComputeContracts:
    def test_normal_balance(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $100 = 10000c, 10% = 1000c, half for bid = 500c, at 50c = 10 contracts
        assert compute_contracts(10000, 50) == 10

    def test_large_balance_capped_at_max(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $1000 = 100000c, 10% = 10000c, half = 5000c, at 50c = 100, capped at 50
        assert compute_contracts(100000, 50) == 50

    def test_small_balance_still_works(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $10 = 1000c, 10% = 100c, half = 50c, at 50c = 1 contract
        assert compute_contracts(1000, 50) == 1

    def test_too_small_returns_none(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $5 = 500c, 10% = 50c, half = 25c, at 50c = 0 contracts
        assert compute_contracts(500, 50) is None

    def test_compounds_with_balance(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # As balance grows, contracts grow (half of 10% budget used for bid)
        assert compute_contracts(5000, 50) == 5    # $50 -> 5 contracts
        assert compute_contracts(10000, 50) == 10   # $100 -> 10 contracts
        assert compute_contracts(20000, 50) == 20   # $200 -> 20 contracts
