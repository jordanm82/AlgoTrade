# tests/test_grid_bot.py
import pytest
import pandas as pd
import numpy as np
from strategy.strategies.grid_bot import GridBot
from strategy.base import Signal


def _make_df(prices: list[float]):
    """Build a minimal DataFrame from a list of close prices."""
    n = len(prices)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 10 for p in prices],
        "low": [p - 10 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestGridBot:
    def test_grid_spacing_correct(self):
        """Grid levels should be evenly spaced by grid_pct."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        bot.set_reference_price(10000.0)
        grids = bot.get_grid_levels()
        assert len(grids["buy_levels"]) == 5
        assert len(grids["sell_levels"]) == 5
        # First buy level = ref * (1 - 0.01)
        assert grids["buy_levels"][0] == pytest.approx(10000 * 0.99)
        assert grids["buy_levels"][1] == pytest.approx(10000 * 0.98)
        # First sell level = ref * (1 + 0.01)
        assert grids["sell_levels"][0] == pytest.approx(10000 * 1.01)
        assert grids["sell_levels"][1] == pytest.approx(10000 * 1.02)

    def test_buy_at_lower_grid(self):
        """Price crossing down through a buy level should produce BUY signal."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        ref = 10000.0
        bot.set_reference_price(ref)
        # Price goes from above first buy level to below it
        buy_level = ref * 0.99  # 9900
        prices = [ref, buy_level - 1]  # 10000 -> 9899
        df = _make_df(prices)
        signals = bot.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "BUY"
        assert "Grid buy level 1" in signals[0].metadata["reason"]

    def test_no_signal_at_reference_price(self):
        """Price staying at reference should produce no signals."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        bot.set_reference_price(10000.0)
        prices = [10000.0, 10000.0]
        df = _make_df(prices)
        signals = bot.signals(df)
        assert signals == []

    def test_sell_at_upper_grid(self):
        """Price crossing up through a sell level should produce SELL signal."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        ref = 10000.0
        bot.set_reference_price(ref)
        sell_level = ref * 1.01  # 10100
        prices = [ref, sell_level + 1]  # 10000 -> 10101
        df = _make_df(prices)
        signals = bot.signals(df)
        assert len(signals) == 1
        assert signals[0].direction == "SELL"
        assert "Grid sell level 1" in signals[0].metadata["reason"]

    def test_no_duplicate_fill(self):
        """Same grid level should not trigger twice."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        ref = 10000.0
        bot.set_reference_price(ref)
        buy_level = ref * 0.99
        # First cross
        df1 = _make_df([ref, buy_level - 1])
        signals1 = bot.signals(df1)
        assert len(signals1) == 1
        # Second cross at same level
        df2 = _make_df([ref, buy_level - 1])
        signals2 = bot.signals(df2)
        assert signals2 == []

    def test_no_reference_price_returns_empty(self):
        """Without setting reference price, no signals produced."""
        bot = GridBot()
        df = _make_df([10000.0, 9900.0])
        signals = bot.signals(df)
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        """Fewer than 2 rows should return empty."""
        bot = GridBot()
        bot.set_reference_price(10000.0)
        df = _make_df([10000.0])
        signals = bot.signals(df)
        assert signals == []

    def test_signals_have_stop_and_tp(self):
        """Grid signals should include stop_price and take_profit."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        ref = 10000.0
        bot.set_reference_price(ref)
        prices = [ref, ref * 0.99 - 1]
        df = _make_df(prices)
        signals = bot.signals(df)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.stop_price < sig.take_profit  # buy: stop below, TP above

    def test_grid_levels_empty_without_reference(self):
        """get_grid_levels without reference returns empty lists."""
        bot = GridBot()
        grids = bot.get_grid_levels()
        assert grids["buy_levels"] == []
        assert grids["sell_levels"] == []

    def test_strength_between_zero_and_one(self):
        """Signal strength should be between 0 and 1."""
        bot = GridBot(grid_pct=0.01, num_grids=5)
        ref = 10000.0
        bot.set_reference_price(ref)
        # Cross multiple buy levels at once
        prices = [ref, ref * 0.95]  # crosses all 5 buy levels
        df = _make_df(prices)
        signals = bot.signals(df)
        for sig in signals:
            assert 0.0 <= sig.strength <= 1.0
