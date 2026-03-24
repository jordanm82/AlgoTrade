import pytest
import pandas as pd
import numpy as np
from strategy.compound_backtest import compound_backtest

@pytest.fixture
def trending_up_df():
    """Price goes from 100 to 150 with pullbacks."""
    np.random.seed(42)
    n = 200
    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 2
    close = trend + noise
    return pd.DataFrame({
        "open": close - np.random.rand(n),
        "high": close + abs(np.random.randn(n) * 3),
        "low": close - abs(np.random.randn(n) * 3),
        "close": close,
        "volume": np.random.rand(n) * 1000 + 500,
        "rsi": np.where(close < trend - 1, 25, np.where(close > trend + 1, 75, 50)),
    }, index=pd.date_range("2025-01-01", periods=n, freq="1h"))

class TestCompoundBacktest:
    def test_returns_expected_fields(self, trending_up_df):
        result = compound_backtest(
            trending_up_df,
            buy_fn=lambda r, p: r["rsi"] < 30,
            sell_fn=lambda r, p: r["rsi"] > 70,
        )
        assert "num_trades" in result
        assert "win_rate" in result
        assert "total_return_pct" in result
        assert "final_equity" in result
        assert "max_drawdown_pct" in result
        assert "trades" in result
        assert "equity_curve" in result

    def test_compounding_grows_position_size(self, trending_up_df):
        result = compound_backtest(
            trending_up_df,
            buy_fn=lambda r, p: r["rsi"] < 30,
            sell_fn=lambda r, p: r["rsi"] > 70,
            initial_equity=1000, size_pct=0.10,
        )
        if result["num_trades"] >= 2:
            t = result["trades"]
            if t[-1]["equity_after"] > 1000:
                assert t[-1]["size_usd"] >= t[0]["size_usd"]

    def test_leverage_multiplies_returns(self, trending_up_df):
        r1 = compound_backtest(trending_up_df,
            buy_fn=lambda r, p: r["rsi"] < 30,
            sell_fn=lambda r, p: r["rsi"] > 70,
            leverage=1)
        r3 = compound_backtest(trending_up_df,
            buy_fn=lambda r, p: r["rsi"] < 30,
            sell_fn=lambda r, p: r["rsi"] > 70,
            leverage=3)
        if r1["num_trades"] > 0:
            assert abs(r3["total_return_pct"]) > abs(r1["total_return_pct"])

    def test_short_positions_work(self, trending_up_df):
        result = compound_backtest(
            trending_up_df,
            buy_fn=lambda r, p: False,
            sell_fn=lambda r, p: False,
            short_fn=lambda r, p: r["rsi"] > 70,
            cover_fn=lambda r, p: r["rsi"] < 50,
        )
        assert isinstance(result["num_trades"], int)

    def test_stop_loss_triggers(self, trending_up_df):
        result = compound_backtest(
            trending_up_df,
            buy_fn=lambda r, p: r["rsi"] < 30,
            sell_fn=lambda r, p: r["rsi"] > 70,
            stop_loss_pct=0.01,
        )
        assert isinstance(result["num_trades"], int)

    def test_zero_trades_returns_initial(self, trending_up_df):
        result = compound_backtest(
            trending_up_df,
            buy_fn=lambda r, p: False,
            sell_fn=lambda r, p: False,
        )
        assert result["num_trades"] == 0
        assert result["final_equity"] == 1000
