# tests/test_strategy.py
import pytest
import pandas as pd
from strategy.base import BaseStrategy, Signal
from strategy.backtest import run_backtest

class DummyStrategy(BaseStrategy):
    """Simple strategy for testing: buy when RSI < 30, sell when RSI > 70."""
    name = "dummy"

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        results = []
        if "rsi" not in df.columns:
            return results
        last = df.iloc[-1]
        if last["rsi"] < 30:
            results.append(Signal(
                symbol="BTC-USD", direction="BUY", strength=0.8,
                stop_price=last["close"] - last.get("atr", 100) * 2,
                take_profit=last["close"] + last.get("atr", 100) * 3,
                metadata={"reason": "RSI oversold"},
            ))
        elif last["rsi"] > 70:
            results.append(Signal(
                symbol="BTC-USD", direction="SELL", strength=0.7,
                stop_price=last["close"] + last.get("atr", 100) * 2,
                take_profit=last["close"] - last.get("atr", 100) * 3,
                metadata={"reason": "RSI overbought"},
            ))
        return results

class TestBaseStrategy:
    def test_signal_dataclass(self):
        s = Signal(symbol="BTC-USD", direction="BUY", strength=0.8,
                   stop_price=48000.0, take_profit=55000.0, metadata={})
        assert s.symbol == "BTC-USD"
        assert 0 <= s.strength <= 1

    def test_strategy_returns_signals(self, sample_ohlcv):
        from data.indicators import add_indicators
        df = add_indicators(sample_ohlcv)
        strat = DummyStrategy()
        signals = strat.signals(df)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, Signal)
            assert s.direction in ("BUY", "SELL")

class TestBacktest:
    def test_run_backtest_returns_results(self, sample_ohlcv):
        from data.indicators import add_indicators
        df = add_indicators(sample_ohlcv)
        results = run_backtest(
            df=df,
            buy_condition=lambda row: row["rsi"] < 30 if pd.notna(row["rsi"]) else False,
            sell_condition=lambda row: row["rsi"] > 70 if pd.notna(row["rsi"]) else False,
            cash=10000,
            commission=0.001,
        )
        assert "equity_final" in results
        assert "max_drawdown" in results
        assert "num_trades" in results
        assert "sharpe" in results
        assert "win_rate" in results
