# strategy/strategies/sma_crossover.py
import pandas as pd
from strategy.base import BaseStrategy, Signal


class SMACrossover(BaseStrategy):
    """SMA 20/50 crossover strategy with ATR-based stops."""
    name = "sma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        if len(df) < self.slow + 2:
            return []
        if "sma_20" not in df.columns or "sma_50" not in df.columns:
            return []
        if "atr" not in df.columns:
            return []

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(last["sma_20"]) or pd.isna(last["sma_50"]):
            return []
        if pd.isna(prev["sma_20"]) or pd.isna(prev["sma_50"]):
            return []

        atr = last["atr"] if pd.notna(last["atr"]) else 0
        if atr <= 0:
            return []

        signals = []

        # Golden cross: fast SMA crosses above slow SMA
        if prev["sma_20"] <= prev["sma_50"] and last["sma_20"] > last["sma_50"]:
            signals.append(Signal(
                symbol="",  # filled by caller
                direction="BUY",
                strength=min(1.0, abs(last["sma_20"] - last["sma_50"]) / atr * 0.5),
                stop_price=last["close"] - atr * 2,
                take_profit=last["close"] + atr * 3,
                metadata={"reason": "SMA golden cross", "atr": float(atr)},
            ))

        # Death cross: fast SMA crosses below slow SMA
        elif prev["sma_20"] >= prev["sma_50"] and last["sma_20"] < last["sma_50"]:
            signals.append(Signal(
                symbol="",
                direction="SELL",
                strength=min(1.0, abs(last["sma_50"] - last["sma_20"]) / atr * 0.5),
                stop_price=last["close"] + atr * 2,
                take_profit=last["close"] - atr * 3,
                metadata={"reason": "SMA death cross", "atr": float(atr)},
            ))

        return signals
