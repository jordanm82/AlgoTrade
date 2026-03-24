# strategy/strategies/rsi_mean_reversion.py
"""RSI Mean Reversion — long AND short.
Long: RSI < oversold. Exit: RSI > exit_oversold.
Short: RSI > overbought. Exit: RSI < exit_overbought.
ATR-based stop-losses and take-profits."""
import pandas as pd
from strategy.base import BaseStrategy, Signal


class RSIMeanReversion(BaseStrategy):
    name = "rsi_mean_reversion"

    def __init__(self, oversold: int = 30, overbought: int = 70,
                 exit_oversold: int = 65, exit_overbought: int = 35,
                 atr_stop: float = 2.0, atr_tp: float = 3.0):
        self.oversold = oversold
        self.overbought = overbought
        self.exit_oversold = exit_oversold
        self.exit_overbought = exit_overbought
        self.atr_stop = atr_stop
        self.atr_tp = atr_tp

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        if len(df) < 20:
            return []
        if "rsi" not in df.columns or "atr" not in df.columns:
            return []

        last = df.iloc[-1]
        rsi = last.get("rsi")
        atr = last.get("atr")
        close = float(last["close"])

        if pd.isna(rsi) or pd.isna(atr) or atr <= 0:
            return []

        signals = []

        if rsi < self.oversold:
            strength = min(1.0, (self.oversold - rsi) / 20)
            signals.append(Signal(
                symbol="", direction="BUY", strength=strength,
                stop_price=close - atr * self.atr_stop,
                take_profit=close + atr * self.atr_tp,
                metadata={"reason": f"RSI oversold ({rsi:.1f})", "rsi": float(rsi), "atr": float(atr),
                          "exit_target": self.exit_oversold},
            ))

        if rsi > self.overbought:
            strength = min(1.0, (rsi - self.overbought) / 20)
            signals.append(Signal(
                symbol="", direction="SELL", strength=strength,
                stop_price=close + atr * self.atr_stop,
                take_profit=close - atr * self.atr_tp,
                metadata={"reason": f"RSI overbought ({rsi:.1f})", "rsi": float(rsi), "atr": float(atr),
                          "exit_target": self.exit_overbought},
            ))

        return signals
