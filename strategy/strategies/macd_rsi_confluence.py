# strategy/strategies/macd_rsi_confluence.py
"""MACD + RSI Confluence — entries when both indicators agree.
Buy: MACD crosses above signal AND RSI < 40.
Short: MACD crosses below signal AND RSI > 70."""
import pandas as pd
from strategy.base import BaseStrategy, Signal


class MACDRSIConfluence(BaseStrategy):
    name = "macd_rsi_confluence"

    def __init__(self, rsi_buy_threshold: int = 40, rsi_sell_threshold: int = 60,
                 rsi_short_threshold: int = 70, rsi_cover_threshold: int = 30,
                 atr_stop: float = 2.0, atr_tp: float = 3.0):
        self.rsi_buy = rsi_buy_threshold
        self.rsi_sell = rsi_sell_threshold
        self.rsi_short = rsi_short_threshold
        self.rsi_cover = rsi_cover_threshold
        self.atr_stop = atr_stop
        self.atr_tp = atr_tp

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        required = ["rsi", "macd", "macd_signal", "atr"]
        if len(df) < 30 or not all(c in df.columns for c in required):
            return []

        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = last.get("rsi")
        macd = last.get("macd")
        macd_sig = last.get("macd_signal")
        prev_macd = prev.get("macd")
        prev_macd_sig = prev.get("macd_signal")
        atr = last.get("atr")
        close = float(last["close"])

        if any(pd.isna(v) for v in [rsi, macd, macd_sig, prev_macd, prev_macd_sig, atr]):
            return []
        if atr <= 0:
            return []

        signals = []
        macd_crossed_up = prev_macd <= prev_macd_sig and macd > macd_sig
        macd_crossed_down = prev_macd >= prev_macd_sig and macd < macd_sig

        if macd_crossed_up and rsi < self.rsi_buy:
            strength = min(1.0, (self.rsi_buy - rsi) / 30)
            signals.append(Signal(
                symbol="", direction="BUY", strength=strength,
                stop_price=close - atr * self.atr_stop,
                take_profit=close + atr * self.atr_tp,
                metadata={"reason": f"MACD cross up + RSI {rsi:.1f}", "rsi": float(rsi),
                          "leverage": 2 if strength > 0.7 else 1},
            ))

        if macd_crossed_down and rsi > self.rsi_short:
            strength = min(1.0, (rsi - self.rsi_short) / 30)
            signals.append(Signal(
                symbol="", direction="SELL", strength=strength,
                stop_price=close + atr * self.atr_stop,
                take_profit=close - atr * self.atr_tp,
                metadata={"reason": f"MACD cross down + RSI {rsi:.1f}", "rsi": float(rsi),
                          "leverage": 2 if strength > 0.7 else 1},
            ))

        return signals
