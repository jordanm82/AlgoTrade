# strategy/strategies/kalshi_predictor.py
"""Multi-signal confidence scorer for Kalshi crypto prediction markets.

Combines RSI, Bollinger Bands, MACD, volume, momentum, and multi-timeframe
signals into a single confidence score (0-100). Only bets when confidence
exceeds a configurable threshold.

Supported assets: BTC, ETH, SOL, XRP (any with Kalshi contracts).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class KalshiSignal:
    asset: str              # "BTC", "ETH", etc.
    direction: str          # "UP" or "DOWN"
    confidence: float       # 0-100
    components: dict        # individual signal scores
    price: float
    rsi: float


class KalshiPredictor:
    """Scores confidence for short-term crypto direction predictions."""

    def score(self, df: pd.DataFrame) -> KalshiSignal | None:
        """Score confidence for the next 15-minute direction.

        Returns a KalshiSignal if any directional confidence exists,
        or None if the market is neutral.

        The confidence score (0-100) is built from these components:

        1. RSI Signal (0-30 points):
           - RSI < 25: +30 (strong oversold -> UP)
           - RSI < 30: +20
           - RSI < 35: +10
           - RSI > 75: +30 (strong overbought -> DOWN)
           - RSI > 70: +20
           - RSI > 65: +10

        2. Bollinger Band Signal (0-20 points):
           - Price below BB lower: +20 (UP)
           - Price within 0.5% of BB lower: +10
           - Price above BB upper: +20 (DOWN)
           - Price within 0.5% of BB upper: +10

        3. MACD Momentum (0-15 points):
           - MACD histogram positive AND increasing: +15 (UP)
           - MACD histogram positive: +8
           - MACD histogram negative AND decreasing: +15 (DOWN)
           - MACD histogram negative: +8

        4. Volume Confirmation (0-10 points):
           - Volume > 2x average: +10 (confirms direction)
           - Volume > 1.5x average: +5

        5. Price Momentum (0-15 points):
           - Last 3 candles all green: +15 (UP momentum)
           - Last 2 candles green: +8
           - Last 3 candles all red: +15 (DOWN momentum)
           - Last 2 candles red: +8

        6. Multi-Candle RSI Trend (0-10 points):
           - RSI increasing for 3+ candles from oversold: +10 (UP -- early bounce)
           - RSI decreasing for 3+ candles from overbought: +10 (DOWN -- early drop)
        """
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        prev2 = df.iloc[-3] if len(df) > 2 else prev

        close = float(last["close"])
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50

        up_score = 0
        down_score = 0
        components = {}

        # 1. RSI Signal (0-30)
        rsi_up = 0
        rsi_down = 0
        if rsi < 25:
            rsi_up = 30
        elif rsi < 30:
            rsi_up = 20
        elif rsi < 35:
            rsi_up = 10
        elif rsi > 75:
            rsi_down = 30
        elif rsi > 70:
            rsi_down = 20
        elif rsi > 65:
            rsi_down = 10
        up_score += rsi_up
        down_score += rsi_down
        components["rsi"] = {"up": rsi_up, "down": rsi_down, "value": rsi}

        # 2. Bollinger Band Signal (0-20)
        bb_up = 0
        bb_down = 0
        bb_lower = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
        bb_upper = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0
        bb_mid = float(last.get("bb_middle", 0)) if pd.notna(last.get("bb_middle")) else 0
        if bb_lower > 0:
            if close < bb_lower:
                bb_up = 20
            elif close < bb_lower * 1.005:
                bb_up = 10
            if close > bb_upper:
                bb_down = 20
            elif close > bb_upper * 0.995:
                bb_down = 10
        up_score += bb_up
        down_score += bb_down
        components["bb"] = {"up": bb_up, "down": bb_down}

        # 3. MACD Momentum (0-15)
        macd_up = 0
        macd_down = 0
        macd_hist = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        prev_hist = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
        if macd_hist > 0 and macd_hist > prev_hist:
            macd_up = 15
        elif macd_hist > 0:
            macd_up = 8
        elif macd_hist < 0 and macd_hist < prev_hist:
            macd_down = 15
        elif macd_hist < 0:
            macd_down = 8
        up_score += macd_up
        down_score += macd_down
        components["macd"] = {"up": macd_up, "down": macd_down}

        # 4. Volume Confirmation (0-10)
        vol_score = 0
        vol = float(last.get("volume", 0))
        vol_sma = float(last.get("vol_sma_20", 0)) if pd.notna(last.get("vol_sma_20")) else 0
        if vol_sma > 0:
            if vol > vol_sma * 2:
                vol_score = 10
            elif vol > vol_sma * 1.5:
                vol_score = 5
        # Volume confirms the dominant direction
        if up_score > down_score:
            up_score += vol_score
        else:
            down_score += vol_score
        components["volume"] = {"score": vol_score, "ratio": vol / vol_sma if vol_sma > 0 else 0}

        # 5. Price Momentum -- last 3 candles (0-15)
        mom_up = 0
        mom_down = 0
        if len(df) >= 4:
            c1 = float(df.iloc[-1]["close"])
            c2 = float(df.iloc[-2]["close"])
            c3 = float(df.iloc[-3]["close"])
            c4 = float(df.iloc[-4]["close"])
            if c1 > c2 > c3:
                mom_up = 15
            elif c1 > c2:
                mom_up = 8
            if c1 < c2 < c3:
                mom_down = 15
            elif c1 < c2:
                mom_down = 8
        up_score += mom_up
        down_score += mom_down
        components["momentum"] = {"up": mom_up, "down": mom_down}

        # 6. RSI Trend (0-10) -- is RSI recovering from extreme?
        rsi_trend_up = 0
        rsi_trend_down = 0
        if len(df) >= 4:
            rsis = [
                float(df.iloc[i].get("rsi", 50)) if pd.notna(df.iloc[i].get("rsi")) else 50
                for i in range(-3, 0)
            ]
            if rsis[-1] > rsis[-2] > rsis[-3] and rsis[-3] < 40:
                rsi_trend_up = 10  # RSI recovering from oversold
            if rsis[-1] < rsis[-2] < rsis[-3] and rsis[-3] > 60:
                rsi_trend_down = 10  # RSI declining from overbought
        up_score += rsi_trend_up
        down_score += rsi_trend_down
        components["rsi_trend"] = {"up": rsi_trend_up, "down": rsi_trend_down}

        # Determine direction and confidence
        if up_score > down_score and up_score > 0:
            return KalshiSignal(
                asset="", direction="UP", confidence=min(100, up_score),
                components=components, price=close, rsi=rsi,
            )
        elif down_score > up_score and down_score > 0:
            return KalshiSignal(
                asset="", direction="DOWN", confidence=min(100, down_score),
                components=components, price=close, rsi=rsi,
            )
        return None
