# strategy/strategies/kalshi_predictor.py
"""Multi-signal confidence scorer for Kalshi crypto prediction markets.

Combines RSI, Bollinger Bands, MACD, volume, momentum, multi-timeframe
signals, AND leading indicators (order book imbalance, trade flow,
large-trade bias, spread, cross-asset momentum) into a single confidence
score (0-100). Only bets when confidence exceeds a configurable threshold.

Supported assets: BTC, ETH, SOL, XRP (any with Kalshi contracts).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass


# Maximum raw score from lagging signals (components 1-6)
_MAX_LAGGING = 100
# Maximum raw score from leading signals (components 7-11)
_MAX_LEADING = 65
# Combined max before normalization
_MAX_RAW = _MAX_LAGGING + _MAX_LEADING  # 165


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

    def score(self, df: pd.DataFrame, market_data: dict | None = None) -> KalshiSignal | None:
        """Score confidence for the next 15-minute direction.

        Args:
            df: OHLCV DataFrame with indicator columns.
            market_data: Optional dict with leading indicator data from
                ``data.market_data.get_all_signals()``.  Expected keys:
                ``order_book`` and ``trade_flow``.  May also include a
                ``cross_asset`` dict with ``market_direction`` float.

        Returns a KalshiSignal if any directional confidence exists,
        or None if the market is neutral.

        Lagging components (from OHLCV candles):

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

        Leading components (from real-time microstructure):

        7. Order Book Imbalance (0-20 points):
           - imbalance > +0.3: +20 (strong buy pressure -> UP)
           - imbalance > +0.15: +10
           - imbalance < -0.3: +20 (strong sell pressure -> DOWN)
           - imbalance < -0.15: +10

        8. Trade Flow (0-20 points):
           - net_flow > +0.2 AND buy_ratio > 0.55: +20 (aggressive buying -> UP)
           - net_flow > +0.1: +10
           - net_flow < -0.2 AND buy_ratio < 0.45: +20 (aggressive selling -> DOWN)
           - net_flow < -0.1: +10

        9. Large Trade Bias (0-10 points):
           - large_trade_bias > +0.3: +10 (whales buying -> UP)
           - large_trade_bias < -0.3: +10 (whales selling -> DOWN)

        10. Spread Signal (0-5 points):
            - spread_pct > 0.1%: +5 (wide spread = big move coming, confirms direction)

        11. Cross-Asset (0-10 points):
            - BTC down >1% and asset is an alt: +10 DOWN (BTC dragging alts)
            - BTC up >1% and asset is an alt: +10 UP
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

        # 7. Stochastic RSI (0-15)
        stoch_up = 0
        stoch_down = 0
        stochrsi_k = float(last.get("stochrsi_k", 50)) if pd.notna(last.get("stochrsi_k")) else 50
        if stochrsi_k < 10:
            stoch_up = 15
        elif stochrsi_k < 20:
            stoch_up = 8
        elif stochrsi_k > 90:
            stoch_down = 15
        elif stochrsi_k > 80:
            stoch_down = 8
        up_score += stoch_up
        down_score += stoch_down
        components["stochrsi"] = {"up": stoch_up, "down": stoch_down, "value": stochrsi_k}

        # --- Leading indicator components (only when market_data provided) ---
        has_leading = market_data is not None
        ob = (market_data or {}).get("order_book", {})
        tf = (market_data or {}).get("trade_flow", {})
        cross = (market_data or {}).get("cross_asset", {})

        # 7. Order Book Imbalance (0-20)
        ob_up = 0
        ob_down = 0
        imbalance = ob.get("imbalance", 0)
        if imbalance > 0.3:
            ob_up = 20
        elif imbalance > 0.15:
            ob_up = 10
        elif imbalance < -0.3:
            ob_down = 20
        elif imbalance < -0.15:
            ob_down = 10
        up_score += ob_up
        down_score += ob_down
        components["order_book"] = {"up": ob_up, "down": ob_down, "imbalance": imbalance}

        # 8. Trade Flow (0-20)
        tf_up = 0
        tf_down = 0
        net_flow = tf.get("net_flow", 0)
        buy_ratio = tf.get("buy_ratio", 0.5)
        if net_flow > 0.2 and buy_ratio > 0.55:
            tf_up = 20
        elif net_flow > 0.1:
            tf_up = 10
        elif net_flow < -0.2 and buy_ratio < 0.45:
            tf_down = 20
        elif net_flow < -0.1:
            tf_down = 10
        up_score += tf_up
        down_score += tf_down
        components["trade_flow"] = {"up": tf_up, "down": tf_down, "net_flow": net_flow}

        # 9. Large Trade Bias (0-10)
        lt_up = 0
        lt_down = 0
        large_bias = tf.get("large_trade_bias", 0)
        if large_bias > 0.3:
            lt_up = 10
        elif large_bias < -0.3:
            lt_down = 10
        up_score += lt_up
        down_score += lt_down
        components["large_trade"] = {"up": lt_up, "down": lt_down, "bias": large_bias}

        # 10. Spread Signal (0-5) — wide spread confirms the dominant direction
        spread_score = 0
        spread_pct = ob.get("spread_pct", 0)
        if spread_pct > 0.1:
            spread_score = 5
        if up_score > down_score:
            up_score += spread_score
        else:
            down_score += spread_score
        components["spread"] = {"score": spread_score, "spread_pct": spread_pct}

        # 11. Cross-Asset / BTC leader signal (0-10)
        ca_up = 0
        ca_down = 0
        btc_dir = cross.get("market_direction", 0)
        # Only apply cross-asset signal to alts (non-BTC)
        is_alt = True  # caller can set asset; we assume alt unless overridden
        if btc_dir < -1 and is_alt:
            ca_down = 10
        elif btc_dir > 1 and is_alt:
            ca_up = 10
        up_score += ca_up
        down_score += ca_down
        components["cross_asset"] = {"up": ca_up, "down": ca_down, "btc_dir": btc_dir}

        # --- Determine direction and confidence ---
        # When leading indicators are present the raw max is higher (~165),
        # so we normalize to keep the 0-100 scale.
        max_possible = _MAX_RAW if has_leading else _MAX_LAGGING

        if up_score > down_score and up_score > 0:
            confidence = min(100, int(up_score * 100 / max_possible))
            return KalshiSignal(
                asset="", direction="UP", confidence=confidence,
                components=components, price=close, rsi=rsi,
            )
        elif down_score > up_score and down_score > 0:
            confidence = min(100, int(down_score * 100 / max_possible))
            return KalshiSignal(
                asset="", direction="DOWN", confidence=confidence,
                components=components, price=close, rsi=rsi,
            )
        return None
