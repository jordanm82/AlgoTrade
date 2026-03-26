# strategy/strategies/kalshi_predictor_v3.py
"""V3 Kalshi predictor: strike-relative probability model.

Answers the actual Kalshi question: "will price close above or below
this specific strike price?" Uses a pre-computed probability lookup
table calibrated from historical data, adjusted by real-time signals.
"""
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

DISTANCE_BINS = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TIME_BINS = [14, 12, 10, 8, 6, 4, 2]

EDGE_MARGIN = 0.05    # require 5% edge over implied contract price
MAX_BET_PRICE = 50    # never pay more than 50c for either side


@dataclass
class KalshiV3Signal:
    asset: str
    probability: float          # 0.0-1.0 that price closes >= strike
    recommended_side: str       # "YES", "NO", or "SKIP"
    max_price_cents: int        # max we should pay for recommended side
    distance_atr: float         # distance from strike in ATR units
    base_prob: float            # from lookup table before adjustments
    adjustments: dict           # breakdown of each technical adjustment
    current_price: float
    strike_price: float
    minutes_remaining: float


class KalshiPredictorV3:
    """Strike-relative probability predictor for Kalshi 15m contracts."""

    def __init__(self, prob_table_path: str = "data/store/kalshi_prob_table.json"):
        self._prob_table = {}
        try:
            with open(prob_table_path) as f:
                self._prob_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # empty table — will return 0.5 for everything

    def predict(self, df: pd.DataFrame, strike_price: float,
                minutes_remaining: float, market_data: dict | None = None,
                df_1h: pd.DataFrame | None = None,
                current_price: float | None = None) -> KalshiV3Signal | None:
        """Compute strike-relative probability and bet recommendation.

        Args:
            current_price: Override price for distance calculation.
                Use Coinbase price (closer to CF Benchmarks BRTI settlement source)
                instead of BinanceUS candle close which has a ~$15-30 spread.
        """
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        if current_price is None:
            current_price = float(last["close"])
        atr = float(last.get("atr", 0)) if pd.notna(last.get("atr")) else 0

        if atr <= 0 or strike_price <= 0:
            return None

        # 1. Compute distance from strike in ATR units
        distance_atr = (current_price - strike_price) / atr

        # 2. Look up base probability
        base_prob = self._lookup_probability(distance_atr, minutes_remaining)

        # Sanity check: if price is below strike, probability of closing ABOVE
        # cannot exceed 50%. If above strike, probability cannot be below 50%.
        # This prevents the 0.0 bucket's bullish bias from betting YES when below.
        if current_price < strike_price and base_prob > 0.50:
            base_prob = 0.50
        elif current_price > strike_price and base_prob < 0.50:
            base_prob = 0.50

        # 3. Apply technical adjustments
        adjustments = self._compute_adjustments(
            df, current_price, strike_price, distance_atr, market_data, df_1h
        )
        total_adjustment = sum(adjustments.values())

        # 4. Compute adjusted probability
        adjusted_prob = max(0.05, min(0.95, base_prob + total_adjustment))

        # 5. Determine bet recommendation
        recommended_side, max_price = self._decide_bet(adjusted_prob)

        return KalshiV3Signal(
            asset="",
            probability=round(adjusted_prob, 4),
            recommended_side=recommended_side,
            max_price_cents=max_price,
            distance_atr=round(distance_atr, 3),
            base_prob=round(base_prob, 4),
            adjustments=adjustments,
            current_price=current_price,
            strike_price=strike_price,
            minutes_remaining=minutes_remaining,
        )

    def _lookup_probability(self, distance_atr: float, minutes_remaining: float) -> float:
        """Look up base probability from the pre-computed table."""
        # Find nearest distance bucket
        dist_bucket = min(DISTANCE_BINS, key=lambda b: abs(b - distance_atr))
        # Clamp to table range
        dist_bucket = max(DISTANCE_BINS[0], min(DISTANCE_BINS[-1], dist_bucket))

        # Find nearest time bucket
        time_bucket = min(TIME_BINS, key=lambda t: abs(t - minutes_remaining))

        key = f"{dist_bucket}_{time_bucket}"
        cell = self._prob_table.get(key, {})
        return cell.get("probability", 0.5)  # default 50% if cell missing

    def _compute_adjustments(self, df: pd.DataFrame, current_price: float,
                              strike_price: float, distance_atr: float,
                              market_data: dict | None, df_1h: pd.DataFrame | None) -> dict:
        """Compute technical signal adjustments to base probability."""
        adjustments = {}
        last = df.iloc[-1]
        above_strike = current_price >= strike_price

        # --- Positive adjustments (price likely stays on current side) ---

        # Order book confirms (+5%)
        if market_data:
            ob = market_data.get("order_book", {})
            imbalance = ob.get("imbalance", 0)
            if (above_strike and imbalance > 0.2) or (not above_strike and imbalance < -0.2):
                adjustments["ob_confirms"] = 0.05
            elif (above_strike and imbalance < -0.2) or (not above_strike and imbalance > 0.2):
                adjustments["ob_opposes"] = -0.05

        # Trade flow confirms (+5%)
        if market_data:
            tf = market_data.get("trade_flow", {})
            net_flow = tf.get("net_flow", 0)
            buy_ratio = tf.get("buy_ratio", 0.5)
            if (above_strike and net_flow > 0.15 and buy_ratio > 0.55) or \
               (not above_strike and net_flow < -0.15 and buy_ratio < 0.45):
                adjustments["flow_confirms"] = 0.05
            elif (above_strike and net_flow < -0.15) or (not above_strike and net_flow > 0.15):
                adjustments["flow_opposes"] = -0.05

        # 1h trend aligned (+5%)
        if df_1h is not None and len(df_1h) >= 20:
            last_1h = df_1h.iloc[-1]
            rsi_1h = float(last_1h.get("rsi", 50)) if pd.notna(last_1h.get("rsi")) else 50
            macd_1h = float(last_1h.get("macd_hist", 0)) if pd.notna(last_1h.get("macd_hist")) else 0
            trend_1h_up = rsi_1h > 60 and macd_1h > 0
            trend_1h_down = rsi_1h < 40 and macd_1h < 0
            if (above_strike and trend_1h_up) or (not above_strike and trend_1h_down):
                adjustments["1h_aligned"] = 0.05
            elif (above_strike and trend_1h_down) or (not above_strike and trend_1h_up):
                adjustments["1h_opposes"] = -0.05

        # MACD momentum building (+3%)
        macd_hist = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        if len(df) >= 2:
            prev_hist = float(df.iloc[-2].get("macd_hist", 0)) if pd.notna(df.iloc[-2].get("macd_hist")) else 0
            if above_strike and macd_hist > prev_hist and macd_hist > 0:
                adjustments["macd_building"] = 0.03
            elif not above_strike and macd_hist < prev_hist and macd_hist < 0:
                adjustments["macd_building"] = 0.03

        # --- Negative adjustments (potential crossing) ---

        # RSI extreme (-8%)
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
        if above_strike and rsi > 75:
            adjustments["rsi_extreme"] = -0.08
        elif not above_strike and rsi < 25:
            adjustments["rsi_extreme"] = -0.08

        # RSI divergence (-8%)
        if len(df) >= 4:
            c1, c2, c3 = float(df.iloc[-3]["close"]), float(df.iloc[-2]["close"]), current_price
            r1 = float(df.iloc[-3].get("rsi", 50)) if pd.notna(df.iloc[-3].get("rsi")) else 50
            r2 = float(df.iloc[-2].get("rsi", 50)) if pd.notna(df.iloc[-2].get("rsi")) else 50
            r3 = rsi
            if above_strike and c3 > c2 > c1 and r3 < r2 < r1:
                adjustments["rsi_divergence"] = -0.08
            elif not above_strike and c3 < c2 < c1 and r3 > r2 > r1:
                adjustments["rsi_divergence"] = -0.08

        return adjustments

    def _decide_bet(self, adjusted_prob: float) -> tuple[str, int]:
        """Decide bet side and max price from adjusted probability."""
        if adjusted_prob >= 0.65:
            # Bet YES — price likely closes above strike
            fair_price = int(adjusted_prob * 100)
            max_price = min(MAX_BET_PRICE, fair_price - int(EDGE_MARGIN * 100))
            if max_price >= 5:  # minimum viable price
                return "YES", max_price
        elif adjusted_prob <= 0.35:
            # Bet NO — price likely closes below strike
            no_prob = 1.0 - adjusted_prob
            fair_price = int(no_prob * 100)
            max_price = min(MAX_BET_PRICE, fair_price - int(EDGE_MARGIN * 100))
            if max_price >= 5:
                return "NO", max_price

        return "SKIP", 0
