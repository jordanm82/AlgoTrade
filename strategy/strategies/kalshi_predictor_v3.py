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
MAX_BET_PRICE = 85    # max price scales with probability, hard cap at 85c


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
    """Strike-relative probability predictor for Kalshi 15m contracts.

    Two prediction modes:
    - Late entry (minute 10+): probability table + technical adjustments
    - Early entry (minute 0-5): KNN model trained on multi-timeframe features
    """

    def __init__(self, prob_table_path: str = "data/store/kalshi_prob_table.json",
                 knn_model_path: str = "models/knn_kalshi.pkl"):
        self._prob_table = {}
        try:
            with open(prob_table_path) as f:
                self._prob_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Load KNN model for early-entry prediction
        self._knn = None
        self._knn_scaler = None
        try:
            import pickle
            with open(knn_model_path, "rb") as f:
                model_data = pickle.load(f)
                self._knn = model_data["knn"]
                self._knn_scaler = model_data["scaler"]
        except (FileNotFoundError, Exception):
            pass  # no KNN model — early entry disabled

    def predict(self, df: pd.DataFrame, strike_price: float,
                minutes_remaining: float, market_data: dict | None = None,
                df_1h: pd.DataFrame | None = None,
                current_price: float | None = None,
                df_4h: pd.DataFrame | None = None) -> KalshiV3Signal | None:
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

        # 2. Choose prediction mode based on timing + distance
        # Early entry (>= 10 min left, near strike): use KNN for direction prediction
        # Late entry (< 10 min left, or far from strike): use probability table
        use_knn = (minutes_remaining >= 10
                   and abs(distance_atr) < 0.5
                   and self._knn is not None)

        if use_knn:
            # KNN predicts probability of next candle going UP
            # KNN already captures multi-timeframe context — skip hand-tuned adjustments
            knn_prob = self.predict_knn(df, df_1h, df_4h)
            if knn_prob is not None:
                # Use raw KNN probability directly — no adjustments, no gates
                recommended_side, max_price = self._decide_bet(knn_prob)
                return KalshiV3Signal(
                    asset="",
                    probability=round(knn_prob, 4),
                    recommended_side=recommended_side,
                    max_price_cents=max_price,
                    distance_atr=round(distance_atr, 3),
                    base_prob=round(knn_prob, 4),
                    adjustments={"mode": "knn_early_entry"},
                    current_price=current_price,
                    strike_price=strike_price,
                    minutes_remaining=minutes_remaining,
                )
            # KNN failed — fall through to table lookup

        # Standard probability table lookup (late entry or KNN fallback)
        base_prob = self._lookup_probability(distance_atr, minutes_remaining)

        # Sanity check: cap probability when on wrong side of strike
        if current_price < strike_price and base_prob > 0.50:
            base_prob = 0.50
        elif current_price > strike_price and base_prob < 0.50:
            base_prob = 0.50

        # 3. Apply technical adjustments
        adjustments = self._compute_adjustments(
            df, current_price, strike_price, distance_atr, market_data, df_1h
        )
        total_adjustment = sum(adjustments.values())

        # 4. Momentum gate — if price is moving toward the strike (against our side),
        # apply a heavy penalty. Don't bet YES on a falling price near the strike.
        if len(df) >= 3:
            c1 = float(df.iloc[-3]["close"])
            c2 = float(df.iloc[-2]["close"])
            c3 = current_price
            above_strike = current_price >= strike_price

            if above_strike:
                # Betting YES — check if price is falling toward strike
                if c3 < c2 < c1:
                    # 3 consecutive drops while above strike — price heading down to cross
                    adjustments["momentum_against"] = -0.15
            else:
                # Betting NO — check if price is rising toward strike
                if c3 > c2 > c1:
                    # 3 consecutive rises while below strike — price heading up to cross
                    adjustments["momentum_against"] = -0.15

            total_adjustment = sum(adjustments.values())

        # 5. Compute adjusted probability
        adjusted_prob = max(0.05, min(0.95, base_prob + total_adjustment))

        # 6. Determine bet recommendation
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

    def predict_knn(self, df: pd.DataFrame, df_1h: pd.DataFrame | None = None,
                    df_4h: pd.DataFrame | None = None) -> float | None:
        """Use KNN model to predict probability of next candle going UP.

        Returns probability (0.0-1.0) or None if KNN model not available.
        Used for early entry (minute 0-5) when distance from strike is minimal.
        """
        if self._knn is None or self._knn_scaler is None:
            return None
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]

        # Extract same 12 features used in training
        try:
            pct = df["close"].pct_change()
            norm_ret_series = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
            vol_sma = float(last.get("vol_sma_20", 0))
            bb_range = float(last.get("bb_upper", 0)) - float(last.get("bb_lower", 0))

            vals = [
                float(last.get("rsi", 50)),
                float(last.get("stochrsi_k", 50)),
                float(last.get("macd_hist", 0)),
                float(norm_ret_series.iloc[-1]) if pd.notna(norm_ret_series.iloc[-1]) else 0,
                float(last.get("volume", 0)) / vol_sma if vol_sma > 0 else 1.0,
                (float(last["close"]) - float(last.get("bb_lower", 0))) / bb_range if bb_range > 0 else 0.5,
                float(df["ema_12"].pct_change(3).iloc[-1] * 100) if len(df) >= 4 and pd.notna(df["ema_12"].pct_change(3).iloc[-1]) else 0,
                float(last.get("adx", 20)),
                float(last.get("roc_5", 0)),
            ]

            # 1h context
            if df_1h is not None and len(df_1h) >= 20:
                r1h = df_1h.iloc[-1]
                vals.append(float(r1h.get("rsi", 50)))
                vals.append(float(r1h.get("macd_hist", 0)))
            else:
                vals.extend([50.0, 0.0])

            # 4h context
            if df_4h is not None and len(df_4h) >= 10:
                r4h = df_4h.iloc[-1]
                vals.append(float(r4h.get("rsi", 50)))
            else:
                vals.append(50.0)

            # Check for NaN/inf
            if any(pd.isna(v) or np.isinf(v) for v in vals):
                return None

            X = np.array(vals).reshape(1, -1)
            X_scaled = self._knn_scaler.transform(X)
            probs = self._knn.predict_proba(X_scaled)
            return float(probs[0][1])  # probability of UP

        except Exception:
            return None

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
