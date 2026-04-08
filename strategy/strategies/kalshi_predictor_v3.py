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

EDGE_MARGIN = 0.02    # require 2% edge over implied contract price
MAX_BET_PRICE = 60    # max entry — 60c with early exit management caps downside


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

        # Load prediction model(s)
        self._knn = None           # unified model (fallback)
        self._knn_scaler = None
        self._conv_model = None
        self._conv_scaler = None
        self._model_type = None
        self._per_asset_models = {}  # {asset: (model, scaler, feature_names)}
        try:
            import pickle
            # Load per-asset models if they exist
            from pathlib import Path
            for asset_code in ["btc", "eth", "sol", "xrp"]:
                pa_path = Path(f"models/m0_{asset_code}.pkl")
                if pa_path.exists():
                    with open(pa_path, "rb") as f:
                        pa_data = pickle.load(f)
                        self._per_asset_models[asset_code.upper()] = (
                            pa_data["knn"], pa_data["scaler"], pa_data["feature_names"]
                        )
            if self._per_asset_models:
                self._model_type = "per_asset_confluence"
                # Also load unified as fallback
                try:
                    with open(knn_model_path, "rb") as f:
                        model_data = pickle.load(f)
                        self._knn = model_data["knn"]
                        self._knn_scaler = model_data["scaler"]
                except Exception:
                    pass
            else:
                with open(knn_model_path, "rb") as f:
                    model_data = pickle.load(f)
                    self._model_type = model_data.get("model_type", "single")
                    if self._model_type == "dual_trend_conviction":
                        self._knn = model_data["trend_model"]
                        self._knn_scaler = model_data["trend_scaler"]
                        self._conv_model = model_data["conv_model"]
                        self._conv_scaler = model_data["conv_scaler"]
                        self._trend_features = model_data.get("trend_features", [])
                        self._conv_features = model_data.get("conv_features", [])
                    else:
                        self._knn = model_data["knn"]
                        self._knn_scaler = model_data["scaler"]
        except (FileNotFoundError, Exception):
            pass  # no model — early entry disabled

    def predict(self, df: pd.DataFrame, strike_price: float,
                minutes_remaining: float, market_data: dict | None = None,
                df_1h: pd.DataFrame | None = None,
                current_price: float | None = None,
                df_4h: pd.DataFrame | None = None,
                force_table: bool = False,
                kalshi_extra: dict | None = None) -> KalshiV3Signal | None:
        """Compute strike-relative probability and bet recommendation.

        Args:
            current_price: Override price for distance calculation.
                Use Coinbase price (closer to CF Benchmarks BRTI settlement source)
                instead of BinanceUS candle close which has a ~$15-30 spread.
            force_table: If True, skip KNN and use probability table directly.
                Used by daemon for TEK (technical) confluence scoring.
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
        # KNN for early entry (>= 5 min left) — contracts are still cheap
        # Probability table for late entry (< 5 min left) — distance-based
        # force_table=True skips KNN entirely (used for TEK confluence scoring)
        use_knn = (not force_table
                   and minutes_remaining >= 5
                   and self._knn is not None)

        if use_knn:
            # Model predicts probability — strike-relative model needs distance_atr
            knn_prob = self.predict_knn(df, df_1h, df_4h, distance_from_strike=distance_atr,
                                       kalshi_extra=kalshi_extra)
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
                    df_4h: pd.DataFrame | None = None,
                    distance_from_strike: float | None = None,
                    kalshi_extra: dict | None = None) -> float | None:
        """Predict probability that price closes above strike.

        PARITY RULE: All features MUST be read from pre-computed DataFrame columns,
        exactly as the backtest/training scripts compute them. No inline recomputation.
        The daemon's _fetch_all() pre-computes norm_return, vol_ratio, ema_slope,
        price_vs_ema, hourly_return on the full cached DataFrame — use those directly.

        Returns probability (0.0-1.0) or None if models unavailable.
        """
        if self._knn is None and not self._per_asset_models:
            return None
        if df is None or len(df) < 20:
            return None

        # Use the last completed 15m candle — ALWAYS iloc[-1] on pre-filtered data.
        # The daemon passes cached 15m data (no synthetic rows). Backtest does the same.
        # prev_candles = df[df.index < window_start] → iloc[-1] = last completed candle.
        indicator_row = df.iloc[-1]

        try:
            prev_close = float(indicator_row["close"])
            sma_val = float(indicator_row.get("sma_20", prev_close))
            atr_val = float(indicator_row.get("atr", 1))
            adx_val = float(indicator_row.get("adx", 20))

            # trend_direction
            if sma_val > 0:
                trend_sign = 1 if prev_close >= sma_val else -1
            else:
                trend_sign = 0

            # READ pre-computed derived features from DataFrame columns
            # These MUST exist — computed by _fetch_all() on the full cached series
            # If missing, the DataFrame was corrupted by a re-fetch without derived features
            REQUIRED_DERIVED = ["norm_return", "vol_ratio", "ema_slope", "price_vs_ema", "hourly_return"]
            missing = [f for f in REQUIRED_DERIVED if f not in indicator_row.index]
            if missing:
                print(f"  [PARITY ERR] Missing derived features: {missing} — skipping prediction")
                return None

            nr = indicator_row.get("norm_return", 0)
            vr = indicator_row.get("vol_ratio", 1)
            es = indicator_row.get("ema_slope", 0)
            pve = indicator_row.get("price_vs_ema", 0)
            hr = indicator_row.get("hourly_return", 0)

            # 1h/4h: filter by window time from kalshi_extra, matching backtest
            kx = kalshi_extra or {}
            ws_naive = kx.get("window_start_naive")  # set by daemon at eval time

            rsi_1h, macd_1h = 50.0, 0.0
            if df_1h is not None and len(df_1h) >= 20:
                if ws_naive is not None:
                    m1h = df_1h[df_1h.index <= ws_naive]
                    if len(m1h) >= 20:
                        r1h = m1h.iloc[-1]
                        rsi_1h = float(r1h.get("rsi", 50))
                        macd_1h = float(r1h.get("macd_hist", 0))
                else:
                    # Fallback: use iloc[-2] (last completed)
                    r1h = df_1h.iloc[-2] if len(df_1h) >= 21 else df_1h.iloc[-1]
                    rsi_1h = float(r1h.get("rsi", 50))
                    macd_1h = float(r1h.get("macd_hist", 0))

            rsi_4h = 50.0
            if df_4h is not None and len(df_4h) >= 10:
                if ws_naive is not None:
                    m4h = df_4h[df_4h.index <= ws_naive]
                    if len(m4h) >= 10:
                        rsi_4h = float(m4h.iloc[-1].get("rsi", 50))
                else:
                    rsi_4h = float(df_4h.iloc[-2].get("rsi", 50)) if len(df_4h) >= 11 else float(df_4h.iloc[-1].get("rsi", 50))

            all_features = {
                "rsi_15m": float(indicator_row.get("rsi", 50)),
                "stochrsi_15m": float(indicator_row.get("stochrsi_k", 50)),
                "macd_15m": float(indicator_row.get("macd_hist", 0)),
                "norm_return": float(nr) if pd.notna(nr) else 0,
                "vol_ratio": float(vr) if pd.notna(vr) else 1.0,
                "bb_position": 0.5,  # not used by current models
                "ema_slope": float(es) if pd.notna(es) else 0,
                "adx": adx_val,
                "roc_5": float(indicator_row.get("roc_5", 0)),
                "rsi_1h": rsi_1h,
                "macd_1h": macd_1h,
                "rsi_4h": rsi_4h,
                "price_vs_ema": float(pve) if pd.notna(pve) else 0,
                "hourly_return": float(hr) if pd.notna(hr) else 0,
                "trend_direction": adx_val * trend_sign,
            }

            if any(pd.isna(v) or np.isinf(v) for v in all_features.values()):
                return None

            # === Strike-relative / per-asset confluence mode ===
            if self._model_type in ("strike_relative", "per_asset_confluence"):
                if distance_from_strike is None:
                    return None

                all_features["distance_from_strike"] = distance_from_strike

                # Kalshi-specific + time features
                all_features["strike_delta"] = kx.get("strike_delta", 0.0)
                all_features["strike_trend_3"] = kx.get("strike_trend_3", 0.0)

                hour = kx.get("hour", 12)
                all_features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
                all_features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

                all_features["rsi_alignment"] = (
                    (1 if rsi_1h >= 50 else -1) *
                    (1 if rsi_4h >= 50 else -1)
                )
                all_features["atr_percentile"] = kx.get("atr_percentile", 0.5)

                # Bollinger Band Width
                bb_up = float(indicator_row.get("bb_upper", 0))
                bb_lo = float(indicator_row.get("bb_lower", 0))
                bb_m = float(indicator_row.get("sma_20", 0))
                all_features["bbw"] = ((bb_up - bb_lo) / bb_m * 100) if bb_m > 0 else 0

                # Cross-asset confluence features
                all_features["alt_rsi_avg"] = kx.get("alt_rsi_avg", 50)
                all_features["alt_rsi_1h_avg"] = kx.get("alt_rsi_1h_avg", 50)
                all_features["alt_momentum_align"] = kx.get("alt_momentum_align", 0)
                all_features["alt_distance_avg"] = kx.get("alt_distance_avg", 0)

                # Backwards-looking features (may not be in model but included for compatibility)
                all_features["prev_result"] = kx.get("prev_result", 0.5)
                all_features["prev_3_yes_pct"] = kx.get("prev_3_yes_pct", 0.5)
                all_features["streak_length"] = kx.get("streak_length", 0)
                all_features["prev_result_consensus"] = kx.get("prev_result_consensus", 0.5)

                # Select per-asset model or fall back to unified
                asset = kx.get("asset", "")
                if asset in self._per_asset_models:
                    pa_model, pa_scaler, pa_features = self._per_asset_models[asset]
                    vals = [all_features.get(f, 0) for f in pa_features]
                    X = np.array(vals).reshape(1, -1)
                    prob_up = float(pa_model.predict_proba(pa_scaler.transform(X))[0][1])
                elif self._knn is not None:
                    model_features = self._knn_scaler.feature_names_in_ if hasattr(self._knn_scaler, 'feature_names_in_') else None
                    feature_names = model_features if model_features is not None else [
                        "macd_15m", "norm_return", "ema_slope", "roc_5",
                        "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
                        "vol_ratio", "adx", "rsi_1h", "rsi_4h", "distance_from_strike",
                    ]
                    vals = [all_features[f] for f in feature_names]
                    X = np.array(vals).reshape(1, -1)
                    prob_up = float(self._knn.predict_proba(self._knn_scaler.transform(X))[0][1])
                else:
                    return None

                self._log_prediction(prev_close, sma_val, prob_up, all_features)
                return prob_up

            # === Dual-signal mode ===
            if self._model_type == "dual_trend_conviction" and self._conv_model is not None:
                CONV_THRESHOLD = 0.53

                trend_vals = [all_features[f] for f in self._trend_features]
                X_trend = np.array(trend_vals).reshape(1, -1)
                trend_prob = float(self._knn.predict_proba(
                    self._knn_scaler.transform(X_trend))[0][1])

                conv_vals = [all_features[f] for f in self._conv_features]
                X_conv = np.array(conv_vals).reshape(1, -1)
                conv_prob = float(self._conv_model.predict_proba(
                    self._conv_scaler.transform(X_conv))[0][1])

                trend_yes = trend_prob >= CONV_THRESHOLD
                trend_no = trend_prob <= (1 - CONV_THRESHOLD)
                conv_yes = conv_prob >= CONV_THRESHOLD
                conv_no = conv_prob <= (1 - CONV_THRESHOLD)

                if trend_yes and conv_yes:
                    prob_up = trend_prob
                elif trend_no and conv_no:
                    prob_up = trend_prob
                else:
                    prob_up = 0.50

                self._log_prediction(prev_close, sma_val, prob_up, all_features,
                                     trend_prob=trend_prob, conv_prob=conv_prob)
                return prob_up

            # === Single-model mode (legacy) ===
            all_vals = list(all_features.values())
            X = np.array(all_vals).reshape(1, -1)
            X_scaled = self._knn_scaler.transform(X)
            prob_up = float(self._knn.predict_proba(X_scaled)[0][1])

            self._log_prediction(prev_close, sma_val, prob_up, all_features)
            return prob_up

        except Exception:
            return None

    def _log_prediction(self, close, sma, prob, features, trend_prob=None, conv_prob=None):
        """Log prediction features for parity audit."""
        try:
            import json as _json
            from datetime import datetime, timezone
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "close": close,
                "sma_20": sma,
                "prob": round(prob, 4),
                "side": "YES" if prob >= 0.55 else "NO" if prob <= 0.45 else "SKIP",
            }
            if trend_prob is not None:
                entry["trend_prob"] = round(trend_prob, 4)
                entry["conv_prob"] = round(conv_prob, 4)
                entry["model"] = "dual"
            for name, val in features.items():
                entry[name] = round(val, 6)
            with open("data/store/feature_log.jsonl", "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass

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
        """Decide bet side and max price from adjusted probability.

        Gate at 55/45 to match per-asset KNN thresholds (55-60%).
        The daemon's per-asset thresholds do the real filtering.
        """
        if adjusted_prob >= 0.55:
            # Bet YES — max price is the hard cap
            return "YES", MAX_BET_PRICE
        elif adjusted_prob <= 0.45:
            # Bet NO — max price is the hard cap
            return "NO", MAX_BET_PRICE

        return "SKIP", 0

    @staticmethod
    def kelly_size(probability: float, contract_price_cents: int,
                   balance_cents: int, fraction: float = 0.5) -> int:
        """Calculate position size capped by account percentage tiers.

        Uses Kelly direction (higher confidence = larger bet) but caps
        the total risk as a percentage of account balance:
        - prob >= 0.70: up to 10% of balance
        - prob >= 0.65: up to 7.5% of balance
        - prob >= 0.60: up to 5% of balance
        - prob >= 0.55: up to 2.5% of balance

        Args:
            probability: our estimated probability of winning (0-1)
            contract_price_cents: what we'd pay per contract
            balance_cents: total available balance in cents
            fraction: Kelly fraction (unused — kept for API compat)

        Returns:
            Number of contracts to buy
        """
        if contract_price_cents <= 0 or contract_price_cents >= 100:
            return 0
        if balance_cents <= 0:
            return 0

        # Determine max risk as % of balance based on confidence tier
        prob = max(probability, 1.0 - probability)  # use the stronger side
        if prob >= 0.70:
            max_risk_pct = 0.10   # 10% of balance
        elif prob >= 0.65:
            max_risk_pct = 0.075  # 7.5%
        elif prob >= 0.60:
            max_risk_pct = 0.05   # 5%
        elif prob >= 0.55:
            max_risk_pct = 0.025  # 2.5%
        else:
            return 0  # below threshold, don't bet

        # Convert to contract count
        max_risk_cents = int(balance_cents * max_risk_pct)
        count = max_risk_cents // contract_price_cents

        return max(0, count)
