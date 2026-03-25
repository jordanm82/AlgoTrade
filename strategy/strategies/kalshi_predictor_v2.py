# strategy/strategies/kalshi_predictor_v2.py
"""V2 Kalshi predictor: trend/continuation primary, mean-reversion penalties.

Scores confidence for 15-minute crypto direction predictions using:
- Layer 1: 9 trend continuation components (0-100 points)
- Layer 2: 4 mean-reversion penalty components (0 to -45 points)
- Leading indicators (same as V1): order book, trade flow, etc.
- 1-hour trend alignment (same as V1)

Designed to capture trend continuation setups that V1 (mean-reversion) misses.
"""
import pandas as pd
import numpy as np
from strategy.strategies.kalshi_predictor import KalshiSignal

# Trend layer max
_MAX_TREND = 100
# Leading indicators max (same as V1)
_MAX_LEADING = 65
# MTF max
_MAX_MTF = 15


class KalshiPredictorV2:
    """Trend/continuation scorer for Kalshi 15m predictions."""

    def score(self, df: pd.DataFrame, market_data: dict | None = None,
              df_1h: pd.DataFrame | None = None) -> KalshiSignal | None:
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        close = float(last["close"])
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50

        up_score = 0
        down_score = 0
        trend_up = 0  # for filter tracking
        trend_down = 0
        leading_up = 0
        leading_down = 0
        components = {}

        # ═══════════════════════════════════════════════════════
        # LAYER 1: TREND CONTINUATION (9 components, max 100)
        # ═══════════════════════════════════════════════════════

        # 1. Price vs EMA-12 (0-10)
        ema_12 = float(last.get("ema_12", close)) if pd.notna(last.get("ema_12")) else close
        pve_up, pve_down = 0, 0
        pct_from_ema = abs(close - ema_12) / ema_12 * 100 if ema_12 > 0 else 0
        if pct_from_ema > 0.1:
            if close > ema_12:
                pve_up = 10
            else:
                pve_down = 10
        up_score += pve_up; down_score += pve_down
        trend_up += pve_up; trend_down += pve_down
        components["price_vs_ema"] = {"up": pve_up, "down": pve_down}

        # 2. Price vs SMA-20 (0-10)
        sma_20 = float(last.get("sma_20", close)) if pd.notna(last.get("sma_20")) else close
        pvs_up, pvs_down = 0, 0
        pct_from_sma = abs(close - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
        if pct_from_sma > 0.1:
            if close > sma_20:
                pvs_up = 10
            else:
                pvs_down = 10
        up_score += pvs_up; down_score += pvs_down
        trend_up += pvs_up; trend_down += pvs_down
        components["price_vs_sma"] = {"up": pvs_up, "down": pvs_down}

        # 3. EMA-12 vs SMA-20 (0-5)
        evs_up, evs_down = 0, 0
        if ema_12 > sma_20:
            evs_up = 5
        elif ema_12 < sma_20:
            evs_down = 5
        up_score += evs_up; down_score += evs_down
        trend_up += evs_up; trend_down += evs_down
        components["ema_vs_sma"] = {"up": evs_up, "down": evs_down}

        # 4. ADX (0-15)
        adx_val = float(last.get("adx", 0)) if pd.notna(last.get("adx")) else 0
        adx_score = 0
        if adx_val > 40:
            adx_score = 15
        elif adx_val > 30:
            adx_score = 10
        elif adx_val > 20:
            adx_score = 5
        # ADX confirms the dominant direction
        if up_score > down_score:
            up_score += adx_score
            trend_up += adx_score
        elif down_score > up_score:
            down_score += adx_score
            trend_down += adx_score
        components["adx"] = {"score": adx_score, "value": adx_val}

        # 5. MACD histogram trend (0-15)
        macd_score = 0
        macd_hist = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        prev_hist = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
        if len(df) >= 3:
            prev2_hist = float(df.iloc[-3].get("macd_hist", 0)) if pd.notna(df.iloc[-3].get("macd_hist")) else 0
            # Growing for 2+ candles
            if macd_hist > prev_hist > prev2_hist and macd_hist > 0:
                macd_score = 15  # growing UP
            elif macd_hist < prev_hist < prev2_hist and macd_hist < 0:
                macd_score = 15  # growing DOWN
            elif (macd_hist > prev_hist and macd_hist > 0) or (macd_hist < prev_hist and macd_hist < 0):
                macd_score = 8   # growing 1 candle
            elif (macd_hist > 0 and up_score > down_score) or (macd_hist < 0 and down_score > up_score):
                macd_score = 3   # positive but not growing
        if up_score > down_score:
            up_score += macd_score
            trend_up += macd_score
        else:
            down_score += macd_score
            trend_down += macd_score
        components["macd_trend"] = {"score": macd_score, "value": macd_hist}

        # 6. Consecutive candles (0-10)
        consec_score = 0
        if len(df) >= 5:
            up_count = 0
            down_count = 0
            for j in range(-1, -6, -1):
                c = float(df.iloc[j]["close"])
                o = float(df.iloc[j]["open"])
                if c >= o:
                    up_count += 1
                    if down_count > 0:
                        break
                else:
                    down_count += 1
                    if up_count > 0:
                        break
            count = max(up_count, down_count)
            if count >= 4:
                consec_score = 10
            elif count >= 3:
                consec_score = 7
            elif count >= 2:
                consec_score = 3
            if up_count > down_count:
                up_score += consec_score
                trend_up += consec_score
            else:
                down_score += consec_score
                trend_down += consec_score
        components["consecutive"] = {"score": consec_score}

        # 7. Higher highs / higher lows (0-10)
        hh_score = 0
        if len(df) >= 4:
            hh_count = 0
            hl_count = 0
            ll_count = 0
            lh_count = 0
            for j in range(-1, -4, -1):
                if float(df.iloc[j]["high"]) > float(df.iloc[j-1]["high"]):
                    hh_count += 1
                if float(df.iloc[j]["low"]) > float(df.iloc[j-1]["low"]):
                    hl_count += 1
                if float(df.iloc[j]["low"]) < float(df.iloc[j-1]["low"]):
                    ll_count += 1
                if float(df.iloc[j]["high"]) < float(df.iloc[j-1]["high"]):
                    lh_count += 1
            if hh_count >= 3 and hl_count >= 3:
                hh_score = 10
                up_score += hh_score; trend_up += hh_score
            elif hh_count >= 2 and hl_count >= 2:
                hh_score = 5
                up_score += hh_score; trend_up += hh_score
            elif ll_count >= 3 and lh_count >= 3:
                hh_score = 10
                down_score += hh_score; trend_down += hh_score
            elif ll_count >= 2 and lh_count >= 2:
                hh_score = 5
                down_score += hh_score; trend_down += hh_score
        components["hh_hl"] = {"score": hh_score}

        # 8. ROC-5 aligned (0-10)
        roc_score = 0
        roc_val = float(last.get("roc_5", 0)) if pd.notna(last.get("roc_5")) else 0
        if roc_val > 1.0:
            roc_score = 10
            up_score += roc_score; trend_up += roc_score
        elif roc_val > 0.3:
            roc_score = 5
            up_score += roc_score; trend_up += roc_score
        elif roc_val < -1.0:
            roc_score = 10
            down_score += roc_score; trend_down += roc_score
        elif roc_val < -0.3:
            roc_score = 5
            down_score += roc_score; trend_down += roc_score
        components["roc_aligned"] = {"score": roc_score, "value": roc_val}

        # 9. Volume confirmation (0-15: trend 0-8 + vs avg 0-7)
        vol_trend_score = 0
        vol_avg_score = 0
        vol = float(last.get("volume", 0))
        vol_sma = float(last.get("vol_sma_20", 0)) if pd.notna(last.get("vol_sma_20")) else 0
        if len(df) >= 3:
            v1 = float(df.iloc[-3]["volume"])
            v2 = float(df.iloc[-2]["volume"])
            v3 = vol
            if v3 > v2 > v1:
                vol_trend_score = 8
            elif abs(v3 - v2) / max(v2, 1) < 0.1:
                vol_trend_score = 3
        if vol_sma > 0:
            if vol > vol_sma * 2:
                vol_avg_score = 7
            elif vol > vol_sma * 1.5:
                vol_avg_score = 4
            elif vol > vol_sma:
                vol_avg_score = 2
        vol_total = vol_trend_score + vol_avg_score
        if up_score > down_score:
            up_score += vol_total; trend_up += vol_total
        else:
            down_score += vol_total; trend_down += vol_total
        components["volume"] = {"trend": vol_trend_score, "vs_avg": vol_avg_score, "total": vol_total}

        # ═══════════════════════════════════════════════════════
        # LAYER 2: MEAN-REVERSION PENALTIES (max -45)
        # ═══════════════════════════════════════════════════════
        dominant_is_up = up_score > down_score

        # Penalty 1: RSI extreme (-15)
        rsi_penalty = 0
        if dominant_is_up and rsi > 80:
            rsi_penalty = -15
        elif dominant_is_up and rsi > 75:
            rsi_penalty = -8
        elif not dominant_is_up and rsi < 20:
            rsi_penalty = -15
        elif not dominant_is_up and rsi < 25:
            rsi_penalty = -8
        if dominant_is_up:
            up_score += rsi_penalty
        else:
            down_score += rsi_penalty
        components["penalty_rsi"] = {"score": rsi_penalty, "value": rsi}

        # Penalty 2: StochRSI extreme (-10)
        stoch_penalty = 0
        stochrsi_k = float(last.get("stochrsi_k", 50)) if pd.notna(last.get("stochrsi_k")) else 50
        if dominant_is_up and stochrsi_k > 95:
            stoch_penalty = -10
        elif dominant_is_up and stochrsi_k > 90:
            stoch_penalty = -5
        elif not dominant_is_up and stochrsi_k < 5:
            stoch_penalty = -10
        elif not dominant_is_up and stochrsi_k < 10:
            stoch_penalty = -5
        if dominant_is_up:
            up_score += stoch_penalty
        else:
            down_score += stoch_penalty
        components["penalty_stochrsi"] = {"score": stoch_penalty, "value": stochrsi_k}

        # Penalty 3: BB overextension (-10)
        bb_penalty = 0
        bb_upper = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0
        bb_lower = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
        if bb_upper > 0 and dominant_is_up:
            if close > bb_upper * 1.01:
                bb_penalty = -10
            elif close > bb_upper:
                bb_penalty = -5
        if bb_lower > 0 and not dominant_is_up:
            if close < bb_lower * 0.99:
                bb_penalty = -10
            elif close < bb_lower:
                bb_penalty = -5
        if dominant_is_up:
            up_score += bb_penalty
        else:
            down_score += bb_penalty
        components["penalty_bb"] = {"score": bb_penalty}

        # Penalty 4: RSI divergence (-10)
        div_penalty = 0
        if len(df) >= 4:
            c1, c2, c3 = float(df.iloc[-3]["close"]), float(df.iloc[-2]["close"]), close
            r1 = float(df.iloc[-3].get("rsi", 50)) if pd.notna(df.iloc[-3].get("rsi")) else 50
            r2 = float(df.iloc[-2].get("rsi", 50)) if pd.notna(df.iloc[-2].get("rsi")) else 50
            r3 = rsi
            if dominant_is_up:
                if c3 > c2 > c1 and r3 < r2 < r1:
                    div_penalty = -10
            else:
                if c3 < c2 < c1 and r3 > r2 > r1:
                    div_penalty = -10
        if dominant_is_up:
            up_score += div_penalty
        else:
            down_score += div_penalty
        components["penalty_divergence"] = {"score": div_penalty}

        # ═══════════════════════════════════════════════════════
        # LEADING INDICATORS (same as V1, components 9-13)
        # ═══════════════════════════════════════════════════════
        has_leading = market_data is not None
        ob = (market_data or {}).get("order_book", {})
        tf = (market_data or {}).get("trade_flow", {})
        cross = (market_data or {}).get("cross_asset", {})

        # Order Book Imbalance (0-20)
        ob_up, ob_down = 0, 0
        imbalance = ob.get("imbalance", 0)
        if imbalance > 0.3: ob_up = 20
        elif imbalance > 0.15: ob_up = 10
        elif imbalance < -0.3: ob_down = 20
        elif imbalance < -0.15: ob_down = 10
        up_score += ob_up; down_score += ob_down
        leading_up += ob_up; leading_down += ob_down
        components["order_book"] = {"up": ob_up, "down": ob_down, "imbalance": imbalance}

        # Trade Flow (0-20)
        tf_up, tf_down = 0, 0
        net_flow = tf.get("net_flow", 0)
        buy_ratio = tf.get("buy_ratio", 0.5)
        if net_flow > 0.2 and buy_ratio > 0.55: tf_up = 20
        elif net_flow > 0.1: tf_up = 10
        elif net_flow < -0.2 and buy_ratio < 0.45: tf_down = 20
        elif net_flow < -0.1: tf_down = 10
        up_score += tf_up; down_score += tf_down
        leading_up += tf_up; leading_down += tf_down
        components["trade_flow"] = {"up": tf_up, "down": tf_down}

        # Large Trade Bias (0-10)
        lt_up, lt_down = 0, 0
        large_bias = tf.get("large_trade_bias", 0)
        if large_bias > 0.3: lt_up = 10
        elif large_bias < -0.3: lt_down = 10
        up_score += lt_up; down_score += lt_down
        leading_up += lt_up; leading_down += lt_down
        components["large_trade"] = {"up": lt_up, "down": lt_down}

        # Spread (0-5)
        spread_score = 0
        spread_pct = ob.get("spread_pct", 0)
        if spread_pct > 0.1: spread_score = 5
        if up_score > down_score:
            up_score += spread_score; leading_up += spread_score
        else:
            down_score += spread_score; leading_down += spread_score
        components["spread"] = {"score": spread_score}

        # Cross-Asset (0-10)
        ca_up, ca_down = 0, 0
        btc_dir = cross.get("market_direction", 0)
        if btc_dir < -1: ca_down = 10
        elif btc_dir > 1: ca_up = 10
        up_score += ca_up; down_score += ca_down
        leading_up += ca_up; leading_down += ca_down
        components["cross_asset"] = {"up": ca_up, "down": ca_down}

        # ═══════════════════════════════════════════════════════
        # MTF: 1-Hour Trend Alignment (-15 to +15)
        # ═══════════════════════════════════════════════════════
        mtf_score = 0
        if df_1h is not None and len(df_1h) >= 20:
            last_1h = df_1h.iloc[-1]
            rsi_1h = float(last_1h.get("rsi", 50)) if pd.notna(last_1h.get("rsi")) else 50
            macd_1h = float(last_1h.get("macd_hist", 0)) if pd.notna(last_1h.get("macd_hist")) else 0
            trend_1h_up = rsi_1h > 60 and macd_1h > 0
            trend_1h_down = rsi_1h < 40 and macd_1h < 0
            dom_up = up_score > down_score
            if trend_1h_up and dom_up: mtf_score = 15
            elif trend_1h_down and not dom_up: mtf_score = 15
            elif trend_1h_up and not dom_up: mtf_score = -15
            elif trend_1h_down and dom_up: mtf_score = -15
            if mtf_score > 0:
                if dom_up: up_score += mtf_score
                else: down_score += mtf_score
            elif mtf_score < 0:
                if dom_up: up_score += mtf_score
                else: down_score += mtf_score
        components["mtf"] = {"score": mtf_score}

        # ═══════════════════════════════════════════════════════
        # FILTERS (same as V1)
        # ═══════════════════════════════════════════════════════
        if self._apply_filters(up_score, down_score, trend_up, trend_down,
                               leading_up, leading_down, df, components):
            return None

        # ═══════════════════════════════════════════════════════
        # NORMALIZE AND RETURN
        # ═══════════════════════════════════════════════════════
        if has_leading:
            max_possible = _MAX_TREND + _MAX_LEADING + _MAX_MTF
        elif df_1h is not None:
            max_possible = _MAX_TREND + _MAX_MTF
        else:
            max_possible = _MAX_TREND

        if up_score > down_score and up_score > 0:
            confidence = min(100, max(0, int(up_score * 100 / max_possible)))
            return KalshiSignal(asset="", direction="UP", confidence=confidence,
                                components=components, price=close, rsi=rsi)
        elif down_score > up_score and down_score > 0:
            confidence = min(100, max(0, int(down_score * 100 / max_possible)))
            return KalshiSignal(asset="", direction="DOWN", confidence=confidence,
                                components=components, price=close, rsi=rsi)
        return None

    def check_1m_momentum(self, df_1m: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
        """Same as V1 — 2 of 3 candles in predicted direction."""
        if df_1m is None or len(df_1m) < lookback:
            return False
        recent = df_1m.iloc[-lookback:]
        if direction == "UP":
            confirming = (recent["close"] > recent["open"]).sum()
        else:
            confirming = (recent["close"] < recent["open"]).sum()
        return bool(confirming >= 2)

    def compute_5m_booster(self, df_5m: pd.DataFrame, direction: str,
                            window_open_price: float) -> int:
        """Same as V1 — available for future use."""
        if df_5m is None or len(df_5m) < 2:
            return 0
        last = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        booster = 0
        is_up = direction == "UP"
        close = float(last["close"])
        open_price = float(last["open"])
        prev_close = float(prev["close"])
        candle_is_green = close > open_price
        if (is_up and candle_is_green) or (not is_up and not candle_is_green):
            booster += 3
        vol = float(last.get("volume", 0))
        vol_sma = float(last.get("vol_sma_20", 0)) if pd.notna(last.get("vol_sma_20")) else 0
        if vol_sma > 0 and vol > vol_sma * 1.5:
            booster += 3
        atr = float(last.get("atr", 0)) if pd.notna(last.get("atr")) else 0
        if atr > 0 and window_open_price > 0:
            distance = close - window_open_price
            if (is_up and distance > 0.5 * atr) or (not is_up and distance < -0.5 * atr):
                booster += 3
        macd_now = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        macd_prev = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
        crossed = (macd_now > 0 and macd_prev <= 0) or (macd_now < 0 and macd_prev >= 0)
        if crossed:
            aligned = (is_up and macd_now > 0) or (not is_up and macd_now < 0)
            if aligned: booster += 3
            else: booster -= 5
        rsi_now = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
        rsi_prev = float(prev.get("rsi", 50)) if pd.notna(prev.get("rsi")) else 50
        if is_up and close > prev_close and rsi_now < rsi_prev:
            booster -= 5
        elif not is_up and close < prev_close and rsi_now > rsi_prev:
            booster -= 5
        return max(-10, min(12, booster))

    def _apply_filters(self, up_score, down_score, trend_up, trend_down,
                       leading_up, leading_down, df, components):
        """Same filter logic as V1."""
        # Filter 1: Directional conflict
        trend_dir = "UP" if trend_up > trend_down else "DOWN"
        lead_dir = "UP" if leading_up > leading_down else "DOWN"
        trend_strength = max(trend_up, trend_down)
        lead_strength = max(leading_up, leading_down)
        if trend_dir != lead_dir and trend_strength >= 15 and lead_strength >= 15:
            components["filter_conflict"] = True
            return True

        # Filter 2: Volatility regime
        if "atr" in df.columns and len(df) >= 200:
            atr_series = df["atr"].dropna().tail(200)
            if len(atr_series) >= 50:
                current_atr = float(atr_series.iloc[-1])
                percentile = (atr_series < current_atr).sum() / len(atr_series) * 100
                if percentile > 90:
                    components["filter_volatility"] = True
                    return True

        # Filter 3: Margin of victory
        winner = max(up_score, down_score)
        loser = min(up_score, down_score)
        if loser > 0 and winner < loser * 1.5:
            components["filter_margin"] = True
            return True

        return False
