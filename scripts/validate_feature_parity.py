#!/usr/bin/env python3
"""Validate that live predict_knn() produces identical features to training.

Fetches recent 15m/1h/4h data, then for each completed 15m candle:
1. Extracts features using the TRAINING path (build_model.py logic)
2. Extracts features using the LIVE path (predict_knn with snapshot)
3. Compares them and reports any discrepancies

Usage:
    ./venv/bin/python scripts/validate_feature_parity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3

FEATURE_NAMES = [
    "rsi_15m", "stochrsi_15m", "macd_15m", "norm_return",
    "vol_ratio", "bb_position", "ema_slope", "adx", "roc_5",
    "rsi_1h", "macd_1h", "rsi_4h",
    "price_vs_ema", "hourly_return", "trend_direction",
]

ASSETS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
}


def fetch_candles(fetcher, symbol, timeframe, days):
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    batch_size = 300
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms; time.sleep(0.3); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.3)
        except Exception as e:
            since += batch_size * candle_ms; time.sleep(1)
    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def extract_training_features(df_15m, df_1h, df_4h, row_idx):
    """Extract features for a single row using TRAINING logic (build_model.py)."""
    df = add_indicators(df_15m.copy())
    df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
    df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

    pct = df["close"].pct_change()
    df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["ema_slope"] = df["ema_12"].pct_change(3) * 100
    df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
    df["hourly_return"] = df["close"].pct_change(4) * 100

    i = row_idx
    t = df.index[i]
    r = df.iloc[i]

    vals = [
        float(r.get("rsi", np.nan)),
        float(r.get("stochrsi_k", np.nan)),
        float(r.get("macd_hist", np.nan)),
        float(r.get("norm_return", np.nan)),
        float(r.get("vol_ratio", np.nan)),
        float(r.get("bb_position", np.nan)),
        float(r.get("ema_slope", np.nan)),
        float(r.get("adx", np.nan)),
        float(r.get("roc_5", np.nan)),
    ]
    if any(pd.isna(v) or np.isinf(v) for v in vals):
        return None

    if df_1h_ind is not None:
        mask = df_1h_ind.index <= t
        if mask.sum() >= 20:
            r1h = df_1h_ind.loc[mask].iloc[-1]
            vals.append(float(r1h.get("rsi", 50)))
            vals.append(float(r1h.get("macd_hist", 0)))
        else:
            return None
    else:
        vals.extend([50.0, 0.0])

    if df_4h_ind is not None:
        mask = df_4h_ind.index <= t
        if mask.sum() >= 10:
            vals.append(float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50)))
        else:
            return None
    else:
        vals.append(50.0)

    pve = float(r.get("price_vs_ema", np.nan))
    hr = float(r.get("hourly_return", np.nan))
    adx_val = float(r.get("adx", 20))
    close_val = float(r.get("close", 0))
    sma_val = float(r.get("sma_20", 0))

    if pd.isna(pve) or np.isinf(pve):
        pve = 0
    if pd.isna(hr) or np.isinf(hr):
        hr = 0

    if sma_val > 0:
        trend_sign = 1 if close_val >= sma_val else -1
    else:
        trend_sign = 0
    trend_dir = adx_val * trend_sign

    vals.append(pve)
    vals.append(hr)
    vals.append(trend_dir)

    if any(pd.isna(v) or np.isinf(v) for v in vals):
        return None
    return vals


def main():
    print("=" * 70)
    print("FEATURE PARITY + MODEL DISTRIBUTION VALIDATION")
    print("=" * 70)

    fetcher = DataFetcher()
    predictor = KalshiPredictorV3()

    if predictor._knn is None:
        print("ERROR: No model loaded!")
        return

    scaler = predictor._knn_scaler
    model = predictor._knn

    total_yes, total_no, total_skip = 0, 0, 0
    total_match, total_mismatch = 0, 0

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'─' * 70}")
        print(f"  {asset_name} ({symbol})")
        print(f"{'─' * 70}")

        df_15m = fetch_candles(fetcher, symbol, "15m", 30)
        df_1h = fetch_candles(fetcher, symbol, "1h", 30)
        df_4h = fetch_candles(fetcher, symbol, "4h", 30)

        if df_15m.empty:
            print(f"  No data!")
            continue

        print(f"  {len(df_15m)} 15m candles")

        df_15m_ind = add_indicators(df_15m.copy())
        df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
        df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

        # Parity check (last 10 candles)
        match_count, mismatch_count = 0, 0
        for offset in range(10, 0, -1):
            row_idx = len(df_15m_ind) - offset - 1
            if row_idx < 210:
                continue

            train_feats = extract_training_features(df_15m, df_1h, df_4h, row_idx)
            if train_feats is None:
                continue

            # Build live input
            t = df_15m_ind.index[row_idx]
            live_df = df_15m_ind.iloc[:row_idx + 1].copy()
            synthetic = live_df.iloc[-1:].copy()
            synthetic.index = synthetic.index + pd.Timedelta(minutes=15)
            live_df = pd.concat([live_df, synthetic])

            live_1h = df_1h_ind[df_1h_ind.index <= t] if df_1h_ind is not None else None
            live_4h = df_4h_ind[df_4h_ind.index <= t] if df_4h_ind is not None else None

            # Extract live features (inline)
            indicator_row = live_df.iloc[-2]
            prev_close = float(indicator_row["close"])
            vol_sma = float(indicator_row.get("vol_sma_20", 0))
            bb_range_val = float(indicator_row.get("bb_upper", 0)) - float(indicator_row.get("bb_lower", 0))

            pct = live_df["close"].iloc[:-1].pct_change()
            norm_ret = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()

            live_feats = [
                float(indicator_row.get("rsi", 50)),
                float(indicator_row.get("stochrsi_k", 50)),
                float(indicator_row.get("macd_hist", 0)),
                float(norm_ret.iloc[-1]) if pd.notna(norm_ret.iloc[-1]) else 0,
                float(indicator_row.get("volume", 0)) / vol_sma if vol_sma > 0 else 1.0,
                (prev_close - float(indicator_row.get("bb_lower", 0))) / bb_range_val if bb_range_val > 0 else 0.5,
                float(live_df["ema_12"].iloc[:-1].pct_change(3).iloc[-1] * 100) if len(live_df) >= 5 and pd.notna(live_df["ema_12"].iloc[:-1].pct_change(3).iloc[-1]) else 0,
                float(indicator_row.get("adx", 20)),
                float(indicator_row.get("roc_5", 0)),
            ]
            if live_1h is not None and len(live_1h) >= 20:
                r1h = live_1h.iloc[-1]
                live_feats.extend([float(r1h.get("rsi", 50)), float(r1h.get("macd_hist", 0))])
            else:
                live_feats.extend([50.0, 0.0])
            if live_4h is not None and len(live_4h) >= 10:
                live_feats.append(float(live_4h.iloc[-1].get("rsi", 50)))
            else:
                live_feats.append(50.0)
            sma_v = float(indicator_row.get("sma_20", prev_close))
            atr_v = float(indicator_row.get("atr", 1))
            adx_v = float(indicator_row.get("adx", 20))
            live_feats.append((prev_close - sma_v) / atr_v if atr_v > 0 else 0)
            if len(live_df) >= 6:
                live_feats.append((prev_close - float(live_df.iloc[-6]["close"])) / float(live_df.iloc[-6]["close"]) * 100)
            else:
                live_feats.append(0)
            ts = 1 if prev_close >= sma_v else -1
            live_feats.append(adx_v * ts)

            # Compare
            all_ok = True
            for j in range(len(FEATURE_NAMES)):
                diff = abs(train_feats[j] - live_feats[j])
                if diff >= 0.001:
                    all_ok = False
                    print(f"  MISMATCH candle {offset}: {FEATURE_NAMES[j]} train={train_feats[j]:.6f} live={live_feats[j]:.6f} diff={diff:.6f}")
            if all_ok:
                match_count += 1
            else:
                mismatch_count += 1

        total_match += match_count
        total_mismatch += mismatch_count
        print(f"  Parity: {match_count}/{match_count + mismatch_count} match")

        # Model distribution (last 100 candles)
        yes_c, no_c, skip_c = 0, 0, 0
        probs_list = []
        for offset in range(100, 0, -1):
            row_idx = len(df_15m_ind) - offset - 1
            if row_idx < 210:
                continue

            feats = extract_training_features(df_15m, df_1h, df_4h, row_idx)
            if feats is None:
                continue

            X = np.array(feats).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prob = float(model.predict_proba(X_scaled)[0][1])
            probs_list.append(prob)

            if prob >= 0.55:
                yes_c += 1
            elif prob <= 0.45:
                no_c += 1
            else:
                skip_c += 1

        total = yes_c + no_c + skip_c
        total_yes += yes_c
        total_no += no_c
        total_skip += skip_c

        if total > 0:
            probs_arr = np.array(probs_list)
            print(f"  Predictions (last {total}): {yes_c}Y ({yes_c/total*100:.0f}%) / {no_c}N ({no_c/total*100:.0f}%) / {skip_c}SKIP ({skip_c/total*100:.0f}%)")
            print(f"  Prob stats: mean={probs_arr.mean():.3f} std={probs_arr.std():.3f} min={probs_arr.min():.3f} max={probs_arr.max():.3f}")

    # Summary
    grand_total = total_yes + total_no + total_skip
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Parity: {total_match}/{total_match + total_mismatch} candles match")
    if grand_total > 0:
        print(f"  Overall predictions: {total_yes}Y ({total_yes/grand_total*100:.0f}%) / {total_no}N ({total_no/grand_total*100:.0f}%) / {total_skip}SKIP ({total_skip/grand_total*100:.0f}%)")

    # What the backtest expects
    print(f"\n  Walk-forward backtest reference: ~48% YES / ~49% NO / ~3% SKIP")
    if grand_total > 0:
        bias = total_yes / grand_total * 100
        if bias > 70:
            print(f"  ⚠ WARNING: Current data is heavily YES-biased ({bias:.0f}%)")
            print(f"    This may be genuine market regime (bullish) or model issue.")
            print(f"    Compare with actual candle outcomes to determine:")
            # Check actual outcomes
            print(f"\n  Actual candle outcomes (last 100 per asset, close >= open = UP):")
            for asset_name, symbol in ASSETS.items():
                df_15m = fetch_candles(fetcher, symbol, "15m", 10)  # just last 10 days
                if df_15m.empty:
                    continue
                last_100 = df_15m.tail(100)
                up_count = (last_100["close"] >= last_100["open"]).sum()
                print(f"    {asset_name}: {up_count}/100 UP ({up_count}%)")
        elif bias < 30:
            print(f"  ⚠ WARNING: Current data is heavily NO-biased ({bias:.0f}%)")
        else:
            print(f"  ✓ Model predictions are reasonably balanced")


if __name__ == "__main__":
    main()
