#!/usr/bin/env python3
"""Build Logistic Regression model for Kalshi prediction.

Features: 15 total (12 original + 3 trend features)
Trained on Coinbase data (matches BRTI settlement source).

Usage:
    ./venv/bin/python scripts/build_model.py [--days 179]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ASSETS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
}

FEATURE_NAMES = [
    "rsi_15m", "stochrsi_15m", "macd_15m", "norm_return",
    "vol_ratio", "bb_position", "ema_slope", "adx", "roc_5",
    "rsi_1h", "macd_1h", "rsi_4h",
    # Trend features — distinguish bounce from crash
    "price_vs_ema",   # (close - EMA20) / ATR: negative = below trend
    "hourly_return",  # 1h price change %: negative = dropping
    "trend_direction", # ADX * sign(close - EMA): pos = uptrend, neg = downtrend
]


def fetch_candles(fetcher, symbol, timeframe, days):
    """Fetch N days of candles via paginated requests."""
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 24 * 60 * 60 * 1000
    batch_size = 300
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)

    batches = 0
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms
                time.sleep(0.5)
                continue
            all_frames.append(df)
            last_ts = int(df.index[-1].timestamp() * 1000)
            since = last_ts + candle_ms
            batches += 1
            if batches % 10 == 0:
                print(f"    Chunk {batches}: {sum(len(f) for f in all_frames)} candles so far")
            time.sleep(0.5)
        except Exception as e:
            print(f"    Warning: {e}")
            since += batch_size * candle_ms
            time.sleep(2)

    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def extract_features(df_15m, df_1h, df_4h):
    """Extract feature vectors and labels from historical data."""
    df = add_indicators(df_15m.copy())
    df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
    df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

    pct = df["close"].pct_change()
    df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["ema_slope"] = df["ema_12"].pct_change(3) * 100

    # Trend features
    df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
    df["hourly_return"] = df["close"].pct_change(4) * 100  # 4 x 15m = 1 hour

    # Label: next candle closes above its open
    df["next_up"] = (df["close"].shift(-1) >= df["open"].shift(-1)).astype(int)

    features = []
    labels = []

    for i in range(200, len(df) - 1):
        t = df.index[i]
        r = df.iloc[i]

        # Original 9 features
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
            continue

        # 1h context (features 10-11)
        if df_1h_ind is not None:
            mask = df_1h_ind.index <= t
            if mask.sum() >= 20:
                r1h = df_1h_ind.loc[mask].iloc[-1]
                vals.append(float(r1h.get("rsi", 50)))
                vals.append(float(r1h.get("macd_hist", 0)))
            else:
                continue
        else:
            vals.extend([50.0, 0.0])

        # 4h context (feature 12)
        if df_4h_ind is not None:
            mask = df_4h_ind.index <= t
            if mask.sum() >= 10:
                r4h = df_4h_ind.loc[mask].iloc[-1]
                vals.append(float(r4h.get("rsi", 50)))
            else:
                continue
        else:
            vals.append(50.0)

        # Trend features (13-15)
        pve = float(r.get("price_vs_ema", np.nan))
        hr = float(r.get("hourly_return", np.nan))
        adx_val = float(r.get("adx", 20))
        close_val = float(r.get("close", 0))
        sma_val = float(r.get("sma_20", 0))

        if pd.isna(pve) or np.isinf(pve):
            pve = 0
        if pd.isna(hr) or np.isinf(hr):
            hr = 0

        # trend_direction: ADX * sign of (close - sma)
        if sma_val > 0:
            trend_sign = 1 if close_val >= sma_val else -1
        else:
            trend_sign = 0
        trend_dir = adx_val * trend_sign

        vals.append(pve)
        vals.append(hr)
        vals.append(trend_dir)

        if any(pd.isna(v) or np.isinf(v) for v in vals):
            continue

        features.append(vals)
        labels.append(int(r["next_up"]))

    return np.array(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Build LogReg model for Kalshi")
    parser.add_argument("--days", type=int, default=179, help="Days of training data")
    parser.add_argument("--output", default="models/knn_kalshi.pkl")
    args = parser.parse_args()

    fetcher = DataFetcher()
    all_features = []
    all_labels = []

    for asset_name, symbol in ASSETS.items():
        print(f"Fetching {asset_name}...")
        df_15m = fetch_candles(fetcher, symbol, "15m", args.days)
        df_1h = fetch_candles(fetcher, symbol, "1h", args.days)
        df_4h = fetch_candles(fetcher, symbol, "4h", args.days)

        if df_15m.empty:
            print(f"  Skipping {asset_name}: no data")
            continue

        print(f"  {len(df_15m)} 15m candles")
        X, y = extract_features(df_15m, df_1h, df_4h)
        print(f"  {len(X)} samples extracted ({FEATURE_NAMES[-3:]})")
        all_features.append(X)
        all_labels.append(y)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    print(f"\nTotal training samples: {len(X_all)}")
    print(f"Features: {len(FEATURE_NAMES)}")
    print(f"Base rate UP: {y_all.mean():.1%}")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Fit Logistic Regression
    print(f"\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    model.fit(X_scaled, y_all)

    # Self-check
    probs = model.predict_proba(X_scaled)
    yes_n = (probs[:, 1] >= 0.55).sum()
    no_n = (probs[:, 1] <= 0.45).sum()
    skip_n = len(probs) - yes_n - no_n
    high_conf = (probs[:, 1] >= 0.55) | (probs[:, 1] <= 0.45)
    preds = np.where(probs[high_conf, 1] > 0.5, 1, 0)
    train_wr = (preds == y_all[high_conf]).mean() * 100
    print(f"In-sample WR: {train_wr:.1f}% ({yes_n}Y/{no_n}N/{skip_n}SKIP)")

    # Bias check
    mean_prob = model.predict_proba(np.zeros((1, X_scaled.shape[1])))[0][1]
    print(f"Bias at mean: {mean_prob:.3f}")

    # Trend feature test
    print(f"\nTrend feature tests:")

    # Downtrend: low RSI + below EMA + negative hourly return
    test_down = scaler.mean_.copy()
    test_down[0] = 40   # low RSI
    test_down[12] = -2   # price well below EMA
    test_down[13] = -1   # hourly return negative
    test_down[14] = -30  # ADX*-1 = strong downtrend
    X_down = scaler.transform(test_down.reshape(1, -1))
    prob_down = model.predict_proba(X_down)[0][1]

    # Bounce: low RSI + above EMA + positive hourly return
    test_bounce = scaler.mean_.copy()
    test_bounce[0] = 40   # low RSI (same)
    test_bounce[12] = 1    # price above EMA
    test_bounce[13] = 0.5  # hourly return slightly positive
    test_bounce[14] = 20   # moderate uptrend
    X_bounce = scaler.transform(test_bounce.reshape(1, -1))
    prob_bounce = model.predict_proba(X_bounce)[0][1]

    print(f"  Crash (RSI=40 + below EMA + dropping): {prob_down:.3f} -> {'YES' if prob_down >= 0.55 else 'NO' if prob_down <= 0.45 else 'SKIP'}")
    print(f"  Bounce (RSI=40 + above EMA + rising):  {prob_bounce:.3f} -> {'YES' if prob_bounce >= 0.55 else 'NO' if prob_bounce <= 0.45 else 'SKIP'}")
    print(f"  Separation: {prob_bounce - prob_down:.3f}")

    # Overbought tests
    test_top = scaler.mean_.copy()
    test_top[0] = 70
    test_top[12] = 2
    test_top[13] = 1
    test_top[14] = 30
    X_top = scaler.transform(test_top.reshape(1, -1))
    prob_top = model.predict_proba(X_top)[0][1]
    print(f"  Overbought + uptrend: {prob_top:.3f} -> {'YES' if prob_top >= 0.55 else 'NO' if prob_top <= 0.45 else 'SKIP'}")

    # Feature importance
    print(f"\nFeature coefficients:")
    for name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = "UP" if coef > 0 else "DN"
        print(f"  {name:<16}: {coef:>+8.4f} ({direction})")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "knn": model,
            "scaler": scaler,
            "model_type": "logreg",
            "feature_names": FEATURE_NAMES,
            "training_samples": len(X_all),
            "base_rate": float(y_all.mean()),
        }, f)

    print(f"\nSaved model to {output_path}")


if __name__ == "__main__":
    main()
