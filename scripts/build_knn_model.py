#!/usr/bin/env python3
"""Build KNN model for Kalshi V3 early-entry prediction.

Trains on historical 15m + 1h + 4h data using 12 features.
Saves the fitted model + scaler for live use.

Usage:
    ./venv/bin/python scripts/build_knn_model.py [--days 90] [--k 50]
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
from backtest_kalshi import fetch_candles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "BNB": "BNB/USDT",
}

FEATURE_NAMES = [
    "rsi_15m", "stochrsi_15m", "macd_15m", "norm_return",
    "vol_ratio", "bb_position", "ema_slope", "adx", "roc_5",
    "rsi_1h", "macd_1h", "rsi_4h",
]


def extract_features(df_15m, df_1h, df_4h):
    """Extract feature vectors and labels from historical data."""
    df = add_indicators(df_15m.copy())
    df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
    df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

    # Extra features
    pct = df["close"].pct_change()
    df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    df["ema_slope"] = df["ema_12"].pct_change(3) * 100

    # Label: next candle closes above its open
    df["next_up"] = (df["close"].shift(-1) >= df["open"].shift(-1)).astype(int)

    features = []
    labels = []

    for i in range(200, len(df) - 1):
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
            continue

        # 1h context
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

        # 4h context
        if df_4h_ind is not None:
            mask = df_4h_ind.index <= t
            if mask.sum() >= 10:
                r4h = df_4h_ind.loc[mask].iloc[-1]
                vals.append(float(r4h.get("rsi", 50)))
            else:
                continue
        else:
            vals.append(50.0)

        if any(pd.isna(v) or np.isinf(v) for v in vals):
            continue

        features.append(vals)
        labels.append(int(r["next_up"]))

    return np.array(features), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Build KNN model for Kalshi V3")
    parser.add_argument("--days", type=int, default=90, help="Days of training data")
    parser.add_argument("--k", type=int, default=50, help="Number of neighbors")
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
        print(f"  {len(X)} samples extracted")
        all_features.append(X)
        all_labels.append(y)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    print(f"\nTotal training samples: {len(X_all)}")
    print(f"Base rate UP: {y_all.mean():.1%}")

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Fit KNN
    print(f"\nTraining KNN with K={args.k}...")
    knn = KNeighborsClassifier(n_neighbors=args.k, weights="distance")
    knn.fit(X_scaled, y_all)

    # Quick self-check (not out-of-sample — just verifying the model works)
    probs = knn.predict_proba(X_scaled)
    high_conf = (probs[:, 1] >= 0.60) | (probs[:, 1] <= 0.40)
    if high_conf.sum() > 0:
        hc_preds = np.where(probs[high_conf, 1] > 0.5, 1, 0)
        hc_wr = (hc_preds == y_all[high_conf]).mean() * 100
        print(f"Self-check (in-sample, >=60% conf): {hc_wr:.1f}% WR ({high_conf.sum()} bets)")

    # Save model + scaler
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({
            "knn": knn,
            "scaler": scaler,
            "k": args.k,
            "feature_names": FEATURE_NAMES,
            "training_samples": len(X_all),
            "base_rate": float(y_all.mean()),
        }, f)

    print(f"\nSaved model to {output_path}")
    print(f"Features: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
