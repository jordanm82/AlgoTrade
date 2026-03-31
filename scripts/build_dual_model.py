#!/usr/bin/env python3
"""Build dual LogReg models — one for UP regime, one for DOWN regime.

Regime split: close > SMA_20 = UP, close < SMA_20 = DOWN
Each model learns the patterns that work in its regime.

Usage:
    ./venv/bin/python scripts/build_dual_model.py [--days 179]
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
    "price_vs_ema", "hourly_return", "trend_direction",
]


def fetch_candles(fetcher, symbol, timeframe, days):
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    batch_size = 300
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    batches = 0
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms; time.sleep(0.5); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            batches += 1
            if batches % 10 == 0:
                print(f"    Chunk {batches}: {sum(len(f) for f in all_frames)} candles")
            time.sleep(0.5)
        except Exception as e:
            print(f"    Warning: {e}")
            since += batch_size * candle_ms; time.sleep(2)
    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def extract_features(df_15m, df_1h, df_4h):
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
    df["next_up"] = (df["close"].shift(-1) >= df["open"].shift(-1)).astype(int)

    # Regime: close vs SMA_20
    df["regime"] = np.where(df["close"] >= df["sma_20"], "UP", "DOWN")

    features = []
    labels = []
    regimes = []

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

        if df_1h_ind is not None:
            mask = df_1h_ind.index <= t
            if mask.sum() >= 20:
                r1h = df_1h_ind.loc[mask].iloc[-1]
                vals.extend([float(r1h.get("rsi", 50)), float(r1h.get("macd_hist", 0))])
            else:
                continue
        else:
            vals.extend([50.0, 0.0])

        if df_4h_ind is not None:
            mask = df_4h_ind.index <= t
            if mask.sum() >= 10:
                vals.append(float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50)))
            else:
                continue
        else:
            vals.append(50.0)

        pve = float(r.get("price_vs_ema", 0))
        hr = float(r.get("hourly_return", 0))
        adx_val = float(r.get("adx", 20))
        close_val = float(r.get("close", 0))
        sma_val = float(r.get("sma_20", 0))
        trend_sign = 1 if close_val >= sma_val else -1
        vals.extend([
            pve if not (pd.isna(pve) or np.isinf(pve)) else 0,
            hr if not (pd.isna(hr) or np.isinf(hr)) else 0,
            adx_val * trend_sign,
        ])

        if any(pd.isna(v) or np.isinf(v) for v in vals):
            continue

        features.append(vals)
        labels.append(int(r["next_up"]))
        regimes.append(r["regime"])

    return np.array(features), np.array(labels), np.array(regimes)


def train_and_evaluate(X, y, name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)[:, 1]
    yes_n = (probs >= 0.55).sum()
    no_n = (probs <= 0.45).sum()
    skip_n = len(probs) - yes_n - no_n
    high_conf = (probs >= 0.55) | (probs <= 0.45)
    preds = np.where(probs[high_conf, 1] if high_conf.ndim > 1 else probs[high_conf] > 0.5, 1, 0)
    wr = (preds == y[high_conf]).mean() * 100

    print(f"  {name}: {len(X)} samples | {wr:.1f}% WR | {yes_n}Y/{no_n}N/{skip_n}SKIP")
    print(f"    Base rate UP: {y.mean():.1%}")

    # Show top features
    for feat_name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"    {feat_name:<16}: {coef:>+8.4f}")

    return model, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=179)
    parser.add_argument("--output", default="models/knn_kalshi.pkl")
    args = parser.parse_args()

    fetcher = DataFetcher()
    all_features, all_labels, all_regimes = [], [], []

    for asset_name, symbol in ASSETS.items():
        print(f"Fetching {asset_name}...")
        df_15m = fetch_candles(fetcher, symbol, "15m", args.days)
        df_1h = fetch_candles(fetcher, symbol, "1h", args.days)
        df_4h = fetch_candles(fetcher, symbol, "4h", args.days)
        if df_15m.empty:
            continue
        print(f"  {len(df_15m)} 15m candles")
        X, y, r = extract_features(df_15m, df_1h, df_4h)
        print(f"  {len(X)} samples ({(r == 'UP').sum()} UP regime / {(r == 'DOWN').sum()} DOWN regime)")
        all_features.append(X)
        all_labels.append(y)
        all_regimes.append(r)

    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    r_all = np.concatenate(all_regimes)

    print(f"\nTotal: {len(X_all)} samples")
    print(f"Regime split: {(r_all == 'UP').sum()} UP / {(r_all == 'DOWN').sum()} DOWN")

    # Train regime-specific models
    up_mask = r_all == "UP"
    down_mask = r_all == "DOWN"

    print(f"\n{'=' * 60}")
    print("TRAINING DUAL MODELS")
    print(f"{'=' * 60}")

    up_model, up_scaler = train_and_evaluate(X_all[up_mask], y_all[up_mask], "UP model")
    down_model, down_scaler = train_and_evaluate(X_all[down_mask], y_all[down_mask], "DOWN model")

    # Also train a single model for comparison
    print()
    single_model, single_scaler = train_and_evaluate(X_all, y_all, "SINGLE model")

    # Test: what does each model predict for different scenarios?
    print(f"\n{'=' * 60}")
    print("SCENARIO TESTS")
    print(f"{'=' * 60}")

    scenarios = [
        ("Low RSI + uptrend", {0: 30, 12: 1.5, 13: 0.5, 14: 25}),
        ("Low RSI + downtrend", {0: 30, 12: -1.5, 13: -0.5, 14: -25}),
        ("Mid RSI + uptrend", {0: 50, 12: 1.0, 13: 0.3, 14: 20}),
        ("Mid RSI + downtrend", {0: 50, 12: -1.0, 13: -0.3, 14: -20}),
        ("High RSI + uptrend", {0: 70, 12: 2.0, 13: 1.0, 14: 30}),
        ("High RSI + downtrend", {0: 70, 12: -0.5, 13: -0.3, 14: -15}),
    ]

    print(f"{'Scenario':<28} {'Single':>8} {'UP mdl':>8} {'DOWN mdl':>8} {'Dual':>8}")
    print("-" * 60)

    for name, overrides in scenarios:
        test = single_scaler.mean_.copy()
        for idx, val in overrides.items():
            test[idx] = val

        # Single model
        X_s = single_scaler.transform(test.reshape(1, -1))
        p_s = single_model.predict_proba(X_s)[0][1]

        # UP model
        X_u = up_scaler.transform(test.reshape(1, -1))
        p_u = up_model.predict_proba(X_u)[0][1]

        # DOWN model
        X_d = down_scaler.transform(test.reshape(1, -1))
        p_d = down_model.predict_proba(X_d)[0][1]

        # Dual: pick based on regime (price_vs_ema > 0 = UP)
        is_up = overrides.get(12, 0) > 0
        p_dual = p_u if is_up else p_d

        def side(p):
            return f"{'YES' if p >= 0.55 else 'NO' if p <= 0.45 else 'SKIP'} {int(p*100)}%"

        print(f"{name:<28} {side(p_s):>8} {side(p_u):>8} {side(p_d):>8} {side(p_dual):>8}")

    # Walk-forward: train on first 70%, test on last 30%
    print(f"\n{'=' * 60}")
    print("WALK-FORWARD VALIDATION (70/30 split)")
    print(f"{'=' * 60}")

    split = int(len(X_all) * 0.7)
    X_train, y_train, r_train = X_all[:split], y_all[:split], r_all[:split]
    X_test, y_test, r_test = X_all[split:], y_all[split:], r_all[split:]

    # Train on train set
    up_tr = r_train == "UP"
    dn_tr = r_train == "DOWN"

    up_scaler_wf = StandardScaler()
    up_model_wf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    up_model_wf.fit(up_scaler_wf.fit_transform(X_train[up_tr]), y_train[up_tr])

    dn_scaler_wf = StandardScaler()
    dn_model_wf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    dn_model_wf.fit(dn_scaler_wf.fit_transform(X_train[dn_tr]), y_train[dn_tr])

    single_scaler_wf = StandardScaler()
    single_model_wf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    single_model_wf.fit(single_scaler_wf.fit_transform(X_train), y_train)

    # Evaluate on test set
    for label, model_fn in [
        ("Single model", lambda X, r_i: single_model_wf.predict_proba(single_scaler_wf.transform(X.reshape(1, -1)))[0][1]),
        ("Dual model", lambda X, r_i: up_model_wf.predict_proba(up_scaler_wf.transform(X.reshape(1, -1)))[0][1] if r_i == "UP" else dn_model_wf.predict_proba(dn_scaler_wf.transform(X.reshape(1, -1)))[0][1]),
    ]:
        wins, losses, yes_n, no_n, skip_n = 0, 0, 0, 0, 0
        for i in range(len(X_test)):
            prob = model_fn(X_test[i], r_test[i])
            if prob >= 0.55:
                yes_n += 1
                if y_test[i] == 1: wins += 1
                else: losses += 1
            elif prob <= 0.45:
                no_n += 1
                if y_test[i] == 0: wins += 1
                else: losses += 1
            else:
                skip_n += 1

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        w = wins
        pnl = w * 0.50 - losses * 0.50
        print(f"  {label:<15}: {wr:.1f}% WR ({total} bets, {yes_n}Y/{no_n}N/{skip_n}SKIP) P&L ${pnl:+.0f}")

    # Save dual model
    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump({
            "knn": up_model,  # default for compat
            "scaler": up_scaler,
            "up_model": up_model,
            "up_scaler": up_scaler,
            "down_model": down_model,
            "down_scaler": down_scaler,
            "model_type": "dual_logreg",
            "feature_names": FEATURE_NAMES,
            "training_samples": len(X_all),
            "base_rate": float(y_all.mean()),
        }, f)

    print(f"\nSaved dual model to {output_path}")


if __name__ == "__main__":
    main()
