#!/usr/bin/env python3
"""Train strike-relative model — predicts the actual Kalshi question.

Instead of "will next candle be green?" (mean-reversion), this answers:
"Given price is at X relative to strike at minute 5, will it close above strike?"

Key difference: adds distance_from_strike as a feature, which transforms the
model from mean-reversion (predict bounce) to continuation (predict stay-on-side).

Walk-forward: train on days 60-179, test on days 0-59 (out-of-sample).

Usage:
    ./venv/bin/python scripts/retrain_strike_relative.py [--days 179]
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

# Features: 15 original + distance_from_strike
# distance_from_strike is the KEY new feature — tells the model WHERE price
# is relative to the strike at the moment of prediction
FEATURE_NAMES = [
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
    "vol_ratio", "adx",
    "rsi_1h", "rsi_4h",
    "distance_from_strike",  # (price_at_min5 - strike) / ATR
]


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
                since += batch_size * candle_ms; time.sleep(0.5); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.5)
        except Exception as e:
            print(f"    Warning: {e}")
            since += batch_size * candle_ms; time.sleep(2)
    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def extract_strike_features(df_15m, df_5m, df_1h, df_4h):
    """Extract features with strike-relative label and distance.

    For each 15m candle (the "window"):
    - Strike = candle open
    - Price at minute 5 = corresponding 5m candle close
    - distance_from_strike = (price_at_min5 - strike) / ATR
    - Label = candle close >= candle open (= close >= strike)
    - Features from the PREVIOUS completed 15m candle + distance
    """
    df = add_indicators(df_15m.copy())
    df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
    df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

    pct = df["close"].pct_change()
    df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
    df["ema_slope"] = df["ema_12"].pct_change(3) * 100
    df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
    df["hourly_return"] = df["close"].pct_change(4) * 100

    rows = []
    for i in range(50, len(df) - 1):
        t = df.index[i]  # this is the current 15m window start
        window = df.iloc[i]  # the current 15m candle
        prev = df.iloc[i - 1]  # previous completed 15m candle (features source)

        # Strike = this window's open
        strike = float(window["open"])
        if strike <= 0:
            continue

        # ATR from previous candle
        atr = float(prev.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            continue

        # Find 5m candle at minute 5 of this window
        # The 5m candle that starts at window_start and closes at window_start + 5min
        min5_start = t
        min5_end = t + pd.Timedelta(minutes=5)
        if df_5m is not None and not df_5m.empty:
            mask = (df_5m.index >= min5_start) & (df_5m.index < min5_end)
            if mask.sum() > 0:
                price_at_min5 = float(df_5m[mask].iloc[-1]["close"])
            else:
                # No exact 5m candle — try nearest before
                before = df_5m[df_5m.index <= min5_end]
                if len(before) > 0 and (min5_end - before.index[-1]).total_seconds() < 600:
                    price_at_min5 = float(before.iloc[-1]["close"])
                else:
                    continue
        else:
            continue

        # Distance from strike at minute 5
        distance_from_strike = (price_at_min5 - strike) / atr

        # Label: did this window close >= strike?
        window_close = float(window["close"])
        label = 1 if window_close >= strike else 0

        # Features from PREVIOUS completed candle (i-1)
        r = prev
        sma_val = float(r.get("sma_20", 0))
        adx_val = float(r.get("adx", 20))
        close_val = float(r.get("close", 0))
        trend_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0

        pve = float(r.get("price_vs_ema", 0))
        hr = float(r.get("hourly_return", 0))
        if pd.isna(pve) or np.isinf(pve): pve = 0
        if pd.isna(hr) or np.isinf(hr): hr = 0

        feat = {
            "macd_15m": float(r.get("macd_hist", 0)),
            "norm_return": float(r.get("norm_return", 0)) if pd.notna(r.get("norm_return")) else 0,
            "ema_slope": float(r.get("ema_slope", 0)) if pd.notna(r.get("ema_slope")) else 0,
            "roc_5": float(r.get("roc_5", 0)),
            "macd_1h": 0.0,
            "price_vs_ema": pve,
            "hourly_return": hr,
            "trend_direction": adx_val * trend_sign,
            "vol_ratio": float(r.get("vol_ratio", 1)) if pd.notna(r.get("vol_ratio")) else 1,
            "adx": adx_val,
            "rsi_1h": 50.0,
            "rsi_4h": 50.0,
            "distance_from_strike": distance_from_strike,
        }

        # 1h context
        if df_1h_ind is not None:
            mask = df_1h_ind.index <= t
            if mask.sum() >= 20:
                r1h = df_1h_ind.loc[mask].iloc[-1]
                feat["rsi_1h"] = float(r1h.get("rsi", 50))
                feat["macd_1h"] = float(r1h.get("macd_hist", 0))

        # 4h context
        if df_4h_ind is not None:
            mask = df_4h_ind.index <= t
            if mask.sum() >= 10:
                feat["rsi_4h"] = float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50))

        if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
            continue

        rows.append({
            **feat,
            "label": label,
            "ts": t,
            "atr": atr,
            "close": window_close,
            "strike": strike,
            "price_at_min5": price_at_min5,
            "asset": "",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=179)
    parser.add_argument("--output", default="models/knn_kalshi.pkl")
    parser.add_argument("--test-days", type=int, default=59)
    args = parser.parse_args()

    now_ms = int(time.time() * 1000)
    train_end_ms = now_ms - args.test_days * 86400 * 1000
    train_end_ts = pd.Timestamp(train_end_ms, unit="ms").tz_localize(None)

    print("=" * 80)
    print("STRIKE-RELATIVE MODEL — Predicts the actual Kalshi question")
    print(f"'Will price close above strike?' not 'Will candle be green?'")
    print(f"TRAIN: days {args.test_days}-{args.days} | TEST: days 0-{args.test_days}")
    print(f"Train ends: {train_end_ts.strftime('%Y-%m-%d')}")
    print(f"Features: {len(FEATURE_NAMES)} (includes distance_from_strike)")
    print("=" * 80)

    fetcher = DataFetcher()
    all_dfs = []
    five_m_data = {}

    print("\nFetching data (15m + 5m + 1h + 4h)...")
    for asset_name, symbol in ASSETS.items():
        print(f"  {asset_name}...", end=" ", flush=True)
        extra = 30
        df_15m = fetch_candles(fetcher, symbol, "15m", args.days + extra)
        df_5m = fetch_candles(fetcher, symbol, "5m", args.days + extra)
        df_1h = fetch_candles(fetcher, symbol, "1h", args.days + extra)
        df_4h = fetch_candles(fetcher, symbol, "4h", args.days + extra)

        if df_15m.empty or df_5m.empty:
            print("SKIP"); continue

        five_m_data[asset_name] = df_5m

        features_df = extract_strike_features(df_15m, df_5m, df_1h, df_4h)
        features_df["asset"] = asset_name
        print(f"15m={len(df_15m)} 5m={len(df_5m)} → {len(features_df)} samples")
        all_dfs.append(features_df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(df_all)} | Base rate (close>=strike): {df_all['label'].mean():.1%}")

    # Walk-forward split
    train_end_np = np.datetime64(train_end_ts)
    df_train = df_all[df_all["ts"] < train_end_np]
    df_test = df_all[df_all["ts"] >= train_end_np]
    print(f"Train: {len(df_train)} ({df_train['label'].mean():.1%} above strike)")
    print(f"Test:  {len(df_test)} ({df_test['label'].mean():.1%} above strike)")

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    X_train = df_train[FEATURE_NAMES].values
    X_test = df_test[FEATURE_NAMES].values

    # === Train ===
    print(f"\n{'=' * 80}")
    print("TRAINING")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    high_conf = (train_probs >= 0.55) | (train_probs <= 0.45)
    train_wr = (np.where(train_probs[high_conf] > 0.5, 1, 0) == y_train[high_conf]).mean() * 100
    yes_n = (train_probs >= 0.55).sum()
    no_n = (train_probs <= 0.45).sum()
    skip_n = len(train_probs) - yes_n - no_n
    print(f"Train WR: {train_wr:.1f}% ({yes_n}Y/{no_n}N/{skip_n}SKIP)")

    print(f"\nFeature coefficients:")
    for name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        direction = "ABOVE" if coef > 0 else "BELOW"
        print(f"  {name:<22}: {coef:>+8.4f} ({direction})")

    # === Test ===
    print(f"\n{'=' * 80}")
    print("OUT-OF-SAMPLE TEST")
    print(f"{'=' * 80}")

    test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]

    # Results at various thresholds
    print(f"\n{'Thresh':>6} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 70)

    best_pnl = -999
    best_thresh = 0

    for thresh in [53, 55, 57, 60]:
        wins, losses = 0, 0
        yes_w, yes_l, no_w, no_l = 0, 0, 0, 0

        for i in range(len(y_test)):
            pct = int(test_probs[i] * 100)
            if pct >= thresh:
                side = "YES"
            elif pct <= (100 - thresh):
                side = "NO"
            else:
                continue

            won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
            if won:
                wins += 1
                if side == "YES": yes_w += 1
                else: no_w += 1
            else:
                losses += 1
                if side == "YES": yes_l += 1
                else: no_l += 1

        total = wins + losses
        if total < 20: continue
        wr = wins / total * 100
        pnl = wins * 0.50 - losses * 0.50
        yn = yes_w + yes_l
        nn = no_w + no_l
        ywr = yes_w / yn * 100 if yn > 0 else 0
        nwr = no_w / nn * 100 if nn > 0 else 0
        ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"

        if pnl > best_pnl:
            best_pnl = pnl
            best_thresh = thresh

        print(f"{thresh:>6} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")

    # === Add TEK confluence ===
    print(f"\n{'=' * 80}")
    print("WITH TEK CONFLUENCE")
    print(f"{'=' * 80}")

    from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
    predictor = KalshiPredictorV3()

    # Compute TEK scores — we already have distance and ATR in the test data
    tek_scores_yes = []
    tek_scores_no = []
    for _, row in df_test.iterrows():
        dist = row["distance_from_strike"]
        base_prob = predictor._lookup_probability(dist, 10)
        tek_scores_yes.append(int(base_prob * 100))
        tek_scores_no.append(100 - int(base_prob * 100))

    print(f"\n{'Thresh':>6} {'TEK':>4} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 75)

    best_pnl_tek = -999
    best_config_tek = ""

    for thresh in [53, 55, 57, 60]:
        for tek_t in [0, 30, 50]:
            wins, losses = 0, 0
            yes_w, yes_l, no_w, no_l = 0, 0, 0, 0

            for i in range(len(y_test)):
                pct = int(test_probs[i] * 100)
                if pct >= thresh:
                    side = "YES"
                elif pct <= (100 - thresh):
                    side = "NO"
                else:
                    continue

                if tek_t > 0:
                    tek = tek_scores_yes[i] if side == "YES" else tek_scores_no[i]
                    if tek < tek_t:
                        continue

                won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
                if won:
                    wins += 1
                    if side == "YES": yes_w += 1
                    else: no_w += 1
                else:
                    losses += 1
                    if side == "YES": yes_l += 1
                    else: no_l += 1

            total = wins + losses
            if total < 20: continue
            wr = wins / total * 100
            pnl = wins * 0.50 - losses * 0.50
            yn = yes_w + yes_l
            nn = no_w + no_l
            ywr = yes_w / yn * 100 if yn > 0 else 0
            nwr = no_w / nn * 100 if nn > 0 else 0
            ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"

            if pnl > best_pnl_tek:
                best_pnl_tek = pnl
                best_config_tek = f">={thresh} TEK>={tek_t}"

            tek_label = "OFF" if tek_t == 0 else str(tek_t)
            print(f"{thresh:>6} {tek_label:>4} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")
        print()

    print(f"Best config: {best_config_tek} (${best_pnl_tek:+.0f})")

    # === Simulate today ===
    print(f"\n{'=' * 80}")
    print("TODAY SIMULATION (8am CT / 13:00 UTC to now)")
    print(f"{'=' * 80}")

    start_utc = pd.Timestamp("2026-03-30 13:00", tz=None)
    today = df_test[df_test["ts"] >= np.datetime64(start_utc)]

    if len(today) > 0:
        today_probs = model.predict_proba(scaler.transform(today[FEATURE_NAMES].values))[:, 1]
        w, l = 0, 0
        y_w, y_l, n_w, n_l = 0, 0, 0, 0

        for i in range(len(today)):
            pct = int(today_probs[i] * 100)
            if pct >= 55:
                side = "YES"
            elif pct <= 45:
                side = "NO"
            else:
                continue

            actual = today.iloc[i]["label"]
            won = (side == "YES" and actual == 1) or (side == "NO" and actual == 0)
            dist = today.iloc[i]["distance_from_strike"]

            if won:
                w += 1
                if side == "YES": y_w += 1
                else: n_w += 1
            else:
                l += 1
                if side == "YES": y_l += 1
                else: n_l += 1

            ct = (pd.Timestamp(today.iloc[i]["ts"]) - pd.Timedelta(hours=5)).strftime("%I:%M %p")
            marker = "✓" if won else "✗"
            asset = today.iloc[i]["asset"]
            print(f"  {asset:<4} {ct:<10} {side:<4} prob={today_probs[i]:.2f} dist={dist:+.2f} → {'ABOVE' if actual==1 else 'BELOW'} {marker}")

        total = w + l
        if total > 0:
            wr = w / total * 100
            print(f"\n  Today: {total} bets | {wr:.1f}% WR ({w}W/{l}L) | {y_w+y_l}Y/{n_w+n_l}N")
    else:
        print("  No test data for today (may be in training period)")

    # === Save ===
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "knn": model,
            "scaler": scaler,
            "model_type": "strike_relative",
            "feature_names": FEATURE_NAMES,
            "training_samples": len(df_train),
            "base_rate": float(y_train.mean()),
            "train_wr": train_wr,
            "train_end": str(train_end_ts),
        }, f)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
