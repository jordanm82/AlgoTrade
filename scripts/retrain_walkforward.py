#!/usr/bin/env python3
"""Retrain LogReg model with walk-forward validation.

Walk-forward split:
  TRAIN: days 60-179 (120 days, oldest data)
  TEST:  days 0-59  (59 days, most recent, out-of-sample)

Uses 15 features (12 original + 3 trend) from completed 15m candles.
Evaluates with and without TEK (probability table) confluence.
Saves the walk-forward model as production model.

Usage:
    ./venv/bin/python scripts/retrain_walkforward.py [--days 179] [--output models/knn_kalshi.pkl]
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
    """Fetch N days of candles via paginated requests."""
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
                since += batch_size * candle_ms
                time.sleep(0.5)
                continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.5)
        except Exception as e:
            print(f"    Warning: {e}")
            since += batch_size * candle_ms
            time.sleep(2)

    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def extract_all_features(df_15m, df_1h, df_4h):
    """Extract feature vectors, labels, and timestamps from historical data.

    Uses the EXACT same logic as build_model.py — completed 15m candle features.
    Returns (features, labels, timestamps) arrays.
    """
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

    features = []
    labels = []
    timestamps = []
    atr_vals = []
    close_vals = []

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
                vals.append(float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50)))
            else:
                continue
        else:
            vals.append(50.0)

        # Trend features
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

        vals.append(pve)
        vals.append(hr)
        vals.append(adx_val * trend_sign)

        if any(pd.isna(v) or np.isinf(v) for v in vals):
            continue

        features.append(vals)
        labels.append(int(r["next_up"]))
        timestamps.append(t)
        atr_val = float(r.get("atr", 0))
        atr_vals.append(atr_val if not pd.isna(atr_val) else 0)
        close_vals.append(close_val)

    return (np.array(features), np.array(labels),
            np.array(timestamps), np.array(atr_vals), np.array(close_vals))


def compute_tek_score(predictor, price_at_min5, strike, atr, side):
    """Compute TEK (probability table) score using actual price at minute 5.

    In the daemon, at minute 5:
    - strike = window open (from Kalshi market)
    - current_price = Coinbase price at that moment
    - distance_atr = (current_price - strike) / ATR
    - minutes_remaining = 10 (15 - 5 = 10 minutes left)

    We simulate this using 5m data: the 5m candle that closes at minute 5
    gives us the actual price at that point.
    """
    if atr <= 0 or strike <= 0:
        return 50

    distance_atr = (price_at_min5 - strike) / atr
    base_prob = predictor._lookup_probability(distance_atr, 10)
    prob_pct = int(base_prob * 100)

    if side == "YES":
        return prob_pct
    else:
        return 100 - prob_pct


def main():
    parser = argparse.ArgumentParser(description="Retrain LogReg with walk-forward validation")
    parser.add_argument("--days", type=int, default=179, help="Total days of data")
    parser.add_argument("--output", default="models/knn_kalshi.pkl", help="Output model path")
    parser.add_argument("--train-days", type=int, default=120, help="Days for training (oldest)")
    parser.add_argument("--test-days", type=int, default=59, help="Days for testing (newest)")
    args = parser.parse_args()

    now_ms = int(time.time() * 1000)
    train_end_ms = now_ms - args.test_days * 86400 * 1000
    train_end_ts = pd.Timestamp(train_end_ms, unit="ms").tz_localize(None)

    print("=" * 80)
    print("WALK-FORWARD RETRAIN — LogReg 15-feature model")
    print(f"TRAIN: days {args.test_days}-{args.days} ({args.train_days} days)")
    print(f"TEST:  days 0-{args.test_days} ({args.test_days} days, out-of-sample)")
    print(f"Train ends: {train_end_ts.strftime('%Y-%m-%d')}")
    print(f"Features: {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES[-3:])})")
    print("=" * 80)

    # === Fetch data ===
    print("\nFetching data...")
    fetcher = DataFetcher()
    all_X, all_y, all_ts, all_atr, all_close = [], [], [], [], []
    all_5m_data = {}  # for TEK scoring

    for asset_name, symbol in ASSETS.items():
        print(f"  {asset_name}...", end=" ", flush=True)
        # Fetch extra history for indicators to warm up
        extra_days = 30
        df_15m = fetch_candles(fetcher, symbol, "15m", args.days + extra_days)
        df_1h = fetch_candles(fetcher, symbol, "1h", args.days + extra_days)
        df_4h = fetch_candles(fetcher, symbol, "4h", args.days + extra_days)

        if df_15m.empty:
            print("SKIP (no data)")
            continue

        # Fetch 5m data for TEK scoring (just test period + some buffer)
        df_5m = fetch_candles(fetcher, symbol, "5m", args.test_days + 5)
        all_5m_data[asset_name] = df_5m
        print(f"5m={len(df_5m)}", end=" ", flush=True)

        X, y, ts, atr, close = extract_all_features(df_15m, df_1h, df_4h)
        print(f"15m={len(df_15m)} → {len(X)} samples")
        all_X.append(X)
        all_y.append(y)
        all_ts.append(ts)
        all_atr.append(atr)
        all_close.append(close)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    ts_all = np.concatenate(all_ts)
    atr_all = np.concatenate(all_atr)
    close_all = np.concatenate(all_close)

    print(f"\nTotal samples: {len(X_all)}")
    print(f"Base rate UP: {y_all.mean():.1%}")

    # === Walk-forward split ===
    # Convert train_end_ts to numpy datetime64 for comparison
    train_end_np = np.datetime64(train_end_ts)
    train_mask = ts_all < train_end_np
    test_mask = ts_all >= train_end_np

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    ts_test = ts_all[test_mask]
    atr_test = atr_all[test_mask]
    close_test = close_all[test_mask]

    print(f"Train: {len(X_train)} samples ({y_train.mean():.1%} UP)")
    print(f"Test:  {len(X_test)} samples ({y_test.mean():.1%} UP)")

    # === PHASE 1: Train on train set ===
    print(f"\n{'=' * 80}")
    print("PHASE 1: TRAINING")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    # In-sample metrics
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    high_conf = (train_probs >= 0.55) | (train_probs <= 0.45)
    train_preds = np.where(train_probs[high_conf] > 0.5, 1, 0)
    train_wr = (train_preds == y_train[high_conf]).mean() * 100
    yes_n = (train_probs >= 0.55).sum()
    no_n = (train_probs <= 0.45).sum()
    skip_n = len(train_probs) - yes_n - no_n
    print(f"Train WR (in-sample): {train_wr:.1f}% ({yes_n}Y/{no_n}N/{skip_n}SKIP)")

    # Feature importance
    print(f"\nTop features by |coefficient|:")
    for feat_name, coef in sorted(
        zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True
    )[:8]:
        direction = "UP" if coef > 0 else "DN"
        print(f"  {feat_name:<18}: {coef:>+8.4f} ({direction})")

    # === PHASE 2: Test on out-of-sample data ===
    print(f"\n{'=' * 80}")
    print("PHASE 2: TESTING (out-of-sample)")
    print(f"{'=' * 80}")

    X_test_scaled = scaler.transform(X_test)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Build a mapping from test timestamp -> asset name for 5m lookups
    # We need to reconstruct which asset each test sample came from
    # Do this by tracking per-asset sample ranges
    asset_ranges = {}
    offset = 0
    for asset_name in ASSETS:
        # Count how many train+test samples came from each asset
        n_samples = len(all_X[list(ASSETS.keys()).index(asset_name)])
        n_train = (ts_all[offset:offset + n_samples] < train_end_np).sum()
        n_test = n_samples - n_train
        asset_ranges[asset_name] = (offset + n_train, offset + n_samples)
        offset += n_samples

    # Build test signals with KNN scores
    test_signals = []
    test_idx_global = 0
    for i in range(len(X_test)):
        prob = test_probs[i]
        knn_pct = int(prob * 100)

        if knn_pct >= 55:
            side = "YES"
            knn_score = knn_pct
        elif knn_pct <= 45:
            side = "NO"
            knn_score = 100 - knn_pct
        else:
            continue  # SKIP

        won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)

        # Find which asset this sample belongs to
        global_idx = np.where(test_mask)[0][i]
        asset_name = None
        for a, (start, end) in asset_ranges.items():
            if start <= global_idx < end:
                asset_name = a
                break

        test_signals.append({
            "side": side,
            "knn_score": knn_score,
            "knn_pct": knn_pct,
            "won": won,
            "actual_up": y_test[i],
            "atr": atr_test[i],
            "close": close_test[i],
            "ts": ts_test[i],
            "asset": asset_name,
        })

    df_sig = pd.DataFrame(test_signals)
    yes_count = len(df_sig[df_sig["side"] == "YES"])
    no_count = len(df_sig[df_sig["side"] == "NO"])
    print(f"Test signals: {len(df_sig)} ({yes_count}Y/{no_count}N)")

    # === Compute TEK scores using 5m data ===
    print("\nComputing TEK scores (with 5m price at minute 5)...")
    from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
    predictor = KalshiPredictorV3()

    tek_scores = []
    tek_valid = 0
    tek_fallback = 0

    for _, row in df_sig.iterrows():
        asset = row["asset"]
        ts = pd.Timestamp(row["ts"])
        atr = row["atr"]
        side = row["side"]

        # The 15m candle at timestamp 'ts' predicts the NEXT candle
        # Next candle starts at ts + 15min
        window_start = ts + pd.Timedelta(minutes=15)
        strike = row["close"]  # current close ≈ next candle open ≈ strike

        # Look up 5m price at minute 5 of the next window
        # That's window_start + 5min (the 5m candle closing at minute 5)
        min5_time = window_start + pd.Timedelta(minutes=5)

        price_at_min5 = None
        if asset in all_5m_data and not all_5m_data[asset].empty:
            df_5m = all_5m_data[asset]
            # Find the 5m candle at or just before min5_time
            mask = df_5m.index <= min5_time
            if mask.sum() > 0:
                closest = df_5m[mask].iloc[-1]
                # Only use if within 5 minutes of target
                time_diff = (min5_time - df_5m[mask].index[-1]).total_seconds()
                if time_diff < 600:  # within 10 min
                    price_at_min5 = float(closest["close"])

        if price_at_min5 is not None:
            tek = compute_tek_score(predictor, price_at_min5, strike, atr, side)
            tek_valid += 1
        else:
            # Fallback: distance=0 (conservative estimate)
            tek = compute_tek_score(predictor, strike, strike, atr, side)
            tek_fallback += 1

        tek_scores.append(tek)

    df_sig["tek_score"] = tek_scores
    print(f"  TEK computed: {tek_valid} with 5m data, {tek_fallback} fallback (distance=0)")

    # === Results table: LR threshold × TEK threshold ===
    print(f"\n{'=' * 80}")
    print("RESULTS: LR (KNN) threshold × TEK threshold")
    print(f"{'=' * 80}")
    print(f"{'LR':>4} {'TEK':>4} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 75)

    best_pnl = -999
    best_config = ""

    for lr_thresh in [55, 56, 57, 58, 60]:
        for tek_thresh in [0, 30, 40, 50]:
            filtered = df_sig[
                (df_sig["knn_score"] >= lr_thresh) &
                (df_sig["tek_score"] >= tek_thresh)
            ]
            if len(filtered) < 20:
                continue

            wr = filtered["won"].mean() * 100
            wins = filtered["won"].sum()
            losses = len(filtered) - wins
            pnl = wins * 0.50 - losses * 0.50
            yn = len(filtered[filtered["side"] == "YES"])
            nn = len(filtered[filtered["side"] == "NO"])
            ywr = filtered[filtered["side"] == "YES"]["won"].mean() * 100 if yn > 0 else 0
            nwr = filtered[filtered["side"] == "NO"]["won"].mean() * 100 if nn > 0 else 0
            ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"

            marker = ""
            if pnl > best_pnl:
                best_pnl = pnl
                best_config = f"LR>={lr_thresh} TEK>={tek_thresh}"

            # Highlight no-TEK (TEK=0) rows
            tek_label = "OFF" if tek_thresh == 0 else str(tek_thresh)
            print(f"{lr_thresh:>4} {tek_label:>4} | {len(filtered):>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")

        # Separator between LR threshold groups
        if lr_thresh < 60:
            print(f"{'':>4} {'':>4} |{'':>5} {'':>6} {'':>8} |")

    print(f"\nBest P&L config: {best_config} (${best_pnl:+.0f})")

    # === Compounding simulation ===
    print(f"\n{'=' * 80}")
    print("COMPOUNDING SIMULATION (starting balance $100)")
    print("  Rules: 5% of balance per bet, max 100 contracts, max 3 bets per window")
    print(f"{'=' * 80}")

    STARTING_BALANCE = 100.0
    RISK_PCT = 0.05         # 5% of balance per bet
    MAX_CONTRACTS = 100     # Kalshi realistic cap
    MAX_PER_WINDOW = 3      # max 3 bets per 15-minute window
    ENTRY_CENTS = 50        # typical fill price

    for lr_t, tek_t in [(55, 0), (55, 30), (58, 30), (60, 50)]:
        filtered = df_sig[
            (df_sig["knn_score"] >= lr_t) &
            (df_sig["tek_score"] >= tek_t)
        ].copy()
        if len(filtered) < 20:
            continue

        # Sort chronologically for proper compounding
        filtered = filtered.sort_values("ts").reset_index(drop=True)

        balance = STARTING_BALANCE
        peak = balance
        max_dd = 0
        skipped_full = 0
        skipped_cap = 0
        bets_placed = 0

        # Group by 15-minute window to enforce max 3 per window
        # Window = floor timestamp to 15-minute boundary
        filtered["window"] = filtered["ts"].apply(
            lambda t: pd.Timestamp(t).floor("15min")
        )

        # Track equity at regular intervals (weekly)
        equity_log = [(0, balance)]
        days_elapsed = 0

        for window_ts, group in filtered.groupby("window"):
            # Take top MAX_PER_WINDOW bets by confidence (knn_score)
            window_bets = group.nlargest(MAX_PER_WINDOW, "knn_score")

            for _, row in window_bets.iterrows():
                # Size: 5% of current balance
                risk_budget = balance * RISK_PCT
                contracts = int(risk_budget / (ENTRY_CENTS / 100))
                contracts = max(1, min(contracts, MAX_CONTRACTS))
                cost = contracts * (ENTRY_CENTS / 100)

                if cost > balance:
                    skipped_full += 1
                    continue
                if contracts == MAX_CONTRACTS and risk_budget > cost * 2:
                    skipped_cap += 1  # capped but still placed

                bets_placed += 1

                if row["won"]:
                    profit = contracts * ((100 - ENTRY_CENTS) / 100)
                    balance += profit
                else:
                    balance -= cost

                # Track drawdown
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            # Log equity weekly (every ~672 windows = 7 days of 15m candles)
            day = (window_ts - filtered["window"].min()).days if hasattr(window_ts - filtered["window"].min(), 'days') else 0
            if bets_placed > 0 and bets_placed % max(1, len(filtered) // 8) < MAX_PER_WINDOW:
                equity_log.append((bets_placed, balance))

        equity_log.append((bets_placed, balance))
        wr = filtered["won"].mean() * 100
        ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        tek_label = "OFF" if tek_t == 0 else str(tek_t)

        print(f"\n  LR>={lr_t} TEK>={tek_label}:")
        print(f"    Bets placed: {bets_placed} / {len(filtered)} qualifying | WR: {wr:.1f}%")
        print(f"    Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
        print(f"    Max drawdown: {max_dd:.1f}%")
        if skipped_full > 0:
            print(f"    Skipped (insufficient balance): {skipped_full}")
        if skipped_cap > 0:
            print(f"    Hit contract cap ({MAX_CONTRACTS}): {skipped_cap} times")

        # Mini equity curve
        print(f"    Equity: ", end="")
        seen = set()
        for i, (bet_num, eq) in enumerate(equity_log):
            key = f"{eq:.0f}"
            if key in seen and i < len(equity_log) - 1:
                continue
            seen.add(key)
            if i > 0:
                print(" → ", end="")
            print(f"${eq:,.0f}", end="")
        print()

    # === Summary ===
    print(f"\n{'=' * 80}")
    print("TRAIN vs TEST COMPARISON")
    print(f"{'=' * 80}")

    # LR-only (no TEK)
    lr_only = df_sig[df_sig["knn_score"] >= 55]
    lr_wr = lr_only["won"].mean() * 100 if len(lr_only) > 0 else 0

    # LR + TEK 30
    lr_tek30 = df_sig[(df_sig["knn_score"] >= 55) & (df_sig["tek_score"] >= 30)]
    lr_tek30_wr = lr_tek30["won"].mean() * 100 if len(lr_tek30) > 0 else 0

    print(f"  Train (in-sample):        {train_wr:.1f}% WR")
    print(f"  Test LR-only (no TEK):    {lr_wr:.1f}% WR ({len(lr_only)} bets)")
    print(f"  Test LR + TEK>=30:        {lr_tek30_wr:.1f}% WR ({len(lr_tek30)} bets)")
    print(f"  LR degradation:           {lr_wr - train_wr:+.1f}pp")

    if lr_tek30_wr > lr_wr:
        print(f"  TEK adds:                 {lr_tek30_wr - lr_wr:+.1f}pp WR")
    else:
        print(f"  TEK effect:               {lr_tek30_wr - lr_wr:+.1f}pp WR (hurts)")

    # === Save model ===
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "knn": model,
            "scaler": scaler,
            "model_type": "logreg_walkforward",
            "feature_names": FEATURE_NAMES,
            "training_samples": len(X_train),
            "base_rate": float(y_train.mean()),
            "train_wr": train_wr,
            "test_wr": lr_wr,
            "test_wr_with_tek30": lr_tek30_wr,
            "train_end": str(train_end_ts),
        }, f)

    print(f"\nSaved walk-forward model to {output_path}")
    print(f"  Trained on: {len(X_train)} samples (before {train_end_ts.strftime('%Y-%m-%d')})")
    print(f"  Out-of-sample test: {lr_wr:.1f}% WR ({len(lr_only)} bets)")


if __name__ == "__main__":
    main()
