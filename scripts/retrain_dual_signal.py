#!/usr/bin/env python3
"""Build dual-signal prediction system: Trend model + Conviction model.

Trend model (direction): Predicts which way price will go using momentum
and flow features. NO RSI — avoids mean-reversion bias. Answers: "YES or NO?"

Conviction model (confidence): Uses RSI/BB/volume mean-reversion features
to gauge how confident we should be. Answers: "How strong is this signal?"

Both models must agree for a bet. Trend picks direction, conviction confirms.

Walk-forward: train on days 60-179, test on days 0-59 (out-of-sample).

Usage:
    ./venv/bin/python scripts/retrain_dual_signal.py [--days 179]
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

# Trend model features — momentum/flow, NO RSI
# These predict direction without mean-reversion bias
TREND_FEATURES = [
    "macd_15m",         # MACD histogram — momentum direction
    "norm_return",      # Normalized return — unusual move detector
    "ema_slope",        # EMA slope — short-term trend direction
    "roc_5",            # Rate of change — 5-bar momentum
    "macd_1h",          # 1h MACD — higher timeframe momentum
    "price_vs_ema",     # Distance from SMA in ATR units — trend position
    "hourly_return",    # 1h price change % — recent direction
    "trend_direction",  # ADX * sign(close-SMA) — trend strength + direction
    "vol_ratio",        # Volume vs average — is the move backed by volume?
]

# Conviction model features — mean-reversion / confidence gauge
# These tell us HOW CONFIDENT to be, not which direction
CONVICTION_FEATURES = [
    "rsi_15m",          # RSI — oversold/overbought
    "stochrsi_15m",     # Stochastic RSI — extreme readings
    "bb_position",      # Bollinger Band position — 0=lower, 1=upper
    "adx",              # ADX — trend strength (not direction)
    "rsi_1h",           # 1h RSI — higher TF momentum
    "rsi_4h",           # 4h RSI — macro context
    "norm_return",      # Shared — unusual returns boost conviction
    "vol_ratio",        # Shared — volume confirms conviction
]

# Combined (for the single model baseline comparison)
ALL_FEATURES = [
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


def extract_all_features(df_15m, df_1h, df_4h):
    """Extract ALL 15 features, labels, timestamps, and metadata."""
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

    rows = []
    for i in range(200, len(df) - 1):
        t = df.index[i]
        r = df.iloc[i]

        # Build feature dict with all 15 features
        feat = {}
        feat["rsi_15m"] = float(r.get("rsi", np.nan))
        feat["stochrsi_15m"] = float(r.get("stochrsi_k", np.nan))
        feat["macd_15m"] = float(r.get("macd_hist", np.nan))
        feat["norm_return"] = float(r.get("norm_return", np.nan))
        feat["vol_ratio"] = float(r.get("vol_ratio", np.nan))
        feat["bb_position"] = float(r.get("bb_position", np.nan))
        feat["ema_slope"] = float(r.get("ema_slope", np.nan))
        feat["adx"] = float(r.get("adx", np.nan))
        feat["roc_5"] = float(r.get("roc_5", np.nan))

        if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
            continue

        # 1h context
        if df_1h_ind is not None:
            mask = df_1h_ind.index <= t
            if mask.sum() >= 20:
                r1h = df_1h_ind.loc[mask].iloc[-1]
                feat["rsi_1h"] = float(r1h.get("rsi", 50))
                feat["macd_1h"] = float(r1h.get("macd_hist", 0))
            else:
                continue
        else:
            feat["rsi_1h"] = 50.0
            feat["macd_1h"] = 0.0

        # 4h context
        if df_4h_ind is not None:
            mask = df_4h_ind.index <= t
            if mask.sum() >= 10:
                feat["rsi_4h"] = float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50))
            else:
                continue
        else:
            feat["rsi_4h"] = 50.0

        # Trend features
        pve = float(r.get("price_vs_ema", np.nan))
        hr = float(r.get("hourly_return", np.nan))
        adx_val = float(r.get("adx", 20))
        close_val = float(r.get("close", 0))
        sma_val = float(r.get("sma_20", 0))

        if pd.isna(pve) or np.isinf(pve): pve = 0
        if pd.isna(hr) or np.isinf(hr): hr = 0

        trend_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
        feat["price_vs_ema"] = pve
        feat["hourly_return"] = hr
        feat["trend_direction"] = adx_val * trend_sign

        if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
            continue

        rows.append({
            **feat,
            "label": int(r["next_up"]),
            "ts": t,
            "atr": float(r.get("atr", 0)) if pd.notna(r.get("atr")) else 0,
            "close": close_val,
        })

    return pd.DataFrame(rows)


def train_model(X, y, name):
    """Train a LogReg model and return (model, scaler, metrics)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)[:, 1]
    high_conf = (probs >= 0.55) | (probs <= 0.45)
    preds = np.where(probs[high_conf] > 0.5, 1, 0)
    wr = (preds == y[high_conf]).mean() * 100 if high_conf.sum() > 0 else 0
    yes_n = (probs >= 0.55).sum()
    no_n = (probs <= 0.45).sum()
    skip_n = len(probs) - yes_n - no_n

    return model, scaler, {"wr": wr, "yes": yes_n, "no": no_n, "skip": skip_n}


def evaluate_model(model, scaler, X, y, threshold=0.55):
    """Evaluate model on test data. Returns (signals_df)."""
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        pct = int(prob * 100)
        if pct >= threshold * 100:
            side = "YES"
            score = pct
        elif pct <= (100 - threshold * 100):
            side = "NO"
            score = 100 - pct
        else:
            side = "SKIP"
            score = 0

        won = (side == "YES" and y[i] == 1) or (side == "NO" and y[i] == 0)
        results.append({"side": side, "score": score, "prob": prob, "won": won, "actual": y[i]})

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Build dual-signal prediction system")
    parser.add_argument("--days", type=int, default=179)
    parser.add_argument("--output", default="models/knn_kalshi.pkl")
    parser.add_argument("--test-days", type=int, default=59)
    args = parser.parse_args()

    now_ms = int(time.time() * 1000)
    train_end_ms = now_ms - args.test_days * 86400 * 1000
    train_end_ts = pd.Timestamp(train_end_ms, unit="ms").tz_localize(None)

    print("=" * 80)
    print("DUAL-SIGNAL SYSTEM: Trend (direction) + Conviction (confidence)")
    print(f"TRAIN: days {args.test_days}-{args.days} | TEST: days 0-{args.test_days}")
    print(f"Train ends: {train_end_ts.strftime('%Y-%m-%d')}")
    print("=" * 80)
    print(f"\n  Trend features ({len(TREND_FEATURES)}):     {', '.join(TREND_FEATURES)}")
    print(f"  Conviction features ({len(CONVICTION_FEATURES)}): {', '.join(CONVICTION_FEATURES)}")

    # === Fetch data ===
    print("\nFetching data...")
    fetcher = DataFetcher()
    all_dfs = []

    for asset_name, symbol in ASSETS.items():
        print(f"  {asset_name}...", end=" ", flush=True)
        extra = 30
        df_15m = fetch_candles(fetcher, symbol, "15m", args.days + extra)
        df_1h = fetch_candles(fetcher, symbol, "1h", args.days + extra)
        df_4h = fetch_candles(fetcher, symbol, "4h", args.days + extra)
        if df_15m.empty:
            print("SKIP"); continue

        # Also fetch 5m for TEK scoring
        df_5m = fetch_candles(fetcher, symbol, "5m", args.test_days + 5)

        features_df = extract_all_features(df_15m, df_1h, df_4h)
        features_df["asset"] = asset_name
        features_df["_5m"] = [df_5m] * len(features_df)  # attach 5m ref
        print(f"{len(features_df)} samples")
        all_dfs.append(features_df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    # Drop the 5m column after noting we have it per asset
    five_m_data = {}
    for asset_name in ASSETS:
        subset = df_all[df_all["asset"] == asset_name]
        if len(subset) > 0 and len(subset.iloc[0]["_5m"]) > 0:
            five_m_data[asset_name] = subset.iloc[0]["_5m"]
    df_all = df_all.drop(columns=["_5m"])

    print(f"\nTotal samples: {len(df_all)} | Base rate UP: {df_all['label'].mean():.1%}")

    # === Walk-forward split ===
    train_end_np = np.datetime64(train_end_ts)
    train_mask = df_all["ts"] < train_end_np
    test_mask = df_all["ts"] >= train_end_np

    df_train = df_all[train_mask]
    df_test = df_all[test_mask]
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    # === Train 3 models ===
    print(f"\n{'=' * 80}")
    print("TRAINING MODELS")
    print(f"{'=' * 80}")

    # Model A: Trend (direction) — no RSI
    X_trend_train = df_train[TREND_FEATURES].values
    X_trend_test = df_test[TREND_FEATURES].values
    trend_model, trend_scaler, trend_metrics = train_model(X_trend_train, y_train, "Trend")
    print(f"\n  TREND model (no RSI): {trend_metrics['wr']:.1f}% WR "
          f"({trend_metrics['yes']}Y/{trend_metrics['no']}N/{trend_metrics['skip']}SKIP)")
    print(f"  Top features:")
    for name, coef in sorted(zip(TREND_FEATURES, trend_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<18}: {coef:>+8.4f}")

    # Model B: Conviction (RSI-based)
    X_conv_train = df_train[CONVICTION_FEATURES].values
    X_conv_test = df_test[CONVICTION_FEATURES].values
    conv_model, conv_scaler, conv_metrics = train_model(X_conv_train, y_train, "Conviction")
    print(f"\n  CONVICTION model (RSI-based): {conv_metrics['wr']:.1f}% WR "
          f"({conv_metrics['yes']}Y/{conv_metrics['no']}N/{conv_metrics['skip']}SKIP)")
    print(f"  Top features:")
    for name, coef in sorted(zip(CONVICTION_FEATURES, conv_model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:<18}: {coef:>+8.4f}")

    # Baseline: Single combined model (current approach)
    X_all_train = df_train[ALL_FEATURES].values
    X_all_test = df_test[ALL_FEATURES].values
    single_model, single_scaler, single_metrics = train_model(X_all_train, y_train, "Single")
    print(f"\n  SINGLE model (baseline): {single_metrics['wr']:.1f}% WR "
          f"({single_metrics['yes']}Y/{single_metrics['no']}N/{single_metrics['skip']}SKIP)")

    # === Evaluate on test data ===
    print(f"\n{'=' * 80}")
    print("OUT-OF-SAMPLE TEST RESULTS")
    print(f"{'=' * 80}")

    trend_sigs = evaluate_model(trend_model, trend_scaler, X_trend_test, y_test)
    conv_sigs = evaluate_model(conv_model, conv_scaler, X_conv_test, y_test)
    single_sigs = evaluate_model(single_model, single_scaler, X_all_test, y_test)

    # Individual model results
    for label, sigs in [("Trend only", trend_sigs), ("Conviction only", conv_sigs), ("Single (baseline)", single_sigs)]:
        bets = sigs[sigs["side"] != "SKIP"]
        if len(bets) == 0:
            print(f"\n  {label}: No bets")
            continue
        wr = bets["won"].mean() * 100
        yes_n = len(bets[bets["side"] == "YES"])
        no_n = len(bets[bets["side"] == "NO"])
        yes_wr = bets[bets["side"] == "YES"]["won"].mean() * 100 if yes_n > 0 else 0
        no_wr = bets[bets["side"] == "NO"]["won"].mean() * 100 if no_n > 0 else 0
        pnl = bets["won"].sum() * 0.50 - (~bets["won"]).sum() * 0.50
        print(f"\n  {label}:")
        print(f"    {len(bets)} bets | {wr:.1f}% WR | ${pnl:+.0f} P&L")
        print(f"    {yes_n}Y ({yes_wr:.0f}% WR) / {no_n}N ({no_wr:.0f}% WR) | Ratio: {yes_n/no_n:.1f}:1" if no_n > 0 else f"    {yes_n}Y ({yes_wr:.0f}% WR) / {no_n}N")

    # === Dual-signal: Trend picks direction, conviction confirms ===
    print(f"\n{'=' * 80}")
    print("DUAL-SIGNAL: Trend direction + Conviction confirmation")
    print(f"{'=' * 80}")

    # Get probabilities from both models
    trend_probs = trend_model.predict_proba(trend_scaler.transform(X_trend_test))[:, 1]
    conv_probs = conv_model.predict_proba(conv_scaler.transform(X_conv_test))[:, 1]

    print(f"\n{'Trend':>5} {'Conv':>5} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 75)

    best_pnl = -999
    best_config = ""

    for trend_t in [53, 55, 57]:
        for conv_t in [53, 55, 57, 60]:
            wins, losses = 0, 0
            yes_w, yes_l, no_w, no_l = 0, 0, 0, 0

            for i in range(len(y_test)):
                t_pct = int(trend_probs[i] * 100)
                c_pct = int(conv_probs[i] * 100)

                # Trend model picks direction
                if t_pct >= trend_t:
                    trend_side = "YES"
                    trend_score = t_pct
                elif t_pct <= (100 - trend_t):
                    trend_side = "NO"
                    trend_score = 100 - t_pct
                else:
                    continue  # Trend says SKIP

                # Conviction model must AGREE on direction
                if trend_side == "YES" and c_pct >= conv_t:
                    pass  # Conviction confirms YES
                elif trend_side == "NO" and c_pct <= (100 - conv_t):
                    pass  # Conviction confirms NO
                else:
                    continue  # Conviction disagrees — SKIP

                # Both agree
                won = (trend_side == "YES" and y_test[i] == 1) or \
                      (trend_side == "NO" and y_test[i] == 0)

                if won:
                    wins += 1
                    if trend_side == "YES": yes_w += 1
                    else: no_w += 1
                else:
                    losses += 1
                    if trend_side == "YES": yes_l += 1
                    else: no_l += 1

            total = wins + losses
            if total < 50:
                continue
            wr = wins / total * 100
            pnl = wins * 0.50 - losses * 0.50
            yn = yes_w + yes_l
            nn = no_w + no_l
            ywr = yes_w / yn * 100 if yn > 0 else 0
            nwr = no_w / nn * 100 if nn > 0 else 0
            ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"

            if pnl > best_pnl:
                best_pnl = pnl
                best_config = f"T>={trend_t} C>={conv_t}"

            print(f"{trend_t:>5} {conv_t:>5} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")

        print()

    print(f"Best P&L config: {best_config} (${best_pnl:+.0f})")

    # === Add TEK confluence ===
    print(f"\n{'=' * 80}")
    print("DUAL-SIGNAL + TEK CONFLUENCE")
    print(f"{'=' * 80}")

    from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
    predictor = KalshiPredictorV3()

    # Compute TEK scores using 5m data
    print("Computing TEK scores...")
    tek_scores_yes = []
    tek_scores_no = []

    for i in range(len(df_test)):
        row = df_test.iloc[i]
        atr = row["atr"]
        close = row["close"]
        asset = row["asset"]
        ts = pd.Timestamp(row["ts"])

        window_start = ts + pd.Timedelta(minutes=15)
        strike = close

        price_at_min5 = None
        if asset in five_m_data and not five_m_data[asset].empty:
            df_5m = five_m_data[asset]
            min5_time = window_start + pd.Timedelta(minutes=5)
            mask = df_5m.index <= min5_time
            if mask.sum() > 0:
                time_diff = (min5_time - df_5m[mask].index[-1]).total_seconds()
                if time_diff < 600:
                    price_at_min5 = float(df_5m[mask].iloc[-1]["close"])

        if price_at_min5 is not None and atr > 0 and strike > 0:
            dist = (price_at_min5 - strike) / atr
            base_prob = predictor._lookup_probability(dist, 10)
            tek_yes = int(base_prob * 100)
            tek_no = 100 - tek_yes
        else:
            tek_yes = 50
            tek_no = 50

        tek_scores_yes.append(tek_yes)
        tek_scores_no.append(tek_no)

    print(f"\n{'Trend':>5} {'Conv':>5} {'TEK':>4} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 80)

    best_pnl_tek = -999
    best_config_tek = ""

    for trend_t in [53, 55, 57]:
        for conv_t in [53, 55, 57]:
            for tek_t in [0, 30, 50]:
                wins, losses = 0, 0
                yes_w, yes_l, no_w, no_l = 0, 0, 0, 0

                for i in range(len(y_test)):
                    t_pct = int(trend_probs[i] * 100)
                    c_pct = int(conv_probs[i] * 100)

                    if t_pct >= trend_t:
                        trend_side = "YES"
                    elif t_pct <= (100 - trend_t):
                        trend_side = "NO"
                    else:
                        continue

                    # Conviction agrees?
                    if trend_side == "YES" and c_pct >= conv_t:
                        pass
                    elif trend_side == "NO" and c_pct <= (100 - conv_t):
                        pass
                    else:
                        continue

                    # TEK agrees?
                    if tek_t > 0:
                        tek_score = tek_scores_yes[i] if trend_side == "YES" else tek_scores_no[i]
                        if tek_score < tek_t:
                            continue

                    won = (trend_side == "YES" and y_test[i] == 1) or \
                          (trend_side == "NO" and y_test[i] == 0)

                    if won:
                        wins += 1
                        if trend_side == "YES": yes_w += 1
                        else: no_w += 1
                    else:
                        losses += 1
                        if trend_side == "YES": yes_l += 1
                        else: no_l += 1

                total = wins + losses
                if total < 50:
                    continue
                wr = wins / total * 100
                pnl = wins * 0.50 - losses * 0.50
                yn = yes_w + yes_l
                nn = no_w + no_l
                ywr = yes_w / yn * 100 if yn > 0 else 0
                nwr = no_w / nn * 100 if nn > 0 else 0
                ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"

                if pnl > best_pnl_tek:
                    best_pnl_tek = pnl
                    best_config_tek = f"T>={trend_t} C>={conv_t} TEK>={tek_t}"

                tek_label = "OFF" if tek_t == 0 else str(tek_t)
                print(f"{trend_t:>5} {conv_t:>5} {tek_label:>4} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")

            print()

    print(f"Best config: {best_config_tek} (${best_pnl_tek:+.0f})")

    # === Save dual model ===
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            # Trend model (direction)
            "trend_model": trend_model,
            "trend_scaler": trend_scaler,
            "trend_features": TREND_FEATURES,
            # Conviction model (confidence)
            "conv_model": conv_model,
            "conv_scaler": conv_scaler,
            "conv_features": CONVICTION_FEATURES,
            # Backward compat: "knn" and "scaler" point to trend model
            # so predict_knn can use it as primary signal
            "knn": trend_model,
            "scaler": trend_scaler,
            "model_type": "dual_trend_conviction",
            "feature_names": ALL_FEATURES,
            "training_samples": len(df_train),
            "base_rate": float(y_train.mean()),
            "train_end": str(train_end_ts),
        }, f)

    print(f"\nSaved dual model to {output_path}")
    print(f"  Trend model: {len(TREND_FEATURES)} features (no RSI)")
    print(f"  Conviction model: {len(CONVICTION_FEATURES)} features (RSI-based)")


if __name__ == "__main__":
    main()
