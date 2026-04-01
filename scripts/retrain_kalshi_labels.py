#!/usr/bin/env python3
"""Retrain with CORRECT labels — Kalshi settlement results, not Coinbase candle direction.

Training data:
  - LABELS: Kalshi settled market result (yes/no) — ground truth
  - STRIKE: Kalshi floor_strike — the actual strike
  - PRICE AT MIN 5: Coinbase + Bitstamp 5m average — matches live BRTI proxy
  - INDICATORS: Coinbase 15m/1h/4h candles — matches live snapshot

Walk-forward: train on oldest 70%, test on newest 30%.

Usage:
    ./venv/bin/python scripts/retrain_kalshi_labels.py
"""
import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}

FEATURE_NAMES = [
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
    "vol_ratio", "adx",
    "rsi_1h", "rsi_4h",
    "distance_from_strike",
]


def fetch_kalshi_settlements(client):
    """Fetch ALL settled K15 markets from Kalshi."""
    all_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for page in range(30):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = client._get("/trade-api/v2/markets", params)
            batch = resp.get("markets", [])
            cursor = resp.get("cursor", "")
            markets.extend(batch)
            if not batch or not cursor:
                break
        all_markets[asset] = markets
        print(f"  {asset}: {len(markets)} settled markets")
    return all_markets


def fetch_5m_history(symbol, exchange_name, days):
    """Fetch 5m candle history from a single exchange."""
    constructors = {
        "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
        "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
    }
    ex = constructors[exchange_name]()
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    candle_ms = 300000  # 5 minutes

    while since < now_ms:
        try:
            candles = ex.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            if not candles:
                since += 1000 * candle_ms
                time.sleep(0.3)
                continue
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop(columns=["timestamp"])
            all_frames.append(df)
            since = int(candles[-1][0]) + candle_ms
            time.sleep(0.3)
        except Exception as e:
            since += 500 * candle_ms
            time.sleep(1)

    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/knn_kalshi.pkl")
    args = parser.parse_args()

    print("=" * 80)
    print("RETRAIN WITH KALSHI SETTLEMENT LABELS")
    print("Labels: Kalshi result (yes/no) — NOT Coinbase candle direction")
    print("Price: Coinbase + Bitstamp at MINUTE 1 (5m candle open as proxy)")
    print("Strike: Kalshi floor_strike — actual settlement strike")
    print("=" * 80)

    # Step 1: Fetch Kalshi settlements
    print("\n[1/4] Fetching Kalshi settlement history...")
    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    kalshi_markets = fetch_kalshi_settlements(client)

    # Step 2: Fetch 5m candle history from both exchanges
    print("\n[2/4] Fetching 5m candle history (Coinbase + Bitstamp)...")
    assets_syms = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}
    five_m_data = {}  # {asset: {exchange: df}}

    for asset, sym in assets_syms.items():
        five_m_data[asset] = {}
        for ex_name in ["coinbase", "bitstamp"]:
            print(f"  {asset} from {ex_name}...", end=" ", flush=True)
            df = fetch_5m_history(sym, ex_name, 100)
            five_m_data[asset][ex_name] = df
            print(f"{len(df)} candles")

    # Step 3: Fetch 15m/1h/4h from Coinbase for indicators
    print("\n[3/4] Fetching indicator data (Coinbase 15m/1h/4h)...")
    fetcher = DataFetcher()
    indicator_data = {}

    def fetch_candles(fetcher, symbol, timeframe, days):
        all_frames = []
        now_ms = int(time.time() * 1000)
        since = now_ms - days * 86400 * 1000
        batch_size = 300
        tf_ms = {"15m": 900000, "1h": 3600000, "4h": 14400000}
        candle_ms = tf_ms.get(timeframe, 900000)
        while since < now_ms:
            try:
                df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
                if df is None or df.empty:
                    since += batch_size * candle_ms; time.sleep(0.5); continue
                all_frames.append(df)
                since = int(df.index[-1].timestamp() * 1000) + candle_ms
                time.sleep(0.5)
            except Exception:
                since += batch_size * candle_ms; time.sleep(2)
        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames)[~pd.concat(all_frames).index.duplicated()].sort_index()

    for asset, sym in assets_syms.items():
        print(f"  {asset}...", end=" ", flush=True)
        df_15m = fetch_candles(fetcher, sym, "15m", 110)
        df_1h = fetch_candles(fetcher, sym, "1h", 110)
        df_4h = fetch_candles(fetcher, sym, "4h", 110)

        df = add_indicators(df_15m)
        pct = df["close"].pct_change()
        df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
        df["ema_slope"] = df["ema_12"].pct_change(3) * 100
        df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
        df["hourly_return"] = df["close"].pct_change(4) * 100

        indicator_data[asset] = {
            "15m": df,
            "1h": add_indicators(df_1h) if not df_1h.empty else None,
            "4h": add_indicators(df_4h) if not df_4h.empty else None,
        }
        print(f"{len(df)} 15m candles")

    # Step 4: Build training samples
    print("\n[4/4] Building training samples from Kalshi settlements...")
    all_rows = []

    for asset in SERIES:
        markets = kalshi_markets.get(asset, [])
        df_15m = indicator_data[asset]["15m"]
        df_1h = indicator_data[asset]["1h"]
        df_4h = indicator_data[asset]["4h"]
        cb_5m = five_m_data[asset].get("coinbase", pd.DataFrame())
        bs_5m = five_m_data[asset].get("bitstamp", pd.DataFrame())

        count = 0
        for m in markets:
            strike = float(m.get("floor_strike") or 0)
            result = m.get("result", "")
            close_time = m.get("close_time", "")
            settled_value = float(m.get("expiration_value") or 0)

            if not strike or not result or not close_time:
                continue

            label = 1 if result == "yes" else 0

            # Window start = close_time - 15 minutes
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = close_dt - timedelta(minutes=15)
            ws_naive = window_start.replace(tzinfo=None)

            # Price at MINUTE 1 — use 5m candle OPEN as proxy
            # The 5m candle starting at window_start has open = price at minute 0
            # By minute 1, price has barely moved from open — this matches live entry
            prices_at_min1 = []

            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty:
                    continue
                # Find the 5m candle that starts at or near window_start
                mask = (df_5m.index >= ws_naive) & (df_5m.index < ws_naive + timedelta(minutes=5))
                if mask.sum() > 0:
                    # Use OPEN of first 5m candle = price at window start (minute 0)
                    prices_at_min1.append(float(df_5m[mask].iloc[0]["open"]))
                else:
                    # Fallback: use close of last candle before window
                    before = df_5m[df_5m.index <= ws_naive]
                    if len(before) > 0 and (ws_naive - before.index[-1]).total_seconds() < 600:
                        prices_at_min1.append(float(before.iloc[-1]["close"]))

            if not prices_at_min1:
                continue
            price_at_min1 = sum(prices_at_min1) / len(prices_at_min1)

            # Get indicators from previous completed 15m candle
            prev_candles = df_15m[df_15m.index < ws_naive]
            if len(prev_candles) < 20:
                continue
            prev = prev_candles.iloc[-1]

            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            distance = (price_at_min1 - strike) / atr

            sma_val = float(prev.get("sma_20", 0))
            adx_val = float(prev.get("adx", 20))
            close_val = float(prev.get("close", 0))
            ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
            pve = float(prev.get("price_vs_ema", 0))
            hr = float(prev.get("hourly_return", 0))
            if pd.isna(pve) or np.isinf(pve): pve = 0
            if pd.isna(hr) or np.isinf(hr): hr = 0

            feat = {
                "macd_15m": float(prev.get("macd_hist", 0)),
                "norm_return": float(prev.get("norm_return", 0)) if pd.notna(prev.get("norm_return")) else 0,
                "ema_slope": float(prev.get("ema_slope", 0)) if pd.notna(prev.get("ema_slope")) else 0,
                "roc_5": float(prev.get("roc_5", 0)),
                "macd_1h": 0.0,
                "price_vs_ema": pve,
                "hourly_return": hr,
                "trend_direction": adx_val * ts_sign,
                "vol_ratio": float(prev.get("vol_ratio", 1)) if pd.notna(prev.get("vol_ratio")) else 1,
                "adx": adx_val,
                "rsi_1h": 50.0,
                "rsi_4h": 50.0,
                "distance_from_strike": distance,
            }

            if df_1h is not None:
                m1h = df_1h.index <= ws_naive
                if m1h.sum() >= 20:
                    r1h = df_1h.loc[m1h].iloc[-1]
                    feat["rsi_1h"] = float(r1h.get("rsi", 50))
                    feat["macd_1h"] = float(r1h.get("macd_hist", 0))
            if df_4h is not None:
                m4h = df_4h.index <= ws_naive
                if m4h.sum() >= 10:
                    feat["rsi_4h"] = float(df_4h.loc[m4h].iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            all_rows.append({
                **feat,
                "label": label,
                "ts": close_dt,
                "asset": asset,
                "strike": strike,
                "settled": settled_value,
                "price_at_min1": price_at_min1,
                "n_exchanges": len(prices_at_min1),
            })
            count += 1

        print(f"  {asset}: {count} samples")

    df_all = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df_all)} | Label YES: {df_all['label'].mean():.1%}")

    # Walk-forward split: train on oldest 70%, test on newest 30%
    split_idx = int(len(df_all) * 0.7)
    df_all = df_all.sort_values("ts").reset_index(drop=True)
    df_train = df_all.iloc[:split_idx]
    df_test = df_all.iloc[split_idx:]

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    X_train = df_train[FEATURE_NAMES].values
    X_test = df_test[FEATURE_NAMES].values

    train_start = df_train["ts"].iloc[0].strftime("%Y-%m-%d")
    train_end = df_train["ts"].iloc[-1].strftime("%Y-%m-%d")
    test_start = df_test["ts"].iloc[0].strftime("%Y-%m-%d")
    test_end = df_test["ts"].iloc[-1].strftime("%Y-%m-%d")

    print(f"Train: {len(df_train)} ({train_start} to {train_end}) | {y_train.mean():.1%} YES")
    print(f"Test:  {len(df_test)} ({test_start} to {test_end}) | {y_test.mean():.1%} YES")

    # Train
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
    print(f"Train WR: {train_wr:.1f}% ({yes_n}Y/{no_n}N/{len(train_probs)-yes_n-no_n}SKIP)")

    print(f"\nFeature coefficients:")
    for name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<22}: {coef:>+8.4f}")

    # Test
    print(f"\n{'=' * 80}")
    print("OUT-OF-SAMPLE TEST (Kalshi settlement labels)")
    print(f"{'=' * 80}")

    test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]

    print(f"\n{'Thresh':>6} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5} | {'Y:N':>5}")
    print("-" * 70)

    for thresh in [53, 55, 57, 60]:
        w, l = 0, 0
        yw, yl, nw, nl = 0, 0, 0, 0
        for i in range(len(y_test)):
            pct = int(test_probs[i] * 100)
            if pct >= thresh: side = "YES"
            elif pct <= (100 - thresh): side = "NO"
            else: continue
            won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
            if won:
                w += 1
                if side == "YES": yw += 1
                else: nw += 1
            else:
                l += 1
                if side == "YES": yl += 1
                else: nl += 1

        total = w + l
        if total < 20: continue
        wr = w / total * 100
        pnl = w * 0.50 - l * 0.50
        yn = yw + yl
        nn = nw + nl
        ywr = yw / yn * 100 if yn > 0 else 0
        nwr = nw / nn * 100 if nn > 0 else 0
        ratio = f"{yn/nn:.1f}" if nn > 0 else "inf"
        print(f"{thresh:>6} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {ywr:>4.0f}% {nwr:>4.0f}% | {ratio:>5}")

    # Save
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
            "labels": "kalshi_settlement",
            "price_source": "coinbase_bitstamp_avg",
            "train_period": f"{train_start} to {train_end}",
            "test_period": f"{test_start} to {test_end}",
        }, f)

    print(f"\nSaved to {output_path}")
    print(f"  Labels: Kalshi settlement results (not Coinbase candle direction)")
    print(f"  Price: Coinbase + Bitstamp 5m average")
    print(f"  Train: {train_start} to {train_end}")


if __name__ == "__main__":
    main()
