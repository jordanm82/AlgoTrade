#!/usr/bin/env python3
"""Train M10 model — predicts at minute 10 whether position should be held.

Different from the M0 (entry) model:
- M0: tiny distance, predicts from indicators, answers "which direction?"
- M10: large distance, predicts from position, answers "will it hold for 5 more min?"

Features at minute 10:
- distance_from_strike (KEY — large and meaningful at this point)
- Same 12 indicator features from the previous 15m candle
- bet_side (are we YES or NO?)

Labels: Kalshi settlement result — did this window actually settle YES or NO?

Training data: 5m candle at minute 10 (second 5m candle close) for price.

Usage:
    ./venv/bin/python scripts/train_m10_model.py
"""
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

# M10 features: same indicators + distance at minute 10
FEATURE_NAMES = [
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
    "vol_ratio", "adx", "rsi_1h", "rsi_4h",
    "distance_from_strike",  # this is the BIG one at minute 10
]


def fetch_kalshi_settlements(client):
    all_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for _ in range(30):
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
        print(f"  {asset}: {len(markets)} markets")
    return all_markets


def fetch_5m_history(symbol, exchange_name, days):
    constructors = {
        "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
        "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
    }
    ex = constructors[exchange_name]()
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    candle_ms = 300000
    while since < now_ms:
        try:
            candles = ex.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            if not candles:
                since += 1000 * candle_ms; time.sleep(0.3); continue
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop(columns=["timestamp"])
            all_frames.append(df)
            since = int(candles[-1][0]) + candle_ms
            time.sleep(0.3)
        except Exception:
            since += 500 * candle_ms; time.sleep(1)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames)[~pd.concat(all_frames).index.duplicated()].sort_index()


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


def main():
    print("=" * 80)
    print("M10 MODEL — Predicts at minute 10: will the position hold?")
    print("'Given price is X ATR from strike with 5 min left, will it stay?'")
    print("Labels: Kalshi settlement | Price at min 10: Coinbase+Bitstamp 5m close")
    print("=" * 80)

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("\n[1/4] Fetching Kalshi settlements...")
    kalshi_markets = fetch_kalshi_settlements(client)

    print("\n[2/4] Fetching 5m candle history...")
    five_m_data = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m_data[asset] = {}
        for ex_name in ["coinbase", "bitstamp"]:
            print(f"  {asset} {ex_name}...", end=" ", flush=True)
            df = fetch_5m_history(sym, ex_name, 100)
            five_m_data[asset][ex_name] = df
            print(f"{len(df)} candles")

    print("\n[3/4] Fetching indicator data...")
    indicator_data = {}
    for asset, sym in ASSETS_SYMS.items():
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
        print(f"  {asset}: {len(df)} 15m candles")

    print("\n[4/4] Building M10 training samples...")
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
            if not strike or not result or not close_time:
                continue

            label = 1 if result == "yes" else 0
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = close_dt - timedelta(minutes=15)
            ws_naive = window_start.replace(tzinfo=None)

            # Price at MINUTE 10 — second 5m candle close (minutes 5-10)
            min10_time = ws_naive + timedelta(minutes=10)
            prices_at_min10 = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty:
                    continue
                # Find 5m candle closing at minute 10 (index at minute 5, close at minute 10)
                mask = (df_5m.index >= ws_naive + timedelta(minutes=5)) & (df_5m.index < min10_time)
                if mask.sum() > 0:
                    prices_at_min10.append(float(df_5m[mask].iloc[-1]["close"]))
                else:
                    # Fallback: closest candle before minute 10
                    before = df_5m[df_5m.index <= min10_time]
                    if len(before) > 0 and (min10_time - before.index[-1]).total_seconds() < 600:
                        prices_at_min10.append(float(before.iloc[-1]["close"]))
            if not prices_at_min10:
                continue
            price_at_min10 = sum(prices_at_min10) / len(prices_at_min10)

            # Previous completed 15m candle for indicators
            prev = df_15m[df_15m.index < ws_naive]
            if len(prev) < 20:
                continue
            pr = prev.iloc[-1]
            atr = float(pr.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            # Distance at minute 10 — this is the KEY feature
            distance = (price_at_min10 - strike) / atr

            sma_val = float(pr.get("sma_20", 0))
            adx_val = float(pr.get("adx", 20))
            close_val = float(pr.get("close", 0))
            if sma_val > 0:
                ts_sign = 1 if close_val >= sma_val else -1
            else:
                ts_sign = 0
            pve = float(pr.get("price_vs_ema", 0))
            hr = float(pr.get("hourly_return", 0))
            if pd.isna(pve) or np.isinf(pve): pve = 0
            if pd.isna(hr) or np.isinf(hr): hr = 0

            feat = {
                "macd_15m": float(pr.get("macd_hist", 0)),
                "norm_return": float(pr.get("norm_return", 0)) if pd.notna(pr.get("norm_return")) else 0,
                "ema_slope": float(pr.get("ema_slope", 0)) if pd.notna(pr.get("ema_slope")) else 0,
                "roc_5": float(pr.get("roc_5", 0)),
                "macd_1h": 0.0,
                "price_vs_ema": pve,
                "hourly_return": hr,
                "trend_direction": adx_val * ts_sign,
                "vol_ratio": float(pr.get("vol_ratio", 1)) if pd.notna(pr.get("vol_ratio")) else 1,
                "adx": adx_val,
                "rsi_1h": 50.0,
                "rsi_4h": 50.0,
                "distance_from_strike": distance,
            }
            if df_1h is not None:
                m1h = df_1h.index <= ws_naive
                if m1h.sum() >= 20:
                    feat["rsi_1h"] = float(df_1h.loc[m1h].iloc[-1].get("rsi", 50))
                    feat["macd_1h"] = float(df_1h.loc[m1h].iloc[-1].get("macd_hist", 0))
            if df_4h is not None:
                m4h = df_4h.index <= ws_naive
                if m4h.sum() >= 10:
                    feat["rsi_4h"] = float(df_4h.loc[m4h].iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            all_rows.append({
                **feat, "label": label, "ts": close_dt, "asset": asset,
                "strike": strike, "price_at_min10": price_at_min10,
            })
            count += 1
        print(f"  {asset}: {count} samples")

    df_all = pd.DataFrame(all_rows).sort_values("ts").reset_index(drop=True)
    print(f"\nTotal: {len(df_all)} | Label YES: {df_all['label'].mean():.1%}")

    # Walk-forward: train 70%, test 30%
    split = int(len(df_all) * 0.7)
    df_train = df_all.iloc[:split]
    df_test = df_all.iloc[split:]

    y_train = df_train["label"].values
    y_test = df_test["label"].values
    X_train = df_train[FEATURE_NAMES].values
    X_test = df_test[FEATURE_NAMES].values

    print(f"Train: {len(df_train)} ({df_train['ts'].iloc[0].strftime('%m/%d')} - {df_train['ts'].iloc[-1].strftime('%m/%d')})")
    print(f"Test:  {len(df_test)} ({df_test['ts'].iloc[0].strftime('%m/%d')} - {df_test['ts'].iloc[-1].strftime('%m/%d')})")

    # Train M10 model
    print(f"\n{'=' * 80}")
    print("TRAINING M10 MODEL")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train_s, y_train)

    # Feature coefficients
    print(f"\nFeature coefficients:")
    for name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name:<22}: {coef:>+8.4f}")

    # Evaluate
    print(f"\n{'=' * 80}")
    print("M10 MODEL — OUT-OF-SAMPLE TEST")
    print(f"{'=' * 80}")

    test_probs = model.predict_proba(scaler.transform(X_test))[:, 1]

    print(f"\n{'Thresh':>6} | {'Bets':>5} {'WR':>6} {'P&L':>8} | {'YES':>5} {'NO':>5} {'Y_WR':>5} {'N_WR':>5}")
    print("-" * 60)

    for thresh in [53, 55, 57, 60, 65, 70]:
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
        print(f"{thresh:>6} | {total:>4} {wr:>5.1f}% ${pnl:>+6.0f} | {yn:>4} {nn:>4} {yw/yn*100 if yn else 0:>4.0f}% {nw/nn*100 if nn else 0:>4.0f}%")

    # Compare: what does the M0 model predict for the same samples?
    print(f"\n{'=' * 80}")
    print("M0 vs M10 COMPARISON (same test set, minute-10 distance)")
    print(f"{'=' * 80}")

    with open("models/knn_kalshi.pkl", "rb") as f:
        m0 = pickle.load(f)
    m0_model = m0["knn"]
    m0_scaler = m0["scaler"]

    m0_probs = m0_model.predict_proba(m0_scaler.transform(X_test))[:, 1]
    m10_probs = test_probs

    # At threshold 55
    for label, probs in [("M0 (entry model)", m0_probs), ("M10 (new model)", m10_probs)]:
        w, l, s = 0, 0, 0
        for i in range(len(y_test)):
            pct = int(probs[i] * 100)
            if pct >= 55: side = "YES"
            elif pct <= 45: side = "NO"
            else: s += 1; continue
            won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
            if won: w += 1
            else: l += 1
        total = w + l
        wr = w / total * 100 if total else 0
        print(f"  {label:<25}: {wr:.1f}% WR ({w}W/{l}L/{s}SKIP)")

    # Save
    output = Path("models/m10_kalshi.pkl")
    with open(output, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "model_type": "m10_confirmation",
            "feature_names": FEATURE_NAMES,
            "training_samples": len(df_train),
            "base_rate": float(y_train.mean()),
        }, f)

    print(f"\nSaved M10 model to {output}")


if __name__ == "__main__":
    main()
