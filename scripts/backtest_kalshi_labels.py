#!/usr/bin/env python3
"""Backtest using Kalshi settlement labels + multi-exchange price.

Uses the SAME data sources as the retrained model:
  - Labels: Kalshi settled market result (yes/no)
  - Strike: Kalshi floor_strike
  - Price at min 5: Coinbase + Bitstamp 5m average
  - Indicators: Coinbase 15m/1h/4h

Tests the saved model against Kalshi ground truth.

Usage:
    ./venv/bin/python scripts/backtest_kalshi_labels.py [--days 30]
"""
import argparse
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

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

STARTING_BALANCE = 100.0
RISK_PCT = 0.05
MAX_CONTRACTS = 100
ENTRY_CENTS = 50
MAX_PER_WINDOW = 3


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    with open("models/knn_kalshi.pkl", "rb") as f:
        m = pickle.load(f)
    model = m["knn"]
    scaler_m = m["scaler"]
    FEATURES = m["feature_names"]

    print("=" * 80)
    print(f"BACKTEST — Kalshi Labels + Multi-Exchange Price ({args.days} days)")
    print(f"Model: {m.get('model_type')} | Labels: {m.get('labels', '?')}")
    print(f"Price source: {m.get('price_source', '?')}")
    print("=" * 80)

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Fetch Kalshi settlements
    print("\nFetching Kalshi settlements...")
    kalshi_markets = {}
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
        # Filter to backtest period
        filtered = []
        for mk in markets:
            ct = mk.get("close_time", "")
            if ct:
                close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if close_dt >= cutoff:
                    filtered.append(mk)
        kalshi_markets[asset] = filtered
        print(f"  {asset}: {len(filtered)} markets in last {args.days} days")

    # Fetch 5m data from both exchanges
    print("\nFetching 5m candles (Coinbase + Bitstamp)...")
    five_m = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m[asset] = {}
        for ex in ["coinbase", "bitstamp"]:
            print(f"  {asset} {ex}...", end=" ", flush=True)
            df = fetch_5m_history(sym, ex, args.days + 5)
            five_m[asset][ex] = df
            print(f"{len(df)} candles")

    # Fetch indicator data
    print("\nFetching indicator data...")
    ind_data = {}
    for asset, sym in ASSETS_SYMS.items():
        df_15m = add_indicators(fetch_candles(fetcher, sym, "15m", args.days + 30))
        df_1h = fetch_candles(fetcher, sym, "1h", args.days + 30)
        df_4h = fetch_candles(fetcher, sym, "4h", args.days + 30)
        df_1h_ind = add_indicators(df_1h) if not df_1h.empty else None
        df_4h_ind = add_indicators(df_4h) if not df_4h.empty else None

        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100

        ind_data[asset] = {"15m": df_15m, "1h": df_1h_ind, "4h": df_4h_ind}
        print(f"  {asset}: {len(df_15m)} 15m candles")

    # Build signals and check against Kalshi settlements
    print("\nScoring...")
    all_signals = []

    for asset in SERIES:
        markets = kalshi_markets.get(asset, [])
        df_15m = ind_data[asset]["15m"]
        df_1h = ind_data[asset]["1h"]
        df_4h = ind_data[asset]["4h"]
        cb_5m = five_m[asset].get("coinbase", pd.DataFrame())
        bs_5m = five_m[asset].get("bitstamp", pd.DataFrame())

        count = 0
        for mk in markets:
            strike = float(mk.get("floor_strike") or 0)
            result = mk.get("result", "")
            close_time = mk.get("close_time", "")
            if not strike or not result or not close_time:
                continue

            label = 1 if result == "yes" else 0
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = close_dt - timedelta(minutes=15)
            ws_naive = window_start.replace(tzinfo=None)
            # Multi-exchange price at MINUTE 1 (5m candle open = window start price)
            prices = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty:
                    continue
                mask = (df_5m.index >= ws_naive) & (df_5m.index < ws_naive + timedelta(minutes=5))
                if mask.sum() > 0:
                    prices.append(float(df_5m[mask].iloc[0]["open"]))
                else:
                    before = df_5m[df_5m.index <= ws_naive]
                    if len(before) > 0 and (ws_naive - before.index[-1]).total_seconds() < 600:
                        prices.append(float(before.iloc[-1]["close"]))
            if not prices:
                continue
            price_at_min1 = sum(prices) / len(prices)

            # Indicators from previous 15m candle
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
                    feat["rsi_1h"] = float(df_1h.loc[m1h].iloc[-1].get("rsi", 50))
                    feat["macd_1h"] = float(df_1h.loc[m1h].iloc[-1].get("macd_hist", 0))
            if df_4h is not None:
                m4h = df_4h.index <= ws_naive
                if m4h.sum() >= 10:
                    feat["rsi_4h"] = float(df_4h.loc[m4h].iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            X = np.array([feat[f] for f in FEATURES]).reshape(1, -1)
            prob = float(model.predict_proba(scaler_m.transform(X))[0][1])
            pct_val = int(prob * 100)

            if pct_val >= 55:
                side = "YES"
            elif pct_val <= 45:
                side = "NO"
            else:
                continue

            won = (side == "YES" and label == 1) or (side == "NO" and label == 0)
            all_signals.append({
                "ts": close_dt,
                "asset": asset,
                "side": side,
                "prob": prob,
                "won": won,
                "label": label,
                "distance": distance,
                "strike": strike,
                "settled": float(mk.get("expiration_value", 0)),
            })
            count += 1
        print(f"  {asset}: {count} signals")

    df_sig = pd.DataFrame(all_signals).sort_values("ts").reset_index(drop=True)
    yes_n = len(df_sig[df_sig["side"] == "YES"])
    no_n = len(df_sig[df_sig["side"] == "NO"])
    print(f"\nTotal signals: {len(df_sig)} ({yes_n}Y / {no_n}N)")

    # Flat P&L
    wr = df_sig["won"].mean() * 100
    flat_pnl = df_sig["won"].sum() * 0.50 - (~df_sig["won"]).sum() * 0.50
    y_wr = df_sig[df_sig["side"] == "YES"]["won"].mean() * 100 if yes_n > 0 else 0
    n_wr = df_sig[df_sig["side"] == "NO"]["won"].mean() * 100 if no_n > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"RESULTS — Kalshi Settlement Ground Truth")
    print(f"{'=' * 80}")
    print(f"  WR:   {wr:.1f}% ({df_sig['won'].sum()}W / {(~df_sig['won']).sum()}L)")
    print(f"  YES:  {y_wr:.1f}% | NO: {n_wr:.1f}%")
    print(f"  Y:N ratio: {yes_n/no_n:.1f}:1" if no_n > 0 else "  Y:N ratio: inf")
    print(f"  P&L:  ${flat_pnl:+.2f}")

    # Daily breakdown
    print(f"\n{'=' * 80}")
    print("DAILY BREAKDOWN")
    print(f"{'=' * 80}")
    df_sig["date"] = df_sig["ts"].apply(lambda t: t.strftime("%m/%d"))
    daily = df_sig.groupby("date").agg(bets=("won", "count"), wins=("won", "sum"))
    daily["wr"] = daily["wins"] / daily["bets"] * 100
    daily["pnl"] = daily["wins"] * 0.50 - (daily["bets"] - daily["wins"]) * 0.50

    print(f"{'Date':<8} {'Bets':>5} {'WR':>6} {'P&L':>8}")
    print("-" * 30)
    for date, row in daily.iterrows():
        print(f"{date:<8} {int(row['bets']):>5} {row['wr']:>5.0f}% ${row['pnl']:>+6.1f}")

    winning_days = (daily["pnl"] > 0).sum()
    losing_days = (daily["pnl"] < 0).sum()
    print(f"\n  Winning days: {winning_days} | Losing days: {losing_days}")

    # Compounding
    print(f"\n{'=' * 80}")
    print(f"COMPOUNDING: $100 start, 5% per bet, max 100 contracts, max 3/window")
    print(f"{'=' * 80}")

    df_sig["window"] = df_sig["ts"].apply(lambda t: t.replace(tzinfo=None).floor("15min") if hasattr(t.replace(tzinfo=None), 'floor') else t)
    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0
    bets_placed = 0
    cap_hits = 0

    for _, group in df_sig.groupby(df_sig["ts"].dt.floor("15min")):
        window_bets = group.nlargest(MAX_PER_WINDOW, "prob")
        for _, row in window_bets.iterrows():
            risk = balance * RISK_PCT
            contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
            cost = contracts * (ENTRY_CENTS / 100)
            if cost > balance:
                continue
            if contracts == MAX_CONTRACTS:
                cap_hits += 1
            bets_placed += 1
            if row["won"]:
                balance += contracts * ((100 - ENTRY_CENTS) / 100)
            else:
                balance -= cost
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

    ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    print(f"  Bets placed: {bets_placed}")
    print(f"  Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
    print(f"  Max drawdown: {max_dd:.1f}%")


if __name__ == "__main__":
    main()
