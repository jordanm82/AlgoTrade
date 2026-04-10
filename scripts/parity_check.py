#!/usr/bin/env python3
"""Compare live feature values vs backtest-computed values for the same windows.

Reads the feature log, identifies the window for each prediction,
recomputes features using the backtest code path, and shows the diff.

Usage:
    ./venv/bin/python scripts/parity_check.py [--last N]
"""
import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from scripts.backtest_kalshi_labels import fetch_5m_history, fetch_candles, get_avg_price, build_features

ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}
SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=10, help="Check last N predictions")
    args = parser.parse_args()

    # Read recent feature log entries
    log_path = Path("data/store/feature_log.jsonl")
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    entries = entries[-args.last * 4:]  # get more than needed, filter later

    # Identify unique windows (by close value → asset)
    recent = []
    for e in entries:
        close = e.get("close", 0)
        if close > 50000: asset = "BTC"
        elif close > 1000: asset = "ETH"
        elif close > 50: asset = "SOL"
        else: asset = "XRP"
        e["_asset"] = asset
        recent.append(e)

    # Deduplicate — keep last entry per (asset, close) combo
    seen = {}
    for e in recent:
        key = f"{e['_asset']}_{e['close']}"
        seen[key] = e
    recent = list(seen.values())[-args.last:]

    print(f"Checking {len(recent)} predictions against backtest computation...\n")

    # Fetch current data (same as live daemon would have)
    fetcher = DataFetcher()
    print("Fetching indicator data...")
    ind_data = {}
    for asset, sym in ASSETS_SYMS.items():
        df_15m = add_indicators(fetcher.ohlcv(sym, "15m", limit=200))
        # Drop in-progress candles (CCXT returns current unfinished candle as last row)
        raw_1h = fetcher.ohlcv(sym, "1h", limit=100)
        df_1h = add_indicators(raw_1h.iloc[:-1]) if raw_1h is not None and len(raw_1h) > 1 else None
        raw_4h = fetcher.ohlcv(sym, "4h", limit=50)
        df_4h = add_indicators(raw_4h.iloc[:-1]) if raw_4h is not None and len(raw_4h) > 1 else None
        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100
        ind_data[asset] = {"15m": df_15m, "1h": df_1h, "4h": df_4h}

    # Compare each prediction
    FEATURES_TO_CHECK = [
        "macd_15m", "norm_return", "ema_slope", "roc_5",
        "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
        "vol_ratio", "adx", "rsi_1h", "rsi_4h", "rsi_15m", "bbw",
        "rsi_alignment", "atr_percentile",
        "return_4h", "return_12h", "price_vs_sma_1h", "lower_lows_4h", "trend_strength",
    ]

    mismatches = 0
    for entry in recent:
        asset = entry["_asset"]
        ts = entry.get("ts", "")
        prob = entry.get("prob", 0)
        side = entry.get("side", "?")

        # Parse timestamp, find window start
        try:
            dt = datetime.fromisoformat(ts)
        except:
            continue
        ws = dt.replace(minute=dt.minute - (dt.minute % 15), second=0, microsecond=0, tzinfo=None)

        # Recompute features using backtest logic
        df_15m = ind_data[asset]["15m"]
        df_1h = ind_data[asset]["1h"]
        df_4h = ind_data[asset]["4h"]

        prev_c = df_15m[df_15m.index < ws]
        if len(prev_c) < 20:
            print(f"  {ts[:16]} {asset}: not enough candles before {ws}")
            continue
        prev = prev_c.iloc[-1]
        atr = float(prev.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            continue

        # ATR percentile (backtest style)
        atr_s = df_15m["atr"].dropna()
        ar20 = atr_s.rolling(20)
        atr_p = ((atr_s - ar20.min()) / (ar20.max() - ar20.min())).fillna(0.5)
        pav = atr_p[atr_p.index < ws]
        apv = float(pav.iloc[-1]) if len(pav) > 0 else 0.5

        # Regime features (backtest style)
        kx = {}
        if len(prev_c) >= 16:
            kx["return_4h"] = (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-16]["close"])) / float(prev_c.iloc[-16]["close"]) * 100
        else:
            kx["return_4h"] = 0
        if len(prev_c) >= 48:
            kx["return_12h"] = (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-48]["close"])) / float(prev_c.iloc[-48]["close"]) * 100
        else:
            kx["return_12h"] = 0
        if df_1h is not None:
            h1f = df_1h[df_1h.index <= ws]
            if len(h1f) >= 20 and atr > 0:
                kx["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
            else:
                kx["price_vs_sma_1h"] = 0
        else:
            kx["price_vs_sma_1h"] = 0
        if df_4h is not None:
            h4f = df_4h[df_4h.index <= ws]
            if len(h4f) >= 4:
                kx["lower_lows_4h"] = sum(1 for i in range(-3, 0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i - 1]["low"]))
            else:
                kx["lower_lows_4h"] = 0
            if len(h4f) >= 10 and atr > 0:
                kx["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr
            else:
                kx["trend_strength"] = 0
        else:
            kx["lower_lows_4h"] = 0
            kx["trend_strength"] = 0

        bt_feat = build_features(prev, df_1h, df_4h, ws, 0,  # distance=0 placeholder
                                 kalshi_extra=kx, atr_pctile_val=apv)
        if not bt_feat:
            continue

        # Compare
        diffs = []
        for f in FEATURES_TO_CHECK:
            live_val = entry.get(f)
            bt_val = bt_feat.get(f)
            if live_val is None or bt_val is None:
                diffs.append((f, live_val, bt_val, "MISSING"))
                continue
            if abs(float(live_val) - float(bt_val)) > 0.001:
                diffs.append((f, float(live_val), float(bt_val), abs(float(live_val) - float(bt_val))))

        if diffs:
            mismatches += 1
            print(f"{'='*70}")
            print(f"{ts[:16]} {asset} prob={prob:.2f} side={side}")
            print(f"  MISMATCHES:")
            for fname, lv, bv, diff in diffs:
                if diff == "MISSING":
                    print(f"    {fname:<22}: live={lv} bt={bv} [MISSING]")
                else:
                    print(f"    {fname:<22}: live={lv:>10.4f}  bt={bv:>10.4f}  diff={diff:.4f}")
        else:
            print(f"  {ts[:16]} {asset} prob={prob:.2f} — ALL MATCH ✓")

    print(f"\n{'='*70}")
    print(f"SUMMARY: {mismatches} predictions with mismatches out of {len(recent)} checked")
    if mismatches == 0:
        print("ALL FEATURES MATCH — no parity issues detected")
    else:
        print(f"PARITY ISSUES FOUND in {mismatches} predictions")


if __name__ == "__main__":
    main()
