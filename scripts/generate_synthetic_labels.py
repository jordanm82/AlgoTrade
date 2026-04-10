#!/usr/bin/env python3
"""Generate synthetic K15 training labels from historical Coinbase data.

For any 15-minute window going back 2+ years:
  - Strike = 5m candle OPEN at window start (Coinbase+Bitstamp averaged)
  - Label = did price close above strike at minute 15? (YES=1, NO=0)
  - All indicator features from Coinbase 15m/1h/4h candles

This gives us training data from bear markets, consolidation, and bull runs
— not just the bullish Dec-Mar 2025/2026 period where every dip bounced.

Usage:
    ./venv/bin/python scripts/generate_synthetic_labels.py [--months 12]
"""
import argparse
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators

ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}


def fetch_long_history(symbol, exchange_name, timeframe, months):
    """Fetch extended history from exchange."""
    constructors = {
        "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
        "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
    }
    ex = constructors[exchange_name]()
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms[timeframe]

    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - months * 30 * 86400 * 1000

    print(f"    Fetching {symbol} {timeframe} from {exchange_name} ({months} months)...", end=" ", flush=True)
    while since < now_ms:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
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
        except Exception:
            since += 500 * candle_ms
            time.sleep(1)

    if not all_frames:
        print("FAILED")
        return pd.DataFrame()
    result = pd.concat(all_frames)[~pd.concat(all_frames).index.duplicated()].sort_index()
    print(f"{len(result)} candles")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=12, help="Months of history")
    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING SYNTHETIC K15 LABELS ({args.months} months)")
    print("Strike = 5m open at window start | Label = close above strike at min 15")
    print("=" * 80)

    # Fetch extended history
    print("\n[1/3] Fetching historical data...")
    data = {}
    for asset, sym in ASSETS_SYMS.items():
        print(f"  {asset}:")
        data[asset] = {
            "5m_cb": fetch_long_history(sym, "coinbase", "5m", args.months + 2),
            "5m_bs": fetch_long_history(sym, "bitstamp", "5m", args.months + 2),
            "15m": fetch_long_history(sym, "coinbase", "15m", args.months + 3),
            "1h": fetch_long_history(sym, "coinbase", "1h", args.months + 3),
            "4h": fetch_long_history(sym, "coinbase", "4h", args.months + 3),
        }

    # Build synthetic labels
    print("\n[2/3] Building synthetic labels...")
    all_samples = {}

    for asset, sym in ASSETS_SYMS.items():
        cb_5m = data[asset]["5m_cb"]
        bs_5m = data[asset]["5m_bs"]
        df_15m = add_indicators(data[asset]["15m"])
        df_1h = add_indicators(data[asset]["1h"])
        df_4h = add_indicators(data[asset]["4h"])

        # Add derived features
        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100

        samples = []
        # Iterate through every 15m candle as a potential window
        for i in range(50, len(df_15m) - 1):
            ws = df_15m.index[i]  # window start = this candle's timestamp
            ws_naive = ws

            # Strike = average of Coinbase + Bitstamp 5m open at window start
            prices = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty:
                    continue
                mask = (df_5m.index >= ws_naive) & (df_5m.index < ws_naive + timedelta(minutes=5))
                if mask.sum() > 0:
                    prices.append(float(df_5m[mask].iloc[0]["open"]))
            if not prices:
                continue
            strike = sum(prices) / len(prices)

            # Settlement price = close of the NEXT 15m candle
            settle_price = float(df_15m.iloc[i + 1]["close"])

            # Label: did price close above strike?
            label = 1 if settle_price > strike else 0

            # Previous candle for indicators (what the model sees at entry)
            prev = df_15m.iloc[i - 1]  # candle BEFORE window start
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            # Distance from strike
            price_at_min0 = sum(prices) / len(prices)
            distance = (price_at_min0 - strike) / atr  # will be ~0 since price_at_min0 ≈ strike

            samples.append({
                "ts": ws,
                "asset": asset,
                "label": label,
                "strike": strike,
                "settled": settle_price,
                "price_at_min0": price_at_min0,
                "distance_from_strike": distance,
                "atr": atr,
                # Store index for feature extraction later
                "_15m_idx": i - 1,
            })

        all_samples[asset] = {
            "samples": samples,
            "15m": df_15m,
            "1h": df_1h,
            "4h": df_4h,
        }
        yes_pct = sum(s["label"] for s in samples) / len(samples) * 100 if samples else 0
        print(f"  {asset}: {len(samples)} synthetic windows | YES: {yes_pct:.1f}%")

    # Save
    output = Path("data/store/synthetic_labels.pkl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(all_samples, f)
    print(f"\n[3/3] Saved to {output}")

    # Summary stats
    total = sum(len(d["samples"]) for d in all_samples.values())
    print(f"\nTotal: {total} synthetic windows across {len(ASSETS_SYMS)} assets")
    print(f"Period: ~{args.months} months")
    print(f"\nUse with train_per_asset_models.py --synthetic to include in training")


if __name__ == "__main__":
    main()
