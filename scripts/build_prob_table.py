#!/usr/bin/env python3
"""Build the Kalshi V3 probability lookup table from historical 1m candle data.

For each historical 15m window, the open price = simulated strike.
At each 1m mark, compute distance from strike in ATR units.
Record outcome: did the 15m window close >= open?
Aggregate into a 2D table: P(close >= strike | distance_bucket, time_bucket).

Usage:
    ./venv/bin/python scripts/build_prob_table.py [--days 90] [--output data/store/kalshi_prob_table.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators

ASSETS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
}

# Distance buckets (in ATR units)
DISTANCE_BINS = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
# Time buckets (minutes remaining in the 15m window)
TIME_BINS = [14, 12, 10, 8, 6, 4, 2]


def fetch_1m_chunked(fetcher: DataFetcher, symbol: str, days: int) -> pd.DataFrame:
    """Fetch 1m candles in weekly chunks to avoid timeouts."""
    all_frames = []
    now_ms = int(time.time() * 1000)
    chunk_days = 7
    candle_ms = 60 * 1000  # 1 minute

    total_chunks = (days + chunk_days - 1) // chunk_days
    for chunk in range(total_chunks):
        start_day = days - (chunk + 1) * chunk_days
        end_day = days - chunk * chunk_days
        if start_day < 0:
            start_day = 0

        since = now_ms - end_day * 24 * 60 * 60 * 1000
        until = now_ms - start_day * 24 * 60 * 60 * 1000

        batch_since = since
        while batch_since < until:
            try:
                df = fetcher.ohlcv(symbol, "1m", limit=300, since=batch_since)
                if df is None or df.empty:
                    break
                all_frames.append(df)
                last_ts = int(df.index[-1].timestamp() * 1000)
                batch_since = last_ts + candle_ms
                if len(df) < 1000:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"    Warning: {e}")
                batch_since += 300 * candle_ms
                time.sleep(1)

        print(f"    Chunk {chunk+1}/{total_chunks}: {len(all_frames)} batches so far")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def compute_15m_atr(df_1m: pd.DataFrame) -> pd.Series:
    """Compute ATR from 1m candles aggregated to 15m for distance normalization."""
    # Resample 1m to 15m
    df_15m = df_1m.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    if len(df_15m) < 15:
        return pd.Series(dtype=float)

    # Compute ATR(14) on 15m
    import pandas_ta as ta
    atr = ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    return atr


def bucket_distance(distance_atr: float) -> float:
    """Assign a distance value to the nearest bucket."""
    closest = min(DISTANCE_BINS, key=lambda b: abs(b - distance_atr))
    return closest


def build_table(df_1m: pd.DataFrame, asset_name: str) -> dict:
    """Build probability observations from 1m candles for one asset."""
    observations = []  # list of (distance_bucket, time_bucket, outcome)

    # Compute 15m ATR series
    atr_series = compute_15m_atr(df_1m)
    if atr_series.empty:
        print(f"  {asset_name}: insufficient data for ATR")
        return {}

    # Group 1m candles into 15m windows
    # Window starts at :00, :15, :30, :45
    df_1m = df_1m.copy()
    df_1m["window_start"] = df_1m.index.floor("15min")
    df_1m["minute_in_window"] = (df_1m.index - df_1m["window_start"]).dt.total_seconds() / 60

    windows = df_1m.groupby("window_start")

    for window_start, group in windows:
        if len(group) < 14:  # need nearly complete window
            continue

        # Strike = open of the window (first 1m candle open)
        strike = float(group.iloc[0]["open"])
        # Outcome = did window close >= strike?
        window_close = float(group.iloc[-1]["close"])
        outcome = 1 if window_close >= strike else 0

        # Get ATR for this window (from the 15m ATR series)
        # Find the nearest 15m ATR value at or before this window
        atr_mask = atr_series.index <= window_start
        if atr_mask.sum() == 0:
            continue
        atr_val = float(atr_series.loc[atr_mask].iloc[-1])
        if atr_val <= 0:
            continue

        # For each 1m candle in the window, record an observation
        for _, row in group.iterrows():
            minute = int(row["minute_in_window"])
            minutes_remaining = 14 - minute
            if minutes_remaining < 1 or minutes_remaining > 14:
                continue

            # Find nearest time bucket
            closest_time = min(TIME_BINS, key=lambda t: abs(t - minutes_remaining))

            current_price = float(row["close"])
            distance = (current_price - strike) / atr_val
            dist_bucket = bucket_distance(distance)

            observations.append({
                "distance": dist_bucket,
                "time": closest_time,
                "outcome": outcome,
            })

    return observations


def aggregate_observations(all_observations: list) -> dict:
    """Aggregate observations into probability table."""
    table = {}

    for obs in all_observations:
        key = f"{obs['distance']}_{obs['time']}"
        if key not in table:
            table[key] = {"wins": 0, "total": 0}
        table[key]["total"] += 1
        table[key]["wins"] += obs["outcome"]

    # Convert to probabilities
    prob_table = {}
    for key, counts in table.items():
        dist_str, time_str = key.split("_")
        if counts["total"] >= 30:  # minimum sample size
            prob = counts["wins"] / counts["total"]
        else:
            prob = 0.5  # insufficient data, default to 50%
        prob_table[key] = {
            "probability": round(prob, 4),
            "sample_size": counts["total"],
            "distance": float(dist_str),
            "time_remaining": int(float(time_str)),
        }

    return prob_table


def main():
    parser = argparse.ArgumentParser(description="Build Kalshi V3 probability table")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--output", default="data/store/kalshi_prob_table.json")
    args = parser.parse_args()

    fetcher = DataFetcher()
    all_observations = []

    for asset_name, symbol in ASSETS.items():
        print(f"\nFetching {asset_name} 1m candles ({args.days} days)...")
        df_1m = fetch_1m_chunked(fetcher, symbol, args.days)
        if df_1m.empty:
            print(f"  {asset_name}: no data")
            continue
        print(f"  {asset_name}: {len(df_1m)} candles ({df_1m.index[0]} to {df_1m.index[-1]})")

        print(f"  Building observations...")
        obs = build_table(df_1m, asset_name)
        if isinstance(obs, list):
            all_observations.extend(obs)
            print(f"  {asset_name}: {len(obs)} observations")

    print(f"\nTotal observations: {len(all_observations)}")
    print("Aggregating into probability table...")

    prob_table = aggregate_observations(all_observations)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prob_table, f, indent=2)

    print(f"\nSaved probability table to {output_path}")
    print(f"Cells: {len(prob_table)}")

    # Print summary
    print("\nProbability Table Summary:")
    print(f"{'Distance':>10} | " + " | ".join(f"{t:>4}min" for t in TIME_BINS))
    print("-" * 70)
    for dist in DISTANCE_BINS:
        row = []
        for t in TIME_BINS:
            key = f"{dist}_{t}"
            cell = prob_table.get(key, {})
            prob = cell.get("probability", 0.5)
            n = cell.get("sample_size", 0)
            row.append(f"{prob:.0%}({n:>3})" if n >= 30 else "  ---  ")
        print(f"{dist:>+10.2f} | " + " | ".join(row))


if __name__ == "__main__":
    main()
