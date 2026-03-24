#!/usr/bin/env python3
"""Backtest Kalshi confidence-based predictor across 6 months of 15m data.

Fetches BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT from Binance US,
applies indicators, scores confidence at each candle, and evaluates
whether the predicted direction matches the next candle's actual movement.

Tests thresholds from 30 to 80 (step 5) and reports:
- Total bets placed
- Win rate
- Estimated P&L (simulating Kalshi-style payouts)

Payout simulation:
  buy_price_cents = min(85, confidence - 10)
  If correct: profit = (100 - buy_price_cents) per contract
  If wrong:   loss   = buy_price_cents per contract
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.strategies.kalshi_predictor import KalshiPredictor


ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
}

TIMEFRAME = "15m"
SIX_MONTHS_CANDLES = 6 * 30 * 24 * 4  # ~17,280 candles per asset
THRESHOLDS = list(range(30, 85, 5))


def fetch_6months(fetcher: DataFetcher, symbol: str) -> pd.DataFrame:
    """Fetch ~6 months of 15m candles via paginated requests (max 1000/req)."""
    all_frames = []
    now_ms = int(time.time() * 1000)
    six_months_ms = 6 * 30 * 24 * 60 * 60 * 1000
    since = now_ms - six_months_ms
    batch_size = 1000
    candle_ms = 15 * 60 * 1000  # 15 minutes in ms

    print(f"  Fetching {symbol} from {pd.Timestamp(since, unit='ms')} ...")

    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, TIMEFRAME, limit=batch_size, since=since)
            if df.empty:
                break
            all_frames.append(df)
            # Advance past the last candle we received
            last_ts = int(df.index[-1].timestamp() * 1000)
            since = last_ts + candle_ms
            if len(df) < batch_size:
                break
            time.sleep(0.3)  # rate limit courtesy
        except Exception as e:
            print(f"    Warning: fetch error at {pd.Timestamp(since, unit='ms')}: {e}")
            since += batch_size * candle_ms  # skip ahead
            time.sleep(1)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    print(f"    Got {len(combined)} candles ({combined.index[0]} to {combined.index[-1]})")
    return combined


def backtest_asset(asset_name: str, df_raw: pd.DataFrame, predictor: KalshiPredictor) -> list[dict]:
    """Run predictor on every candle, record results."""
    df = add_indicators(df_raw)
    results = []

    # We need at least 20 candles of history, and 1 future candle for result
    for i in range(20, len(df) - 1):
        window = df.iloc[:i + 1]
        signal = predictor.score(window)

        if signal is None:
            continue

        # Actual direction: did the NEXT candle close higher or lower?
        current_close = float(df.iloc[i]["close"])
        next_close = float(df.iloc[i + 1]["close"])
        actual_direction = "UP" if next_close > current_close else "DOWN"

        results.append({
            "asset": asset_name,
            "timestamp": df.index[i],
            "price": current_close,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "actual": actual_direction,
            "correct": signal.direction == actual_direction,
            "rsi": signal.rsi,
        })

    return results


def analyze_threshold(results: list[dict], threshold: float) -> dict:
    """Analyze performance at a given confidence threshold."""
    filtered = [r for r in results if r["confidence"] >= threshold]

    if not filtered:
        return {
            "threshold": threshold,
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_pnl_cents": 0,
            "avg_pnl_per_bet": 0.0,
            "roi_pct": 0.0,
        }

    wins = sum(1 for r in filtered if r["correct"])
    total = len(filtered)
    win_rate = wins / total * 100

    # Simulate P&L with Kalshi-style payouts
    total_pnl = 0
    total_risked = 0
    for r in filtered:
        buy_price = min(85, r["confidence"] - 10)
        total_risked += buy_price
        if r["correct"]:
            total_pnl += (100 - buy_price)  # profit on win
        else:
            total_pnl -= buy_price  # loss on loss

    roi = (total_pnl / total_risked * 100) if total_risked > 0 else 0

    return {
        "threshold": threshold,
        "bets": total,
        "wins": wins,
        "win_rate": round(win_rate, 2),
        "total_pnl_cents": total_pnl,
        "avg_pnl_per_bet": round(total_pnl / total, 2) if total else 0,
        "roi_pct": round(roi, 2),
    }


def main():
    print("=" * 70)
    print("KALSHI CONFIDENCE PREDICTOR BACKTEST")
    print("6 months | 15m candles | BTC, ETH, SOL, XRP")
    print("=" * 70)

    fetcher = DataFetcher()
    predictor = KalshiPredictor()
    all_results = []

    # Fetch data for all assets
    for asset_name, symbol in ASSETS.items():
        print(f"\n--- {asset_name} ---")
        df_raw = fetch_6months(fetcher, symbol)
        if df_raw.empty:
            print(f"  Skipping {asset_name}: no data")
            continue

        print(f"  Running predictor on {len(df_raw)} candles ...")
        results = backtest_asset(asset_name, df_raw, predictor)
        print(f"  Generated {len(results)} signals")
        all_results.extend(results)

    if not all_results:
        print("\nNo results generated. Check data availability.")
        return

    # Per-asset breakdown
    print("\n" + "=" * 70)
    print("PER-ASSET RESULTS (threshold=50)")
    print("=" * 70)
    for asset_name in ASSETS:
        asset_results = [r for r in all_results if r["asset"] == asset_name]
        if not asset_results:
            continue
        stats = analyze_threshold(asset_results, 50)
        print(f"\n  {asset_name}: {stats['bets']} bets, "
              f"{stats['win_rate']}% win rate, "
              f"ROI: {stats['roi_pct']}%, "
              f"P&L: {stats['total_pnl_cents']}c")

    # Threshold sweep
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (all assets combined)")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Bets':>8} {'Wins':>8} {'Win%':>8} "
          f"{'P&L(c)':>10} {'Avg P&L':>10} {'ROI%':>10}")
    print("-" * 70)

    best = None
    for t in THRESHOLDS:
        stats = analyze_threshold(all_results, t)
        print(f"{stats['threshold']:>10} {stats['bets']:>8} {stats['wins']:>8} "
              f"{stats['win_rate']:>7.1f}% {stats['total_pnl_cents']:>10} "
              f"{stats['avg_pnl_per_bet']:>10.2f} {stats['roi_pct']:>9.1f}%")

        if stats["bets"] >= 10:  # need minimum sample size
            if best is None or stats["roi_pct"] > best["roi_pct"]:
                best = stats

    # Optimal threshold
    print("\n" + "=" * 70)
    if best:
        print(f"OPTIMAL THRESHOLD: {best['threshold']}")
        print(f"  Bets: {best['bets']}")
        print(f"  Win Rate: {best['win_rate']}%")
        print(f"  ROI: {best['roi_pct']}%")
        print(f"  Total P&L: {best['total_pnl_cents']} cents")
        print(f"  Avg P&L per bet: {best['avg_pnl_per_bet']} cents")
    else:
        print("No threshold with enough bets found.")

    # Per-asset at optimal threshold
    if best:
        print(f"\nPER-ASSET AT OPTIMAL THRESHOLD ({best['threshold']})")
        print("-" * 70)
        for asset_name in ASSETS:
            asset_results = [r for r in all_results if r["asset"] == asset_name]
            if not asset_results:
                continue
            stats = analyze_threshold(asset_results, best["threshold"])
            print(f"  {asset_name}: {stats['bets']} bets, "
                  f"{stats['win_rate']}% win rate, "
                  f"ROI: {stats['roi_pct']}%")

    print("\n" + "=" * 70)
    print("Backtest complete.")


if __name__ == "__main__":
    main()
