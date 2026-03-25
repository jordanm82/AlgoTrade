#!/usr/bin/env python3
"""Iterative backtest for Kalshi 15m predictor optimization.

Runs multiple iterations to measure the effect of each improvement:
1. Baseline — current predictor with filters active
2. Without filters — same predictor but skip _apply_filters()
3. With 1h MTF — add multi-timeframe confirmation
4. Threshold sweep — find optimal confidence threshold

Uses realistic P&L model:
  entry_price = min(50, confidence - 10) cents
  5% of balance per bet, compounding

Usage:
    ./venv/bin/python backtest_kalshi.py [--days N] [--threshold T]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.strategies.kalshi_predictor import KalshiPredictor


ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "BNB": "BNB/USDT",
}

TIMEFRAME_15M = "15m"
TIMEFRAME_1H = "1h"


def fetch_candles(fetcher: DataFetcher, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch N days of candles via paginated requests."""
    all_frames = []
    now_ms = int(time.time() * 1000)
    period_ms = days * 24 * 60 * 60 * 1000
    since = now_ms - period_ms
    batch_size = 1000

    if timeframe == "5m":
        candle_ms = 5 * 60 * 1000
    elif timeframe == "15m":
        candle_ms = 15 * 60 * 1000
    elif timeframe == "1h":
        candle_ms = 60 * 60 * 1000
    else:
        candle_ms = 15 * 60 * 1000

    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                break
            all_frames.append(df)
            last_ts = int(df.index[-1].timestamp() * 1000)
            since = last_ts + candle_ms
            if len(df) < batch_size:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"    Warning: fetch error: {e}")
            since += batch_size * candle_ms
            time.sleep(1)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def run_predictor_on_candles(
    asset_name: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    predictor: KalshiPredictor,
    use_filters: bool = True,
    use_mtf: bool = True,
) -> list[dict]:
    """Run predictor on every candle, return list of signal results.

    Iteration isolation:
    - use_filters=False: monkey-patch _apply_filters to always return False
    - use_mtf=False: pass df_1h=None to score()
    """
    df = add_indicators(df_15m.copy())
    df_1h_ind = None
    if df_1h is not None and use_mtf:
        df_1h_ind = add_indicators(df_1h.copy())

    results = []

    # Save original filter method
    original_filters = predictor._apply_filters
    if not use_filters:
        predictor._apply_filters = lambda *args, **kwargs: False

    try:
        for i in range(50, len(df) - 1):  # 50 candle warmup for indicators
            window = df.iloc[:i + 1]

            # Find matching 1h candle
            mtf_window = None
            if df_1h_ind is not None and not df_1h_ind.empty:
                current_time = df.index[i]
                mask = df_1h_ind.index <= current_time
                if mask.sum() >= 20:
                    mtf_window = df_1h_ind.loc[mask]

            signal = predictor.score(window, df_1h=mtf_window)

            if signal is None:
                continue

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
    finally:
        predictor._apply_filters = original_filters

    return results


def simulate_pnl(
    results: list[dict],
    threshold: int,
    starting_balance: float = 100.0,
    risk_pct: float = 0.05,
    fixed_entry: int | None = None,
    max_concurrent: int = 3,
) -> dict:
    """Simulate realistic P&L with compounding and concurrency limits.

    Models real trading constraints:
    - Max 3 concurrent Kalshi bets (each settles in 15 minutes)
    - Only 1 bet per asset per 15m window (no duplicate bets)
    - 5% of balance risked per bet
    - Entry price capped at 50c

    Args:
        fixed_entry: If set, use this fixed entry price (cents) instead of confidence-based.
        max_concurrent: Max concurrent active bets (default 3).
    """
    balance = starting_balance
    peak_balance = starting_balance
    max_drawdown = 0.0
    total_wins = 0
    total_bets = 0
    gross_wins = 0.0
    gross_losses = 0.0
    per_asset = {}

    filtered = [r for r in results if r["confidence"] >= threshold]

    # Group signals by timestamp, rank by confidence within each group
    from collections import defaultdict
    by_time = defaultdict(list)
    for r in filtered:
        by_time[r["timestamp"]].append(r)

    # Sort timestamps, and within each timestamp sort by confidence descending
    sorted_signals = []
    for ts in sorted(by_time.keys()):
        group = sorted(by_time[ts], key=lambda r: r["confidence"], reverse=True)
        sorted_signals.extend(group)

    # Track active bets: list of (settle_time, asset)
    active_bets = []

    for r in sorted_signals:
        ts = r["timestamp"]

        # Prune settled bets (15 min settlement)
        active_bets = [(t, a) for t, a in active_bets
                       if (ts - t).total_seconds() < 900]

        # Skip if at max concurrent bets
        if len(active_bets) >= max_concurrent:
            continue

        # Skip if already have an active bet on this asset
        active_assets = {a for _, a in active_bets}
        if r["asset"] in active_assets:
            continue

        if fixed_entry is not None:
            entry_cents = fixed_entry
        else:
            entry_cents = min(50, r["confidence"] - 10)

        if entry_cents > 50 or entry_cents <= 0:
            continue

        risk_budget_cents = int(balance * risk_pct * 100)  # convert dollars to cents
        if risk_budget_cents < entry_cents:
            continue

        num_contracts = min(risk_budget_cents // entry_cents, 20)  # cap at 20 contracts
        if num_contracts < 1:
            continue

        total_bets += 1
        asset = r["asset"]
        if asset not in per_asset:
            per_asset[asset] = {"bets": 0, "wins": 0}
        per_asset[asset]["bets"] += 1

        # Track this bet as active
        active_bets.append((ts, asset))

        if r["correct"]:
            profit_cents = num_contracts * (100 - entry_cents)
            balance += profit_cents / 100  # cents to dollars
            gross_wins += profit_cents / 100
            total_wins += 1
            per_asset[asset]["wins"] += 1
        else:
            loss_cents = num_contracts * entry_cents
            balance -= loss_cents / 100
            gross_losses += loss_cents / 100

        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if balance <= 0:
            break

    win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
    roi = ((balance - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf') if gross_wins > 0 else 0

    per_asset_wr = {}
    for a, stats in per_asset.items():
        per_asset_wr[a] = round(stats["wins"] / stats["bets"] * 100, 1) if stats["bets"] > 0 else 0

    return {
        "threshold": threshold,
        "total_bets": total_bets,
        "wins": total_wins,
        "win_rate": round(win_rate, 1),
        "final_balance": round(balance, 2),
        "roi_pct": round(roi, 1),
        "max_drawdown": round(max_drawdown, 1),
        "profit_factor": round(profit_factor, 2),
        "per_asset_wr": per_asset_wr,
        "per_asset_bets": {a: s["bets"] for a, s in per_asset.items()},
    }


def print_metrics(name: str, m: dict):
    """Print formatted metrics for one iteration."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Win Rate:       {m['win_rate']}%  ({m['wins']}/{m['total_bets']} bets)")
    print(f"  Final Balance:  ${m['final_balance']:.2f}  (from $100)")
    print(f"  ROI:            {m['roi_pct']}%")
    print(f"  Max Drawdown:   {m['max_drawdown']}%")
    print(f"  Profit Factor:  {m['profit_factor']}")
    if m['per_asset_wr']:
        print(f"  Per-Asset WR:   ", end="")
        parts = [f"{a}: {wr}% ({m['per_asset_bets'].get(a, 0)})" for a, wr in sorted(m['per_asset_wr'].items())]
        print(", ".join(parts))



def main():
    parser = argparse.ArgumentParser(description="Kalshi predictor iterative backtest")
    parser.add_argument("--days", type=int, default=30, help="Backtest period in days (default: 30)")
    parser.add_argument("--threshold", type=int, default=30, help="Default confidence threshold (default: 30)")
    args = parser.parse_args()

    print("=" * 60)
    print("KALSHI PREDICTOR ITERATIVE BACKTEST")
    print(f"{args.days} days | 15m candles | {', '.join(ASSETS.keys())}")
    print("=" * 60)

    fetcher = DataFetcher()
    predictor = KalshiPredictor()

    # Fetch all data once
    data_15m = {}
    data_1h = {}

    for asset_name, symbol in ASSETS.items():
        print(f"\nFetching {asset_name}...")
        df_15m = fetch_candles(fetcher, symbol, TIMEFRAME_15M, args.days)
        if df_15m.empty:
            print(f"  Skipping {asset_name}: no 15m data")
            continue
        data_15m[asset_name] = df_15m
        print(f"  15m: {len(df_15m)} candles ({df_15m.index[0]} to {df_15m.index[-1]})")

        df_1h = fetch_candles(fetcher, symbol, TIMEFRAME_1H, args.days)
        if not df_1h.empty:
            data_1h[asset_name] = df_1h
            print(f"  1h:  {len(df_1h)} candles")

    if not data_15m:
        print("\nNo data fetched. Check connectivity.")
        return

    # ── Iteration 1: With filters, no MTF ──
    print("\n" + "=" * 60)
    print("ITERATION 1: Filters ON, no MTF")
    print("=" * 60)

    results_filtered_no_mtf = []
    for asset_name, df_15m in data_15m.items():
        r = run_predictor_on_candles(asset_name, df_15m, None, predictor,
                                     use_filters=True, use_mtf=False)
        results_filtered_no_mtf.extend(r)

    m1 = simulate_pnl(results_filtered_no_mtf, args.threshold)
    print_metrics("Filters ON, no MTF", m1)

    # ── Iteration 2: Without filters, no MTF ──
    print("\n" + "=" * 60)
    print("ITERATION 2: Filters OFF, no MTF (baseline comparison)")
    print("=" * 60)

    results_no_filters = []
    for asset_name, df_15m in data_15m.items():
        r = run_predictor_on_candles(asset_name, df_15m, None, predictor,
                                     use_filters=False, use_mtf=False)
        results_no_filters.extend(r)

    m2 = simulate_pnl(results_no_filters, args.threshold)
    print_metrics("Filters OFF, no MTF", m2)

    # ── Iteration 3: With filters + MTF ──
    print("\n" + "=" * 60)
    print("ITERATION 3: Filters ON + 1h MTF")
    print("=" * 60)

    results_full = []
    for asset_name, df_15m in data_15m.items():
        df_1h = data_1h.get(asset_name)
        r = run_predictor_on_candles(asset_name, df_15m, df_1h, predictor,
                                     use_filters=True, use_mtf=True)
        results_full.extend(r)

    m3 = simulate_pnl(results_full, args.threshold)
    print_metrics("Filters ON + 1h MTF", m3)

    # ── Iteration 4: Threshold sweep on best version ──
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP (Filters ON + MTF)")
    print("=" * 60)
    print(f"{'Thresh':>7} {'Bets':>6} {'Wins':>6} {'WR%':>7} {'Balance':>10} {'ROI%':>8} {'MaxDD%':>8} {'PF':>6}")
    print("─" * 60)

    best = None
    for t in range(25, 75, 5):
        m = simulate_pnl(results_full, t)
        if m['total_bets'] > 0:
            print(f"{t:>7} {m['total_bets']:>6} {m['wins']:>6} {m['win_rate']:>6.1f}% "
                  f"${m['final_balance']:>8.2f} {m['roi_pct']:>7.1f}% {m['max_drawdown']:>7.1f}% "
                  f"{m['profit_factor']:>5.2f}")
            if m['total_bets'] >= 5:
                if best is None or m['roi_pct'] > best['roi_pct']:
                    best = m

    # ── Sensitivity: fixed 40c entry ──
    print("\n" + "=" * 60)
    print("SENSITIVITY: Fixed 40c entry price")
    print("=" * 60)

    if best:
        m_fixed = simulate_pnl(results_full, best['threshold'], fixed_entry=40)
        print_metrics(f"Fixed 40c @ threshold={best['threshold']}", m_fixed)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n  {'Iteration':<35} {'WR%':>7} {'Bets':>6} {'ROI%':>8} {'PF':>6}")
    print(f"  {'─' * 55}")
    print(f"  {'Filters OFF, no MTF':<35} {m2['win_rate']:>6.1f}% {m2['total_bets']:>6} {m2['roi_pct']:>7.1f}% {m2['profit_factor']:>5.2f}")
    print(f"  {'Filters ON, no MTF':<35} {m1['win_rate']:>6.1f}% {m1['total_bets']:>6} {m1['roi_pct']:>7.1f}% {m1['profit_factor']:>5.2f}")
    print(f"  {'Filters ON + 1h MTF':<35} {m3['win_rate']:>6.1f}% {m3['total_bets']:>6} {m3['roi_pct']:>7.1f}% {m3['profit_factor']:>5.2f}")

    if best:
        print(f"\n  Optimal threshold: {best['threshold']}")
        print(f"  At optimal: {best['win_rate']}% WR, {best['total_bets']} bets, "
              f"{best['roi_pct']}% ROI, PF={best['profit_factor']}")
        print(f"\n  Per-asset at optimal:")
        for a, wr in sorted(best['per_asset_wr'].items()):
            bets = best['per_asset_bets'].get(a, 0)
            print(f"    {a}: {wr}% WR ({bets} bets)")

    # ── V2 Continuation Predictor comparison ──
    print("\n" + "=" * 60)
    print("V2 CONTINUATION PREDICTOR (vs V1 baseline)")
    print("=" * 60)

    from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
    predictor_v2 = KalshiPredictorV2()

    # Run V2 on same data
    results_v2 = []
    for asset_name, df_15m in data_15m.items():
        df_1h = data_1h.get(asset_name)
        r = run_predictor_on_candles(asset_name, df_15m, df_1h, predictor_v2,
                                     use_filters=True, use_mtf=True)
        results_v2.extend(r)

    # Per-asset breakdown with V1 thresholds
    PER_ASSET_THRESH_V2 = {"BTC": 30, "ETH": 35, "SOL": 35, "XRP": 30, "BNB": 35}
    print("\n  Per-asset (using V1 thresholds as starting point):")
    for asset_name in ASSETS:
        r_asset = [r for r in results_v2 if r["asset"] == asset_name]
        thresh = PER_ASSET_THRESH_V2.get(asset_name, 30)
        filtered = [r for r in r_asset if r["confidence"] >= thresh]
        if filtered:
            wins = sum(1 for r in filtered if r["correct"])
            wr = wins / len(filtered) * 100
            print(f"    {asset_name}: {wr:.1f}% WR ({len(filtered)} bets) @ threshold {thresh}")

    # V2 threshold sweep
    print(f"\n  V2 THRESHOLD SWEEP:")
    print(f"  {'Thresh':>7} {'Bets':>6} {'Wins':>6} {'WR%':>7} {'PF':>6}")
    print(f"  {'─' * 35}")
    for t in [20, 25, 30, 35, 40, 45, 50, 55]:
        m = simulate_pnl(results_v2, t)
        if m['total_bets'] > 0:
            print(f"  {t:>7} {m['total_bets']:>6} {m['wins']:>6} {m['win_rate']:>6.1f}% {m['profit_factor']:>5.2f}")

    # V2 overall at threshold 30
    m_v2 = simulate_pnl(results_v2, 30)
    print_metrics("V2 Continuation @ threshold=30", m_v2)

    print("\n" + "=" * 60)
    print("Backtest complete.")


if __name__ == "__main__":
    main()
