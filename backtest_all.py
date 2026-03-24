#!/usr/bin/env python3
"""
Comprehensive Multi-Strategy Backtest
--------------------------------------
Tests 10 strategy variants x 15 pairs x 2 timeframes = 300 combinations
using the compound_backtest engine (compounding equity, long/short, leverage, stop-losses).

Strategies:
  1. RSI_MR_Long          - buy RSI<30, sell RSI>65
  2. RSI_MR_Short         - short RSI>70, cover RSI<35
  3. RSI_MR_Long+Short    - both combined
  4. RSI_MR_L+S_SL2%      - same with 2% stop loss
  5. MACD_RSI_Long        - MACD cross up + RSI<40 buy, MACD cross down + RSI>60 sell
  6. MACD_RSI_Long+Short  - add short side
  7. MACD_RSI_L+S_2xLev   - same with 2x leverage
  8. MACD_RSI_L+S_3xLev   - same with 3x leverage
  9. BB_Grid_Long+Short   - Bollinger Band grid: long off lower, short off upper
  10. BB_Grid_L+S_2xLev   - same with 2x leverage
"""

import sys
import os
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from strategy.compound_backtest import compound_backtest

# ── Configuration ────────────────────────────────────────────────────────
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "SHIB/USDT", "LTC/USDT", "UNI/USDT", "ATOM/USDT", "FIL/USDT",
]
TIMEFRAMES = ["15m", "1h"]
CANDLE_LIMIT = 1000
INITIAL_EQUITY = 1000
SIZE_PCT = 0.10
COMMISSION = 0.001
WARMUP = 50

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "data", "store", "multi_strategy_results.csv")

# ── Data Fetching ────────────────────────────────────────────────────────
print("=" * 90)
print("COMPREHENSIVE MULTI-STRATEGY BACKTEST")
print(f"Pairs: {len(PAIRS)} | Timeframes: {TIMEFRAMES} | Strategies: 10 | Total combos: 300")
print("=" * 90)

exchange = ccxt.binanceus({"enableRateLimit": True})

# Pre-fetch all available markets
print("\n[*] Loading BinanceUS markets...")
try:
    markets = exchange.load_markets()
    available_pairs = [p for p in PAIRS if p in markets and markets[p].get("active")]
    skipped_pairs = [p for p in PAIRS if p not in available_pairs]
    if skipped_pairs:
        print(f"  WARNING: Skipping unavailable pairs: {skipped_pairs}")
    print(f"  Available: {len(available_pairs)}/{len(PAIRS)} pairs")
except Exception as e:
    print(f"  FATAL: Could not load markets: {e}")
    sys.exit(1)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, Bollinger Bands, ATR, and EMAs."""
    out = df.copy()

    # RSI
    out["rsi"] = ta.rsi(out["close"], length=14)

    # MACD
    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    out["macd"] = macd.iloc[:, 0]
    out["macd_hist"] = macd.iloc[:, 1]
    out["macd_signal"] = macd.iloc[:, 2]

    # Bollinger Bands
    bbands = ta.bbands(out["close"], length=20, std=2)
    out["bb_lower"] = bbands.iloc[:, 0]
    out["bb_mid"] = bbands.iloc[:, 1]
    out["bb_upper"] = bbands.iloc[:, 2]

    # ATR
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    # EMAs
    out["ema_9"] = ta.ema(out["close"], length=9)
    out["ema_21"] = ta.ema(out["close"], length=21)

    return out


# Fetch all data
datasets = {}  # key: (pair, timeframe)

for pair in available_pairs:
    for tf in TIMEFRAMES:
        key = (pair, tf)
        try:
            raw = exchange.fetch_ohlcv(pair, tf, limit=CANDLE_LIMIT)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            df = add_all_indicators(df)
            datasets[key] = df
            print(f"  [OK] {pair:12s} {tf:4s} -> {len(df)} candles  ({df.index[0]} to {df.index[-1]})")
        except Exception as e:
            print(f"  [!!] {pair:12s} {tf:4s} -> FAILED: {e}")
        time.sleep(0.15)

print(f"\nFetched {len(datasets)} datasets out of {len(available_pairs) * len(TIMEFRAMES)} requested.\n")

if not datasets:
    print("No data fetched. Aborting.")
    sys.exit(1)


# ── Signal Functions ─────────────────────────────────────────────────────
def _safe(row, col):
    """Get a value safely, returning NaN if missing."""
    v = row.get(col) if hasattr(row, "get") else getattr(row, col, float("nan"))
    return v


# --- RSI Mean Reversion ---
def rsi_buy_30(row, prev):
    r = _safe(row, "rsi")
    return not pd.isna(r) and r < 30

def rsi_sell_65(row, prev):
    r = _safe(row, "rsi")
    return not pd.isna(r) and r > 65

def rsi_short_70(row, prev):
    r = _safe(row, "rsi")
    return not pd.isna(r) and r > 70

def rsi_cover_35(row, prev):
    r = _safe(row, "rsi")
    return not pd.isna(r) and r < 35


# --- MACD + RSI Confluence ---
def macd_cross_up(row, prev):
    """MACD crosses above signal (current > signal, prev <= signal)."""
    m = _safe(row, "macd")
    s = _safe(row, "macd_signal")
    pm = _safe(prev, "macd")
    ps = _safe(prev, "macd_signal")
    if any(pd.isna(x) for x in [m, s, pm, ps]):
        return False
    return m > s and pm <= ps

def macd_cross_down(row, prev):
    """MACD crosses below signal."""
    m = _safe(row, "macd")
    s = _safe(row, "macd_signal")
    pm = _safe(prev, "macd")
    ps = _safe(prev, "macd_signal")
    if any(pd.isna(x) for x in [m, s, pm, ps]):
        return False
    return m < s and pm >= ps

def macd_rsi_buy(row, prev):
    return macd_cross_up(row, prev) and _safe(row, "rsi") < 40

def macd_rsi_sell(row, prev):
    r = _safe(row, "rsi")
    return macd_cross_down(row, prev) and not pd.isna(r) and r > 60

def macd_rsi_short(row, prev):
    r = _safe(row, "rsi")
    return macd_cross_down(row, prev) and not pd.isna(r) and r > 70

def macd_rsi_cover(row, prev):
    r = _safe(row, "rsi")
    return macd_cross_up(row, prev) and not pd.isna(r) and r < 30


# --- Bollinger Band Grid ---
def bb_grid_buy(row, prev):
    c = _safe(row, "close")
    bbl = _safe(row, "bb_lower")
    r = _safe(row, "rsi")
    if any(pd.isna(x) for x in [c, bbl, r]):
        return False
    return c < bbl and r < 35

def bb_grid_sell(row, prev):
    c = _safe(row, "close")
    bbm = _safe(row, "bb_mid")
    if pd.isna(c) or pd.isna(bbm):
        return False
    return c > bbm

def bb_grid_short(row, prev):
    c = _safe(row, "close")
    bbu = _safe(row, "bb_upper")
    r = _safe(row, "rsi")
    if any(pd.isna(x) for x in [c, bbu, r]):
        return False
    return c > bbu and r > 65

def bb_grid_cover(row, prev):
    c = _safe(row, "close")
    bbm = _safe(row, "bb_mid")
    if pd.isna(c) or pd.isna(bbm):
        return False
    return c < bbm


# ── Strategy Definitions ────────────────────────────────────────────────
STRATEGIES = [
    {
        "name": "RSI_MR_Long",
        "buy_fn": rsi_buy_30, "sell_fn": rsi_sell_65,
        "short_fn": None, "cover_fn": None,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "RSI_MR_Short",
        "buy_fn": None, "sell_fn": None,
        "short_fn": rsi_short_70, "cover_fn": rsi_cover_35,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "RSI_MR_Long+Short",
        "buy_fn": rsi_buy_30, "sell_fn": rsi_sell_65,
        "short_fn": rsi_short_70, "cover_fn": rsi_cover_35,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "RSI_MR_L+S_SL2%",
        "buy_fn": rsi_buy_30, "sell_fn": rsi_sell_65,
        "short_fn": rsi_short_70, "cover_fn": rsi_cover_35,
        "leverage": 1, "stop_loss_pct": 0.02,
    },
    {
        "name": "MACD_RSI_Long",
        "buy_fn": macd_rsi_buy, "sell_fn": macd_rsi_sell,
        "short_fn": None, "cover_fn": None,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "MACD_RSI_Long+Short",
        "buy_fn": macd_rsi_buy, "sell_fn": macd_rsi_sell,
        "short_fn": macd_rsi_short, "cover_fn": macd_rsi_cover,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "MACD_RSI_L+S_2xLev",
        "buy_fn": macd_rsi_buy, "sell_fn": macd_rsi_sell,
        "short_fn": macd_rsi_short, "cover_fn": macd_rsi_cover,
        "leverage": 2, "stop_loss_pct": None,
    },
    {
        "name": "MACD_RSI_L+S_3xLev",
        "buy_fn": macd_rsi_buy, "sell_fn": macd_rsi_sell,
        "short_fn": macd_rsi_short, "cover_fn": macd_rsi_cover,
        "leverage": 3, "stop_loss_pct": None,
    },
    {
        "name": "BB_Grid_Long+Short",
        "buy_fn": bb_grid_buy, "sell_fn": bb_grid_sell,
        "short_fn": bb_grid_short, "cover_fn": bb_grid_cover,
        "leverage": 1, "stop_loss_pct": None,
    },
    {
        "name": "BB_Grid_L+S_2xLev",
        "buy_fn": bb_grid_buy, "sell_fn": bb_grid_sell,
        "short_fn": bb_grid_short, "cover_fn": bb_grid_cover,
        "leverage": 2, "stop_loss_pct": None,
    },
]


# ── Run All Backtests ────────────────────────────────────────────────────
all_results = []
combo_num = 0
total_combos = len(datasets) * len(STRATEGIES)

print("=" * 110)
print(f"{'#':>4}  {'Pair':12s} {'TF':4s} {'Strategy':24s} {'Trades':>6s} {'WR%':>6s} {'Return%':>9s} "
      f"{'Final$':>9s} {'MaxDD%':>8s} {'PF':>6s} {'Flag':>4s}")
print("-" * 110)

for (pair, tf), df in sorted(datasets.items()):
    for strat in STRATEGIES:
        combo_num += 1

        try:
            res = compound_backtest(
                df,
                buy_fn=strat["buy_fn"],
                sell_fn=strat["sell_fn"],
                short_fn=strat["short_fn"],
                cover_fn=strat["cover_fn"],
                initial_equity=INITIAL_EQUITY,
                size_pct=SIZE_PCT,
                leverage=strat["leverage"],
                commission=COMMISSION,
                stop_loss_pct=strat["stop_loss_pct"],
                warmup=WARMUP,
            )

            wr = res["win_rate"]
            nt = res["num_trades"]
            ret = res["total_return_pct"]
            feq = res["final_equity"]
            mdd = res["max_drawdown_pct"]
            pf = res["profit_factor"]

            # Flag strong combos
            flag = ""
            if wr >= 55 and nt >= 5 and ret > 1:
                flag = "***"

            print(f"{combo_num:4d}  {pair:12s} {tf:4s} {strat['name']:24s} "
                  f"{nt:6d} {wr:6.1f} {ret:9.2f} {feq:9.2f} {mdd:8.2f} {pf:6.2f} {flag}")

            row = {
                "pair": pair,
                "timeframe": tf,
                "strategy": strat["name"],
                "num_trades": nt,
                "win_rate": wr,
                "total_return_pct": ret,
                "final_equity": feq,
                "max_drawdown_pct": mdd,
                "avg_return_pct": res["avg_return_pct"],
                "profit_factor": pf,
                "leverage": strat["leverage"],
                "stop_loss": strat["stop_loss_pct"] or 0,
            }
            all_results.append(row)

        except Exception as e:
            print(f"{combo_num:4d}  {pair:12s} {tf:4s} {strat['name']:24s}  ERROR: {e}")
            all_results.append({
                "pair": pair,
                "timeframe": tf,
                "strategy": strat["name"],
                "num_trades": 0,
                "win_rate": 0,
                "total_return_pct": 0,
                "final_equity": INITIAL_EQUITY,
                "max_drawdown_pct": 0,
                "avg_return_pct": 0,
                "profit_factor": 0,
                "leverage": strat["leverage"],
                "stop_loss": strat["stop_loss_pct"] or 0,
            })

# ── Save Results ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\nResults saved to {RESULTS_CSV}")
print(f"Total combinations run: {len(all_results)}")


# ── TOP 25 by total_return_pct (where wr>=55% and trades>=3) ────────────
print("\n" + "=" * 110)
print("TOP 25 COMBOS by total_return_pct  (win_rate >= 55%, trades >= 3)")
print("=" * 110)

qualified = results_df[(results_df["win_rate"] >= 55) & (results_df["num_trades"] >= 3)]
top25 = qualified.sort_values("total_return_pct", ascending=False).head(25)

if len(top25) == 0:
    print("  No combos meet the filter criteria (wr>=55%, trades>=3).")
else:
    print(f"{'#':>3}  {'Pair':12s} {'TF':4s} {'Strategy':24s} {'Trades':>6s} {'WR%':>6s} "
          f"{'Return%':>9s} {'Final$':>9s} {'MaxDD%':>8s} {'PF':>6s}")
    print("-" * 100)
    for rank, (_, row) in enumerate(top25.iterrows(), 1):
        print(f"{rank:3d}  {row['pair']:12s} {row['timeframe']:4s} {row['strategy']:24s} "
              f"{row['num_trades']:6.0f} {row['win_rate']:6.1f} {row['total_return_pct']:9.2f} "
              f"{row['final_equity']:9.2f} {row['max_drawdown_pct']:8.2f} {row['profit_factor']:6.2f}")


# ── Aggregate Stats Per Strategy ─────────────────────────────────────────
print("\n" + "=" * 110)
print("AGGREGATE STATS PER STRATEGY (averaged across all pairs & timeframes)")
print("=" * 110)

agg = results_df.groupby("strategy").agg(
    avg_trades=("num_trades", "mean"),
    avg_wr=("win_rate", "mean"),
    avg_return=("total_return_pct", "mean"),
    avg_maxdd=("max_drawdown_pct", "mean"),
    avg_pf=("profit_factor", "mean"),
    max_return=("total_return_pct", "max"),
    min_return=("total_return_pct", "min"),
    total_positive=("total_return_pct", lambda x: (x > 0).sum()),
    count=("total_return_pct", "count"),
).reset_index()

# Sort by strategy order as defined
strat_order = [s["name"] for s in STRATEGIES]
agg["sort_key"] = agg["strategy"].apply(lambda x: strat_order.index(x) if x in strat_order else 99)
agg = agg.sort_values("sort_key").drop(columns=["sort_key"])

print(f"{'Strategy':24s} {'AvgTr':>6s} {'AvgWR%':>7s} {'AvgRet%':>8s} {'AvgDD%':>8s} "
      f"{'AvgPF':>6s} {'MaxRet%':>8s} {'MinRet%':>8s} {'Win/Tot':>8s}")
print("-" * 100)

for _, row in agg.iterrows():
    print(f"{row['strategy']:24s} {row['avg_trades']:6.1f} {row['avg_wr']:7.1f} "
          f"{row['avg_return']:8.2f} {row['avg_maxdd']:8.2f} {row['avg_pf']:6.2f} "
          f"{row['max_return']:8.2f} {row['min_return']:8.2f} "
          f"{int(row['total_positive']):3d}/{int(row['count']):3d}")

print("\n" + "=" * 110)
print("BACKTEST COMPLETE")
print("=" * 110)
