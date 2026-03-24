"""Optimizer: Sweep per-pair BB Grid and RSI MR thresholds over 6 months.
Tests multiple threshold combos per pair and picks the best WR/return balance.
Then writes optimized config to config/pair_config.py."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
import time
import json
from strategy.compound_backtest import compound_backtest

exchange = ccxt.binanceus({"enableRateLimit": True})

def fetch_full_history(symbol, months=6):
    all_data = []
    candle_ms = 900000  # 15m
    now = int(time.time() * 1000)
    since = now - (months * 30 * 24 * 3600 * 1000)
    while since < now:
        try:
            batch = exchange.fetch_ohlcv(symbol, "15m", since=since, limit=1000)
            if not batch: break
            all_data.extend(batch)
            since = batch[-1][0] + candle_ms
            time.sleep(0.12)
        except: break
    if not all_data: return None
    df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    # Add indicators
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:,0]; df["macd_signal"] = macd.iloc[:,1]; df["macd_hist"] = macd.iloc[:,2]
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:,0]; df["bb_mid"] = bb.iloc[:,1]; df["bb_upper"] = bb.iloc[:,2]
    return df

ALL_PAIRS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","DOGE/USDT",
    "ADA/USDT","AVAX/USDT","LINK/USDT","DOT/USDT","SHIB/USDT",
    "LTC/USDT","UNI/USDT","ATOM/USDT","FIL/USDT",
]

# Fetch all data
print("="*100)
print("FETCHING 6 MONTHS OF 15m DATA FOR OPTIMIZATION")
print("="*100)
data = {}
for sym in ALL_PAIRS:
    df = fetch_full_history(sym)
    if df is not None and len(df) > 500:
        days = (df.index[-1] - df.index[0]).days
        data[sym] = df
        print(f"  {sym}: {len(df)} candles, {days} days")
    else:
        print(f"  {sym}: SKIP")

# Parameter grid
BB_RSI_BUYS = [28, 30, 33, 35, 38]
BB_RSI_SHORTS = [62, 65, 67, 70, 72]
LEVERAGES = [1, 2]

print(f"\nLoaded {len(data)} pairs. Optimizing...")

# Sweep all combos
all_results = []
for sym, df in data.items():
    best = {"ret": -999, "cfg": None}
    pair_results = []
    
    for bb_buy in BB_RSI_BUYS:
        for bb_short in BB_RSI_SHORTS:
            for lev in LEVERAGES:
                buy_fn = lambda r, p, b=bb_buy: (
                    pd.notna(r.get("bb_lower")) and r["close"] < r["bb_lower"]
                    and pd.notna(r.get("rsi")) and r["rsi"] < b)
                sell_fn = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] > r["bb_mid"]
                short_fn = lambda r, p, s=bb_short: (
                    pd.notna(r.get("bb_upper")) and r["close"] > r["bb_upper"]
                    and pd.notna(r.get("rsi")) and r["rsi"] > s)
                cover_fn = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] < r["bb_mid"]
                
                res = compound_backtest(df, buy_fn=buy_fn, sell_fn=sell_fn,
                    short_fn=short_fn, cover_fn=cover_fn,
                    initial_equity=1000, size_pct=0.10, leverage=lev,
                    commission=0.001, warmup=50)
                
                entry = {
                    "pair": sym, "strategy": "bb_grid",
                    "bb_rsi_buy": bb_buy, "bb_rsi_short": bb_short, "leverage": lev,
                    "trades": res["num_trades"], "win_rate": res["win_rate"],
                    "return_pct": res["total_return_pct"], "max_dd": res["max_drawdown_pct"],
                    "pf": res["profit_factor"],
                }
                pair_results.append(entry)
                all_results.append(entry)
                
                if res["win_rate"] >= 60 and res["num_trades"] >= 20 and res["total_return_pct"] > best["ret"]:
                    best = {"ret": res["total_return_pct"], "cfg": entry}
    
    # Also test RSI MR
    for rsi_os in [28, 30, 32, 35]:
        for rsi_ob in [65, 68, 70, 72]:
            for lev in LEVERAGES:
                buy_fn = lambda r, p, os=rsi_os: pd.notna(r.get("rsi")) and r["rsi"] < os
                sell_fn = lambda r, p, ob=rsi_ob: pd.notna(r.get("rsi")) and r["rsi"] > ob
                short_fn = lambda r, p, ob=rsi_ob+5: pd.notna(r.get("rsi")) and r["rsi"] > ob+5
                cover_fn = lambda r, p, os=rsi_os+5: pd.notna(r.get("rsi")) and r["rsi"] < os+5
                
                res = compound_backtest(df, buy_fn=buy_fn, sell_fn=sell_fn,
                    short_fn=short_fn, cover_fn=cover_fn,
                    initial_equity=1000, size_pct=0.10, leverage=lev,
                    commission=0.001, warmup=50)
                
                entry = {
                    "pair": sym, "strategy": "rsi_mr",
                    "bb_rsi_buy": rsi_os, "bb_rsi_short": rsi_ob, "leverage": lev,
                    "trades": res["num_trades"], "win_rate": res["win_rate"],
                    "return_pct": res["total_return_pct"], "max_dd": res["max_drawdown_pct"],
                    "pf": res["profit_factor"],
                }
                all_results.append(entry)
                
                if res["win_rate"] >= 60 and res["num_trades"] >= 20 and res["total_return_pct"] > best["ret"]:
                    best = {"ret": res["total_return_pct"], "cfg": entry}
    
    if best["cfg"]:
        b = best["cfg"]
        print(f"  {sym:12s} BEST: {b['strategy']} buy<{b['bb_rsi_buy']} short>{b['bb_rsi_short']} {b['leverage']}x "
              f"| n={b['trades']:4d} wr={b['win_rate']:5.1f}% ret={b['return_pct']:7.1f}% dd={b['max_dd']:5.1f}% pf={b['pf']:5.2f}")
    else:
        print(f"  {sym:12s} NO PROFITABLE CONFIG (wr>=60%, n>=20)")

# Save all results
rdf = pd.DataFrame(all_results)
rdf.to_csv("data/store/optimization_results.csv", index=False)

# Find best config per pair
print(f"\n{'='*100}")
print("OPTIMIZED CONFIG PER PAIR (best return where wr>=60%, trades>=20)")
print(f"{'='*100}")

optimized = {}
for sym in data.keys():
    pdf = rdf[(rdf["pair"]==sym) & (rdf["win_rate"]>=60) & (rdf["trades"]>=20)]
    if len(pdf) == 0:
        pdf = rdf[(rdf["pair"]==sym) & (rdf["win_rate"]>=55) & (rdf["trades"]>=10)]
    if len(pdf) == 0:
        print(f"  {sym:12s} -> DISABLED (no profitable config)")
        optimized[sym] = None
        continue
    best = pdf.sort_values("return_pct", ascending=False).iloc[0]
    optimized[sym] = best.to_dict()
    print(f"  {sym:12s} -> {best['strategy']:7s} buy<{best['bb_rsi_buy']:.0f} short>{best['bb_rsi_short']:.0f} "
          f"{best['leverage']:.0f}x | n={best['trades']:.0f} wr={best['win_rate']:.1f}% "
          f"ret={best['return_pct']:.1f}% dd={best['max_dd']:.1f}% pf={best['pf']:.2f}")

# Summary stats
print(f"\n{'='*100}")
print("PORTFOLIO PROJECTION (if all optimized pairs traded simultaneously)")
print(f"{'='*100}")
total_ret = 0
enabled = 0
for sym, cfg in optimized.items():
    if cfg is not None:
        total_ret += cfg["return_pct"]
        enabled += 1
avg_ret = total_ret / enabled if enabled > 0 else 0
print(f"  Enabled pairs: {enabled}/{len(data)}")
print(f"  Average per-pair return (6 months): {avg_ret:.1f}%")
print(f"  Sum of returns: {total_ret:.1f}%")
print(f"  With 10% sizing per pair, portfolio estimate: ~{avg_ret * 0.10 * enabled:.1f}%")

# Write optimized config
print(f"\n{'='*100}")
print("WRITING OPTIMIZED config/pair_config.py")
print(f"{'='*100}")

config_lines = ['"""Per-pair strategy configuration — optimized via 6-month parameter sweep."""\n\n']
config_lines.append("PAIR_CONFIG = {\n")
for sym, cfg in optimized.items():
    if cfg is None:
        config_lines.append(f'    # "{sym}": DISABLED — no profitable config found\n')
        continue
    strat = cfg["strategy"]
    bb_buy = int(cfg["bb_rsi_buy"])
    bb_short = int(cfg["bb_rsi_short"])
    lev = int(cfg["leverage"])
    wr = cfg["win_rate"]
    ret = cfg["return_pct"]
    n = int(cfg["trades"])
    config_lines.append(f'    "{sym}": {{\n')
    config_lines.append(f'        "bb_rsi_buy": {bb_buy}, "bb_rsi_short": {bb_short},\n')
    config_lines.append(f'        "rsi_mr_oversold": {bb_buy}, "rsi_mr_overbought": {bb_short + 5},\n')
    config_lines.append(f'        "rsi_mr_exit_long": {bb_short}, "rsi_mr_exit_short": {bb_buy + 5},\n')
    config_lines.append(f'        "leverage": {lev},\n')
    config_lines.append(f'        "enabled_strategies": ["{strat}"],\n')
    config_lines.append(f'        # 6mo: wr={wr:.1f}% ret={ret:.1f}% n={n} trades\n')
    config_lines.append(f'    }},\n')
config_lines.append("}\n\n")

config_lines.append('DEFAULT_PAIR_CONFIG = {\n')
config_lines.append('    "bb_rsi_buy": 30, "bb_rsi_short": 70,\n')
config_lines.append('    "rsi_mr_oversold": 28, "rsi_mr_overbought": 72,\n')
config_lines.append('    "rsi_mr_exit_long": 65, "rsi_mr_exit_short": 35,\n')
config_lines.append('    "leverage": 1,\n')
config_lines.append('    "enabled_strategies": ["bb_grid"],\n')
config_lines.append('}\n\n')

config_lines.append('def get_pair_config(symbol: str) -> dict:\n')
config_lines.append('    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)\n\n')

config_lines.append('ALL_PAIRS = [s for s in PAIR_CONFIG.keys()]\n')
config_lines.append('FUNDING_ARB_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]\n\n')

config_lines.append('COINBASE_MAP = {\n')
for sym in ALL_PAIRS:
    cb = sym.replace("/USDT", "-USD")
    config_lines.append(f'    "{sym}": "{cb}",\n')
config_lines.append('}\n')

with open("config/pair_config.py", "w") as f:
    f.writelines(config_lines)

print("  Written to config/pair_config.py")
print("\nDone.")
