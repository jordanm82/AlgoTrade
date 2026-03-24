"""Compare 1x vs 2x vs 3x leverage on our top optimized pairs over 6 months."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
import time
from strategy.compound_backtest import compound_backtest

exchange = ccxt.binanceus({"enableRateLimit": True})

def fetch_full_history(symbol, months=6):
    all_data = []
    candle_ms = 900000
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
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:,0]; df["bb_mid"] = bb.iloc[:,1]; df["bb_upper"] = bb.iloc[:,2]
    return df

# Our top 6 pairs (BB Grid with optimized 38/62 thresholds)
TOP_PAIRS = ["ATOM/USDT", "FIL/USDT", "DOT/USDT", "LTC/USDT", "UNI/USDT", "SHIB/USDT"]

print("Fetching 6 months of data for top pairs...")
data = {}
for sym in TOP_PAIRS:
    df = fetch_full_history(sym)
    if df is not None:
        data[sym] = df
        print(f"  {sym}: {len(df)} candles")

# BB Grid strategy with optimized thresholds
buy_fn = lambda r, p: (pd.notna(r.get("bb_lower")) and r["close"] < r["bb_lower"]
                        and pd.notna(r.get("rsi")) and r["rsi"] < 38)
sell_fn = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] > r["bb_mid"]
short_fn = lambda r, p: (pd.notna(r.get("bb_upper")) and r["close"] > r["bb_upper"]
                          and pd.notna(r.get("rsi")) and r["rsi"] > 62)
cover_fn = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] < r["bb_mid"]

results = []

print(f"\n{'='*110}")
print(f"LEVERAGE COMPARISON: 1x vs 2x vs 3x  (BB Grid 38/62, $1K start, 10% compounding, 6 months)")
print(f"{'='*110}")
print(f"\n{'PAIR':<12} {'LEV':>4} {'TRADES':>7} {'WR%':>6} {'RETURN':>10} {'$1K ->':>10} {'MAX DD':>8} {'PF':>6} {'WORST TRADE':>12}")
print(f"{'-'*85}")

for sym in TOP_PAIRS:
    if sym not in data:
        continue
    df = data[sym]
    
    for lev in [1, 2, 3]:
        res = compound_backtest(df, buy_fn=buy_fn, sell_fn=sell_fn,
            short_fn=short_fn, cover_fn=cover_fn,
            initial_equity=1000, size_pct=0.10, leverage=lev,
            commission=0.001, warmup=50)
        
        # Find worst single trade
        worst_trade = 0
        if res["trades"]:
            worst_trade = min(t["return_pct"] for t in res["trades"])
        
        results.append({
            "pair": sym, "leverage": lev,
            "trades": res["num_trades"], "win_rate": res["win_rate"],
            "return_pct": res["total_return_pct"], "final_equity": res["final_equity"],
            "max_dd": res["max_drawdown_pct"], "pf": res["profit_factor"],
            "worst_trade_pct": worst_trade,
        })
        
        print(f"{sym:<12} {lev:>3}x {res['num_trades']:>7} {res['win_rate']:>5.1f}% "
              f"{res['total_return_pct']:>9.1f}% ${res['final_equity']:>9.2f} "
              f"{res['max_drawdown_pct']:>7.1f}% {res['profit_factor']:>5.2f} "
              f"{worst_trade:>10.1f}%")
    print()

# Summary
rdf = pd.DataFrame(results)

print(f"\n{'='*110}")
print("AGGREGATE COMPARISON")
print(f"{'='*110}")
for lev in [1, 2, 3]:
    ldf = rdf[rdf["leverage"] == lev]
    print(f"\n  {lev}x LEVERAGE:")
    print(f"    Avg Return:     {ldf['return_pct'].mean():>8.1f}%")
    print(f"    Median Return:  {ldf['return_pct'].median():>8.1f}%")
    print(f"    Best Pair:      {ldf['return_pct'].max():>8.1f}%")
    print(f"    Worst Pair:     {ldf['return_pct'].min():>8.1f}%")
    print(f"    Avg Win Rate:   {ldf['win_rate'].mean():>8.1f}%")
    print(f"    Avg Max DD:     {ldf['max_dd'].mean():>8.1f}%")
    print(f"    Worst Max DD:   {ldf['max_dd'].min():>8.1f}%")
    print(f"    Worst Trade:    {ldf['worst_trade_pct'].min():>8.1f}%")
    total_equity = ldf['final_equity'].sum()
    print(f"    $6K portfolio:  ${total_equity:>8.0f}")

# Risk analysis
print(f"\n{'='*110}")
print("RISK ANALYSIS: What 3x leverage means for drawdowns")
print(f"{'='*110}")
print("""
  With 10% position sizing and 3% stop-loss:
  
  1x: A 3% adverse move costs 0.3% of equity per position
  2x: A 3% adverse move costs 0.6% of equity per position  
  3x: A 3% adverse move costs 0.9% of equity per position

  With 3 concurrent positions all stopping out simultaneously:
  1x: 0.9% equity loss
  2x: 1.8% equity loss
  3x: 2.7% equity loss  <-- still under 5% daily drawdown halt

  Liquidation risk (where leveraged losses exceed margin):
  1x: Impossible
  2x: Requires 50% adverse move — virtually impossible on 15m with 3% stop
  3x: Requires 33% adverse move — still impossible with 3% stop

  The real risk of 3x is NOT liquidation — it's amplified drawdowns
  eating into compounding equity, making recovery slower.
""")
