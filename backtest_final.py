"""Final comprehensive backtest: All 14 pairs x per-pair thresholds x BB Grid + RSI MR + Funding Arb.
Fetches 6 months of 15m data, uses per-pair config from config/pair_config.py,
runs compound_backtest with compounding equity, and reports results."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import pandas_ta as ta
import ccxt
import time
from strategy.compound_backtest import compound_backtest
from config.pair_config import get_pair_config, ALL_PAIRS, FUNDING_ARB_PAIRS
from strategy.strategies.funding_arb import FundingArb

# ---------------------------------------------------------------------------
# Exchange setup
# ---------------------------------------------------------------------------
exchange = ccxt.binanceus({"enableRateLimit": True})
funding_exchange = ccxt.binanceusdm({"enableRateLimit": True})

# ---------------------------------------------------------------------------
# Data fetching — paginated 6-month fetch
# ---------------------------------------------------------------------------
def fetch_full_history(symbol, timeframe="15m", months=6):
    """Fetch 6 months of data by paginating forward from target start."""
    all_data = []
    tf_ms = {"1m": 60000, "5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    now = int(time.time() * 1000)
    target_start = now - (months * 30 * 24 * 3600 * 1000)
    since = target_start
    batch_num = 0
    while since < now:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not batch:
                break
            all_data.extend(batch)
            since = batch[-1][0] + candle_ms
            batch_num += 1
            time.sleep(0.15)
        except Exception as e:
            print(f"    Batch fetch error at {pd.Timestamp(since, unit='ms')}: {e}")
            break
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df

def add_all_indicators(df):
    """Add RSI, ATR, MACD, Bollinger Bands, EMAs."""
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_signal"] = macd.iloc[:, 1]
        df["macd_hist"] = macd.iloc[:, 2]
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_lower"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_upper"] = bb.iloc[:, 2]
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    return df

# ---------------------------------------------------------------------------
# Fetch data for all 14 pairs
# ---------------------------------------------------------------------------
print("=" * 120)
print("FETCHING 6 MONTHS OF 15m DATA — 14 PAIRS")
print("=" * 120)

data = {}
for idx, sym in enumerate(ALL_PAIRS, 1):
    t0 = time.time()
    try:
        df = fetch_full_history(sym, "15m", months=6)
        if df is None or len(df) < 100:
            print(f"  [{idx:2d}/{len(ALL_PAIRS)}] {sym:12s}: SKIP (insufficient data)")
            continue
        df = add_all_indicators(df)
        data[sym] = df
        days = (df.index[-1] - df.index[0]).days
        elapsed = time.time() - t0
        print(f"  [{idx:2d}/{len(ALL_PAIRS)}] {sym:12s}: {len(df):6d} candles, {days:3d} days "
              f"({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}) "
              f"[{elapsed:.1f}s]")
    except Exception as e:
        print(f"  [{idx:2d}/{len(ALL_PAIRS)}] {sym:12s}: FAILED ({e})")
    time.sleep(0.2)

print(f"\nLoaded {len(data)}/{len(ALL_PAIRS)} pairs")

# ---------------------------------------------------------------------------
# Fetch current funding rates for arb pairs
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("FUNDING RATE CHECK (Binance USDM Perps)")
print("=" * 120)

funding_rates = {}
for sym in FUNDING_ARB_PAIRS:
    try:
        # CCXT uses the perp symbol format
        perp_sym = sym.replace("/USDT", "/USDT:USDT")
        fr = funding_exchange.fetch_funding_rate(perp_sym)
        rate = fr.get("fundingRate", 0)
        funding_rates[sym] = rate
        annualized = rate * 3 * 365 * 100
        print(f"  {sym:12s}: funding rate = {rate*100:.4f}% per 8h (~{annualized:.1f}% annualized)")
    except Exception as e:
        print(f"  {sym:12s}: Could not fetch funding rate ({e})")

# Analyze funding arb opportunities
arb = FundingArb()
print("\n  Funding Arb Signals:")
for sym, rate in funding_rates.items():
    price = float(data[sym].iloc[-1]["close"]) if sym in data else 0
    atr_val = float(data[sym].iloc[-1]["atr"]) if sym in data and pd.notna(data[sym].iloc[-1].get("atr")) else 0
    sigs = arb.check_funding(rate, price, atr_val)
    if sigs:
        for s in sigs:
            print(f"    {sym}: {s.metadata['reason']} -> {s.metadata['arb_type']} (strength={s.strength:.2f})")
    else:
        print(f"    {sym}: No arb opportunity (rate={rate*100:.4f}% within normal range)")

# ---------------------------------------------------------------------------
# Run backtests with per-pair config
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print("RUNNING 6-MONTH BACKTESTS — PER-PAIR THRESHOLDS")
print("=" * 120)

results = []

for sym in sorted(data.keys()):
    df = data[sym]
    cfg = get_pair_config(sym)
    days = (df.index[-1] - df.index[0]).days
    print(f"\n--- {sym} ({len(df)} candles, {days}d) | lev={cfg['leverage']}x "
          f"bb_rsi=[{cfg['bb_rsi_buy']}/{cfg['bb_rsi_short']}] "
          f"rsi=[{cfg['rsi_oversold']}/{cfg['rsi_overbought']}] ---")

    # ---- BB Grid (per-pair thresholds) ----
    if "bb_grid" in cfg["enabled_strategies"]:
        bb_buy = lambda r, p, c=cfg: (
            pd.notna(r.get("bb_lower")) and r["close"] < r["bb_lower"]
            and pd.notna(r.get("rsi")) and r["rsi"] < c["bb_rsi_buy"]
        )
        bb_sell = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] > r["bb_mid"]
        bb_short = lambda r, p, c=cfg: (
            pd.notna(r.get("bb_upper")) and r["close"] > r["bb_upper"]
            and pd.notna(r.get("rsi")) and r["rsi"] > c["bb_rsi_short"]
        )
        bb_cover = lambda r, p: pd.notna(r.get("bb_mid")) and r["close"] < r["bb_mid"]

        for sl_pct, sl_label in [(None, ""), (0.02, "_SL2%"), (0.03, "_SL3%")]:
            for lev in [cfg["leverage"]]:
                sname = f"BB_Grid_PP{sl_label}_{lev}x"
                try:
                    r = compound_backtest(
                        df, buy_fn=bb_buy, sell_fn=bb_sell,
                        short_fn=bb_short, cover_fn=bb_cover,
                        initial_equity=1000, size_pct=0.10,
                        leverage=lev, commission=0.001,
                        stop_loss_pct=sl_pct, warmup=50,
                    )
                    results.append({
                        "pair": sym, "strategy": sname, "leverage": lev,
                        "days": days, "candles": len(df),
                        "num_trades": r["num_trades"], "win_rate": r["win_rate"],
                        "total_return_pct": r["total_return_pct"],
                        "final_equity": r["final_equity"],
                        "max_drawdown_pct": r["max_drawdown_pct"],
                        "profit_factor": r["profit_factor"],
                        "avg_return_pct": r["avg_return_pct"],
                        "bb_rsi_buy": cfg["bb_rsi_buy"],
                        "bb_rsi_short": cfg["bb_rsi_short"],
                    })
                    flag = " ***" if r["win_rate"] >= 60 and r["num_trades"] >= 10 and r["total_return_pct"] > 3 else ""
                    print(f"  {sname:28s} | n={r['num_trades']:4d} wr={r['win_rate']:5.1f}% "
                          f"ret={r['total_return_pct']:8.2f}% eq=${r['final_equity']:9.2f} "
                          f"dd={r['max_drawdown_pct']:6.2f}% pf={r['profit_factor']:6.2f}{flag}")
                except Exception as e:
                    print(f"  {sname:28s} | ERROR: {e}")

    # ---- RSI Mean Reversion (per-pair thresholds, only for pairs with rsi_mr enabled) ----
    if "rsi_mr" in cfg.get("enabled_strategies", []):
        mr_oversold = cfg.get("rsi_mr_oversold", cfg["rsi_oversold"])
        mr_overbought = cfg.get("rsi_mr_overbought", cfg["rsi_overbought"])
        mr_exit_long = cfg.get("rsi_mr_exit_long", 65)
        mr_exit_short = cfg.get("rsi_mr_exit_short", 35)

        rsi_buy = lambda r, p, ov=mr_oversold: pd.notna(r.get("rsi")) and r["rsi"] < ov
        rsi_sell = lambda r, p, ex=mr_exit_long: pd.notna(r.get("rsi")) and r["rsi"] > ex
        rsi_short = lambda r, p, ob=mr_overbought: pd.notna(r.get("rsi")) and r["rsi"] > ob
        rsi_cover = lambda r, p, ex=mr_exit_short: pd.notna(r.get("rsi")) and r["rsi"] < ex

        for sl_pct, sl_label in [(None, ""), (0.02, "_SL2%")]:
            sname = f"RSI_MR_PP{sl_label}_{cfg['leverage']}x"
            try:
                r = compound_backtest(
                    df, buy_fn=rsi_buy, sell_fn=rsi_sell,
                    short_fn=rsi_short, cover_fn=rsi_cover,
                    initial_equity=1000, size_pct=0.10,
                    leverage=cfg["leverage"], commission=0.001,
                    stop_loss_pct=sl_pct, warmup=50,
                )
                results.append({
                    "pair": sym, "strategy": sname, "leverage": cfg["leverage"],
                    "days": days, "candles": len(df),
                    "num_trades": r["num_trades"], "win_rate": r["win_rate"],
                    "total_return_pct": r["total_return_pct"],
                    "final_equity": r["final_equity"],
                    "max_drawdown_pct": r["max_drawdown_pct"],
                    "profit_factor": r["profit_factor"],
                    "avg_return_pct": r["avg_return_pct"],
                    "bb_rsi_buy": None,
                    "bb_rsi_short": None,
                })
                flag = " ***" if r["win_rate"] >= 60 and r["num_trades"] >= 10 and r["total_return_pct"] > 3 else ""
                print(f"  {sname:28s} | n={r['num_trades']:4d} wr={r['win_rate']:5.1f}% "
                      f"ret={r['total_return_pct']:8.2f}% eq=${r['final_equity']:9.2f} "
                      f"dd={r['max_drawdown_pct']:6.2f}% pf={r['profit_factor']:6.2f}{flag}")
            except Exception as e:
                print(f"  {sname:28s} | ERROR: {e}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
rdf = pd.DataFrame(results)
rdf.to_csv("data/store/backtest_final_results.csv", index=False)
print(f"\nSaved {len(results)} results to data/store/backtest_final_results.csv")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 120)
print(f"FINAL BACKTEST RESULTS — {len(results)} combinations across {len(data)} pairs")
print("=" * 120)

if len(rdf) > 0:
    print(f"Positive return: {(rdf.total_return_pct > 0).sum()}/{len(rdf)} "
          f"({(rdf.total_return_pct > 0).mean()*100:.0f}%)")

    # Top 30
    print(f"\nTOP 30 (wr>=50%, trades>=5):")
    print(f'{"#":>2} {"Pair":>12} {"Strategy":>28} {"Lv":>2} {"Days":>4} {"N":>5} '
          f'{"WR%":>5} {"Ret%":>8} {"Equity":>10} {"DD%":>7} {"PF":>6}')
    print("-" * 100)
    top = rdf[(rdf.win_rate >= 50) & (rdf.num_trades >= 5)].sort_values(
        "total_return_pct", ascending=False).head(30)
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(f'{rank:2d} {r.pair:>12} {r.strategy:>28} {r.leverage:2.0f}x {r.days:4.0f}d '
              f'{r.num_trades:5.0f} {r.win_rate:5.1f} {r.total_return_pct:8.2f} '
              f'${r.final_equity:9.2f} {r.max_drawdown_pct:6.2f} {r.profit_factor:6.2f}')

    # Aggregate per strategy
    print("\nAGGREGATE BY STRATEGY:")
    # Strip trailing _1x/_2x for cleaner grouping
    rdf["strat_base"] = rdf["strategy"].str.replace(r"_\d+x$", "", regex=True)
    agg = rdf.groupby("strat_base").agg(
        pairs=("pair", "nunique"),
        avg_trades=("num_trades", "mean"),
        avg_wr=("win_rate", "mean"),
        avg_ret=("total_return_pct", "mean"),
        median_ret=("total_return_pct", "median"),
        best=("total_return_pct", "max"),
        worst=("total_return_pct", "min"),
        positive=("total_return_pct", lambda x: f"{(x>0).sum()}/{len(x)}"),
    ).round(2).sort_values("avg_ret", ascending=False)
    print(agg.to_string())

    # Aggregate per pair
    print("\nAGGREGATE BY PAIR (best strategy per pair):")
    best_per_pair = rdf.loc[rdf.groupby("pair")["total_return_pct"].idxmax()]
    best_per_pair = best_per_pair.sort_values("total_return_pct", ascending=False)
    for _, r in best_per_pair.iterrows():
        monthly = r.total_return_pct / max(r.days, 1) * 30
        print(f'  {r.pair:>12} | {r.strategy:>28} | ret={r.total_return_pct:7.2f}% '
              f'wr={r.win_rate:.0f}% n={r.num_trades:.0f} dd={r.max_drawdown_pct:.1f}% '
              f'~{monthly:.1f}%/mo')

    # Monthly projections for top combos
    print("\nMONTHLY PROJECTIONS (top combos, wr>=55%, n>=10):")
    top_proj = rdf[(rdf.win_rate >= 55) & (rdf.num_trades >= 10) & (rdf.total_return_pct > 0)
                   ].sort_values("total_return_pct", ascending=False).head(15)
    for _, r in top_proj.iterrows():
        monthly = r.total_return_pct / max(r.days, 1) * 30
        annual = r.total_return_pct / max(r.days, 1) * 365
        print(f'  {r.pair:>12} {r.strategy:>28} | {r.total_return_pct:7.2f}% in {r.days:.0f}d '
              f'= ~{monthly:.1f}%/mo ~{annual:.0f}%/yr  '
              f'(wr={r.win_rate:.0f}% n={r.num_trades:.0f} dd={r.max_drawdown_pct:.1f}%)')

    # Funding arb summary
    print("\n" + "=" * 120)
    print("FUNDING RATE ARBITRAGE — LIVE OPPORTUNITY SUMMARY")
    print("=" * 120)
    if funding_rates:
        for sym, rate in funding_rates.items():
            annualized = rate * 3 * 365 * 100
            sigs = arb.check_funding(rate, 0, 0)
            status = "OPPORTUNITY" if sigs else "No signal"
            print(f"  {sym:12s}: {rate*100:.4f}%/8h (~{annualized:.1f}% ann) -> {status}")
        print("\n  Note: Funding arb is delta-neutral (hedged). Returns come from funding")
        print("  payments, not price moves. Available for live trading on BTC/ETH/SOL perps.")
    else:
        print("  Could not fetch funding rates. Strategy available for live trading.")
else:
    print("No results generated.")

print("\nDone.")
