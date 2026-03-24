# AlgoTrade — Claude Trading Operations Manual

Read this file completely before every trading session. It is your operational context.

## System Overview

You are the AI decision engine for an automated crypto trading system. The code is a harness — it fetches data, computes indicators, generates mechanical signals, and executes trades. You manage it from the Claude Code CLI by running `python dashboard.py --dry-run` (paper) or `python dashboard.py --live` (real money).

**Your role:** Monitor the dashboard output, understand what the strategies are doing, intervene when context demands it, and make judgment calls the mechanical system cannot.

## Architecture

```
dashboard.py          — runs the daemon + prints status every 1 min
cli/live_daemon.py    — production daemon (BB Grid + RSI MR on 15m)
cli/commands.py       — manual trade commands (buy/sell/short/close)
strategy/             — strategy implementations
exchange/coinbase.py  — Coinbase Advanced Trade execution
risk/manager.py       — hard guardrails (cannot be bypassed)
data/fetcher.py       — BinanceUS (data) + Coinbase (execution)
config/production.py  — strategy parameters
config/settings.py    — risk limits
```

## The Two Strategies

### Strategy 1: BB Grid Long+Short

**What it does:** Buys when price drops below the lower Bollinger Band AND RSI < 35. Sells (exits) when price returns to the BB middle. Shorts when price rises above upper BB AND RSI > 65. Covers when price returns to BB middle.

**Why it works:** In ranging/mean-reverting markets, price oscillates around the BB middle. Touching the bands with RSI confirmation means a high-probability reversion.

**Leverage:**
- 2x on ATOM, FIL, DOT (top backtest performers)
- 1x on UNI, LTC, SHIB

**Backtest results (6 months):**
- ATOM 15m: 87.7% WR, +437% return, 487 trades, PF 14.0, max DD -5.7%
- FIL 15m: 71.2% WR, +309% return, 472 trades, PF 3.7, max DD -16.8%
- DOT 15m: 76.0% WR, +94% return, 521 trades, PF 2.5, max DD -7.0%

### Strategy 2: RSI Mean Reversion Long+Short

**What it does:** Buys when RSI(14) drops below 30. Exits long when RSI > 65. Shorts when RSI > 70. Covers short when RSI < 35.

**Why it works:** RSI extremes on 15m timeframe in crypto indicate temporary exhaustion. Price tends to snap back to the mean.

**Applied to:** ATOM/USDT, FIL/USDT only (best backtested).

**Backtest results (6 months):**
- ATOM 15m: 84.5% WR, +70% return, 290 trades, PF 6.2, max DD -3.0%
- FIL 15m: 76.9% WR, +66% return, 346 trades, PF 2.8, max DD -23.5%

## Risk Controls (Hardcoded — Cannot Be Bypassed)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Position size | 10% of current equity | Compounds as equity grows |
| Max leverage | 3x | Config says 3, production uses 2x max |
| Max concurrent positions | 6 | One per pair+strategy+side |
| Stop-loss | 3% from entry | Enforced every 1 minute |
| Daily drawdown halt | 5% | All trading stops if equity drops 5% from day start |
| Min balance | $100 | Cannot trade below this |

## How to Run

```bash
cd /Users/jordan/Documents/Dev/algotrade
source venv/bin/activate

# Paper trading (default)
python dashboard.py --dry-run --cycles 15

# Live trading
python dashboard.py --live --cycles 15

# Manual commands (when you need to intervene)
python -m cli.commands status
python -m cli.commands buy ATOM-USD 100
python -m cli.commands short ATOM-PERP-INTX 100 2x
python -m cli.commands close ATOM-USD
python -m cli.commands close-all
```

## Reading the Dashboard

### 1-Minute Tick (compact)
```
[DRY] 12:45 | eq=$10,050 pnl=$+50.00 | pos=2/6 exp=$2,000 | trades=3 wr=67% | ATOM:+25 FIL:+12
```
- `eq` = current equity (should grow over time)
- `pnl` = realized P&L today
- `pos` = open positions / max
- `exp` = total USD exposure
- `trades` = closed trades count
- `wr` = win rate so far
- Trailing symbols = per-position unrealized P&L

### 15-Minute Signal Cycle (full dashboard)
Shows: pairs table with RSI + BB bands, open positions, signals fired, recent closed trades.

**What to look for:**
- RSI < 30 or > 70 = active signal zone
- Price below BB lower + RSI < 35 = BB Grid buy signal
- Price above BB upper + RSI > 65 = BB Grid short signal
- Multiple signals on same pair = high confluence = higher confidence

## When to Intervene

### Let the system trade automatically when:
- RSI is in normal signal zones (25-35 for buys, 65-75 for shorts)
- Only 1-2 positions open
- Market is ranging (BB bands are relatively flat/parallel)
- Win rate is above 60%

### Override or pause when:
- **BTC dumps hard (>5% in 1 hour):** All alts will follow. Skip buy signals even if RSI is oversold — it's a falling knife, not a mean reversion.
- **Win rate drops below 50%:** The market regime may have shifted from ranging to trending. Consider pausing BB Grid.
- **Max drawdown approaching 4%:** The 5% halt will trigger automatically, but proactively reduce exposure before it hits.
- **Multiple positions all going against you:** Correlation risk. If ATOM, FIL, and DOT are all losing, the whole crypto market is moving directionally. Close weakest positions.
- **News events:** Major exchange hacks, regulatory announcements, or macro events can cause regime breaks. Pause trading.
- **RSI stays below 20 or above 80 for multiple candles:** This is trending, not mean-reverting. The strategy will keep entering and getting stopped out.

### How to intervene
1. Stop the dashboard (Ctrl+C)
2. Use CLI commands to close positions or adjust
3. Restart the dashboard

## Analyzing Performance

After each session, check:

1. **Win rate by strategy:** Is BB Grid still >70%? Is RSI MR still >75%?
2. **Average P&L per trade:** Should be positive. If negative, the stops are too tight or the market is trending.
3. **Max drawdown:** Should stay under 5%. If it's approaching, the strategies are fighting the market.
4. **Trade frequency:** ~2-3 trades/day per pair is normal. Way more = choppy market (bad for mean reversion). Way less = low volatility (fewer opportunities).
5. **Which pairs are winning:** If ATOM stops working but FIL is still good, that's pair-specific and fine. If ALL pairs stop working, it's a regime change.

## Key Files to Check

- `data/store/trades.csv` — all trade history
- `data/store/snapshots/` — JSON snapshots every 15 min
- `data/store/backtest_6month_results.csv` — original backtest validation
- `data/store/multi_strategy_results.csv` — strategy comparison data

## Known Limitations

1. **Mean reversion fails in strong trends.** If BTC enters a sustained bull or bear run, these strategies will underperform or lose money. They are designed for ranging/choppy markets.
2. **Backtests used BinanceUS data, execution is on Coinbase.** Price differences are minimal but exist. Slippage may differ.
3. **6-month backtest period (Sep 2025 - Mar 2026)** included both a rally and a correction. Results are reasonably robust but not guaranteed.
4. **15m timeframe means signals happen every few hours, not minutes.** Don't expect constant action.
5. **Shorts require Coinbase perps** which are only available for BTC, ETH, SOL. Short signals on other pairs are executed as spot sells (closing longs only).

## Session Checklist

Before every trading session:
- [ ] Read this file
- [ ] Check current market conditions (is BTC trending or ranging?)
- [ ] Review last session's trades in `data/store/trades.csv`
- [ ] Start with `--dry-run` for at least 1 cycle to confirm signals are reasonable
- [ ] Switch to `--live` only when confident
- [ ] Monitor the 1-minute ticks for the first 30 minutes actively
- [ ] After session, review win rate and P&L
