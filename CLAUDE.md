# AlgoTrade — Claude Trading Operations Manual

Read this file completely before every trading session. It is your operational context.

## CRITICAL CODING RULES

**NO FALLBACK CODE. EVER.** If the correct data path fails, raise an error or skip the action. NEVER silently substitute different/stale/partial data. Every major bug in this project came from fallback code that hid errors and produced wrong predictions. Specifically:
- If a data fetch fails → skip the trade, do NOT use cached/stale data
- If a snapshot can't be built → return None, do NOT use raw partial candles
- `except Exception: pass` is BANNED — always log the error
- Never write `if X is None: use Y` where Y is a different data source
- The only acceptable fallback is "do nothing" (skip), NEVER "do the wrong thing"

## System Overview

You are the AI decision engine for an automated crypto trading system. You interact with it via **MCP tools** (`algotrade_*`) — no bash commands needed. The system trades on two exchanges: **Coinbase** (spot crypto) and **Kalshi** (15-minute prediction markets). Each exchange is its own source of truth for orders, positions, and balances.

**Your role:** Monitor via MCP tools, understand what the strategies are doing, intervene when context demands it, and make judgment calls the mechanical system cannot.

## MCP Tools (Primary Interface)

You have 24 tools registered via `.claude/mcp.json`. Use these instead of bash commands:

**Session Management:**
- `algotrade_start` — start live or dry-run (mode, cycles)
- `algotrade_stop` — graceful stop
- `algotrade_force_kill` — SIGKILL when stop doesn't work
- `algotrade_status` — full dashboard state + daemon process status

**Account & Positions:**
- `algotrade_balances` — Coinbase + Kalshi balances with USD token valuations
- `algotrade_positions` — open positions with live P&L
- `algotrade_performance` — session stats (win rate, P&L, trades)
- `algotrade_trade_history` — recent trade records

**Trading:**
- `algotrade_buy` — spot buy on Coinbase (symbol, usd_amount)
- `algotrade_sell` — spot sell on Coinbase (symbol, amount or sell_all)
- `algotrade_close_all` — exit everything on both platforms
- `algotrade_kalshi_bet` — place a 15-min prediction bet (asset, side, amount_usd)
- `algotrade_kalshi_markets` — list available 15-min contracts with prices

**Market Data:**
- `algotrade_ticker` — current price for any pair
- `algotrade_indicators` — RSI, BB, MACD, ATR for any pair + timeframe
- `algotrade_orderbook` — order book imbalance, spread, wall detection
- `algotrade_tradeflow` — net flow, buy ratio, large trade bias
- `algotrade_funding_rates` — perp funding rates
- `algotrade_signals` — all signals firing RIGHT NOW with confidence scores

**Analysis & Config:**
- `algotrade_backtest` — run a quick backtest (pair, strategy, timeframe, days)
- `algotrade_logs` — view recent daemon stdout
- `algotrade_errors` — scan logs for errors/warnings/failures/blocks
- `algotrade_config_get` — all current settings
- `algotrade_config_set` — adjust thresholds, toggle pairs live

## Architecture

```
mcp_server.py             — MCP server (24 tools, stdio transport)
.claude/mcp.json          — MCP server registration
dashboard.py              — runs the daemon + prints status every 1 min
cli/live_daemon.py        — production daemon (BB Grid + RSI MR on 15m + Kalshi)
exchange/coinbase.py      — Coinbase Advanced Trade execution (source of truth for spot)
exchange/kalshi.py        — Kalshi REST client (source of truth for predictions)
exchange/positions.py     — position tracker with persistence + profit taking
data/fetcher.py           — BinanceUS (market data via CCXT)
data/market_data.py       — order book imbalance + trade flow (leading indicators)
data/indicators.py        — RSI, BB, MACD, ATR, EMA (lagging indicators)
data/store.py             — parquet OHLCV, CSV trades, JSON snapshots
strategy/strategies/      — BB Grid, RSI MR, SMA Crossover, Funding Arb, Kalshi Predictor
strategy/compound_backtest.py — backtester with compounding, long/short, leverage
config/pair_config.py     — per-pair optimized thresholds (from 6-month parameter sweep)
config/settings.py        — risk limits
config/production.py      — production settings
```

## Exchange Source of Truth

**Coinbase is source of truth for all spot orders:**
- Before selling: query `get_token_balance()` for actual amount, sell that
- After buying: verify `get_token_balance()` that tokens arrived
- Stop-loss/TP sell: query real balance, not tracker estimate
- Position reconciliation: every signal cycle, compare tracker vs exchange

**Kalshi is source of truth for all prediction bets:**
- Before betting: verify `get_balance()` for available funds
- After betting: check `get_order_status()` and `get_positions()` for fills
- Balance checks: always from `get_balance()`, never internal tracking

## Strategies

### Strategy 1: BB Grid Long+Short (per-pair optimized)

Buys when price < BB lower AND RSI < threshold. Exits when price > BB middle. Each pair has individually tuned thresholds from 6-month parameter sweep.

**Tiered leverage (data-driven from 6-month backtest):**

| Pair | Leverage | BB Buy < | BB Short > | 6mo WR | 6mo Return |
|------|----------|----------|------------|--------|------------|
| ATOM | **3x** | 38 | 62 | 88.4% | +1812% |
| DOT | **3x** | 38 | 62 | 76.4% | +200% |
| LTC | **3x** | 38 | 62 | 76.1% | +156% |
| FIL | 2x | 38 | 62 | 71.5% | +325% |
| UNI | 2x | 38 | 62 | 67.6% | +76% |
| SHIB | 2x | 38 | 62 | 74.4% | +54% |

### Strategy 2: RSI Mean Reversion Long+Short

| Pair | Leverage | Oversold | Overbought | 6mo WR | 6mo Return |
|------|----------|----------|------------|--------|------------|
| SOL | 2x | 32 | 73 | 67.6% | +2.2% |
| AVAX | 2x | 28 | 70 | 69.1% | +6.2% |

### Strategy 3: Kalshi 15-Minute Predictions

Three predictor versions available via `--predictor v1|v2|v3`:

**V3 (recommended) — Strike-Relative + KNN Hybrid Model:**

Two-factor prediction: **LR (LogReg)** for direction + **TEK (probability table)** for confirmation.

**LR — LogReg Direction Model (primary signal):**
- 15-feature LogisticRegression trained on multi-timeframe Coinbase data (15m + 1h + 4h)
- Features: RSI, StochRSI, MACD, normalized returns, volume ratio, BB position, EMA slope, ADX, ROC, plus price_vs_ema, hourly_return, trend_direction
- Walk-forward validated: 60.9% WR solo, **78% WR with TEK** on 11,498 out-of-sample bets
- Trained with `class_weight='balanced'` — produces both YES and NO predictions
- Retrained via `scripts/retrain_walkforward.py` (walk-forward: train on older data only)

**TEK — Probability Table Confluence (filter):**
- Pre-computed 2D probability table (distance_ATR × time_remaining)
- Built from historical 1m data: `./venv/bin/python scripts/build_prob_table.py --days 90`
- At minute 5: uses actual Coinbase price vs Kalshi strike to compute distance
- Called with `force_table=True` to ensure table lookup (not another LR prediction)
- Adds +17pp WR by filtering bets where price hasn't moved to confirm LR direction
- Technical adjustments: OB (±5%), trade flow (±5%), 1h trend (±5%), MACD (+3%), RSI extreme (-8%), RSI divergence (-8%), momentum gate (-15%)

**Bet decision (2-factor gate):**
- LR predicts direction: YES when prob >= 55%, NO when prob <= 45%
- TEK confirms: probability table score >= 30% for predicted side
- Both must agree for bet execution
- **Bets BOTH sides** — YES and NO
- Edge margin: 2c minimum edge over implied contract price
- Max contract price: 85c hard cap
- Price source: Coinbase (closest to CF Benchmarks BRTI settlement)

**Position sizing (tiered by confidence):**

| Confidence | Max Risk (% of balance) |
|-----------|------------------------|
| 70%+ | 10% |
| 65-69% | 7.5% |
| 60-64% | 5% |
| 55-59% | 2.5% |
| Below 55% | No bet |

**Evaluation lifecycle (wall-clock aligned at :01, :06, :11, :12):**
- **SETUP (min 0-4):** Score direction. No betting.
- **OBSERVING (min 5-9):** Re-score with fresh data. No betting.
- **CONFIRMED (min 10-11):** Place bets if signal is strong.
- **LAST_LOOK (min 12):** Final chance, elevated threshold.
- **SETTLING (min 13+):** Bets awaiting settlement.

**Settlement tracking:** Queries Kalshi API for authoritative settlement results (not approximated from price data). Tracks W/L/WR and P&L for both dry-run and live modes.

**Model refresh:**
- LogReg model (walk-forward): `./venv/bin/python scripts/retrain_walkforward.py --days 179`
  - Trains on 120 days oldest data, validates on 59 days most recent (out-of-sample)
  - Shows LR × TEK threshold grid with WR and P&L
  - Auto-triggered on daemon startup when model > 7 days old
- Probability table: `./venv/bin/python scripts/build_prob_table.py --days 90`
- Refresh weekly/monthly to stay current with market regime

**Series tickers:** `KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`, `KXBNB15M`

**V1 — Mean-Reversion (legacy, 63% WR):** Original predictor. 14 scoring components + 3 filters. Still functional via `--predictor v1`.

**V2 — Continuation (shelved, 46% WR):** Trend-following approach. Proved that continuation signals don't predict 15m direction. Available via `--predictor v2` but not recommended.

**Betting rules:**
- Per-asset thresholds (above) — not a single global threshold
- **Hard 50c entry cap** — never pay above 50c (R:R would be < 1:1)
- Hold to settlement — entry price IS total risk per contract
- Sizing: risk budget = 5% of Kalshi balance, count = budget / entry_price
- Pricing: query orderbook, bid between bid and ask based on spread width
- **Max 3 concurrent Kalshi bets** (separate from spot position limit)
- Priority: highest confidence signals first when slots are limited

**Payout structure (hold to settlement):**

| Entry | Risk per contract | Profit if win | R:R |
|-------|-------------------|---------------|-----|
| 30c | $0.30 | $0.70 | 2.3:1 |
| 40c | $0.40 | $0.60 | 1.5:1 |
| 50c | $0.50 | $0.50 | 1:1 |
| >50c | SKIP | — | <1:1 |

**API:** `https://api.elections.kalshi.com` (RSA-PSS auth, no newlines in signature, DIGEST_LENGTH salt)

### Disabled Pairs (net negative in 6-month backtest)
BTC, ETH, XRP, DOGE, ADA, LINK — disabled from Coinbase trading but BTC/ETH/SOL/XRP/BNB data still fetched for Kalshi predictions.

## Risk Controls (Hardcoded)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Position size | 10% of current equity | Compounds as equity grows |
| Max leverage | 3x on ATOM/DOT/LTC, 2x on rest | Per-pair from backtest |
| Max concurrent positions | 3 | Highest confidence takes priority |
| Max concurrent Kalshi bets | 3 | Separate from spot limit, confidence-ranked |
| Stop-loss | 3% from entry | Enforced every 1 minute |
| Daily drawdown halt | 5% | All trading stops |
| Min balance | $10 | Lowered from $100 for small accounts |
| Kalshi max entry | 50c | Never pay above 50c (R:R < 1:1) |
| Kalshi risk per bet | 5% of balance | ~$4-5 per bet |

## Leading Indicator Gate

Every BUY and SHORT signal is validated against real-time order book and trade flow before execution:

- **BLOCKED** if order book imbalance AND trade flow strongly contradict the signal direction
- **CONFIRMED** if both align with the signal
- **NEUTRAL** otherwise (proceed)
- Close signals are NEVER blocked — always let exits execute
- If data fetch fails, trade proceeds (fail-open)

This prevented us from buying ATOM during active selling (flow=-0.53) — the position would have been underwater.

## Profit Taking & Trailing Stops

Positions have automatic profit-taking built into the PositionTracker:
- **+10% gain** → sell 25% of position
- **+20% gain** → sell another 25%
- **Remaining 50%** → trailing stop at 5% below peak price, OR RSI exit signal
- Trailing stop activates after first TP level hit

## Position Persistence

Positions survive daemon restarts:
- Saved to `data/store/positions.json` on every open/close/reduce
- Loaded on startup
- Exchange sync on startup: queries Coinbase balances and Kalshi positions for any untracked holdings
- Reconciliation every signal cycle: warns if tracker and exchange disagree

## Accounts

| Account | Purpose | Auth |
|---------|---------|------|
| **Coinbase** | Spot crypto trading | CDP API key (`cdp_api_key.json`) |
| **Kalshi** | 15-min prediction markets | RSA key (`KalshiPrimaryKey.txt`, API key ID in settings.py) |

**Note:** Coinbase perp trading returns 403 (Permission Denied) — account doesn't have perp access enabled. Short signals on Coinbase will fail. Only long (spot buy) trades execute on Coinbase. Kalshi is where we take directional short bets.

## Monitoring

Use MCP tools for monitoring — no bash commands or sleep needed:
- `algotrade_status` — read dashboard state instantly
- `algotrade_balances` — both account balances
- `algotrade_positions` — open positions with P&L
- `algotrade_errors` — scan for issues
- `algotrade_logs` — raw daemon output

**Monitoring cadence:**
- Active management: check `algotrade_status` every few minutes
- Passive: let daemon run, check `algotrade_performance` periodically
- After issues: check `algotrade_errors` then `algotrade_logs`

## When to Intervene

### Let the system trade when:
- RSI is in normal signal zones (25-38 for buys, 62-75 for shorts, per pair)
- Market is RANGING (regime indicator)
- Win rate above 60%
- Leading indicators not contradicting

### Override or pause when:
- **BTC dumps >5% in 1 hour** — all alts follow, skip buy signals
- **Win rate below 50%** — regime shift from ranging to trending
- **Multiple positions all losing** — correlation risk, close weakest
- **RSI stays below 20 or above 80 for multiple candles** — trending, not reverting
- **Leading indicators show heavy selling on ALL pairs** — market-wide selloff

### How to intervene:
1. `algotrade_stop` or `algotrade_force_kill`
2. `algotrade_sell` or `algotrade_close_all`
3. `algotrade_start` to restart

## Key Files

| File | Purpose |
|------|---------|
| `data/store/dashboard.log` | Latest dashboard frame |
| `data/store/trades.csv` | All trade history |
| `data/store/positions.json` | Persisted open positions |
| `data/store/snapshots/` | JSON snapshots every 15 min |
| `/tmp/dashboard_stdout.log` | Daemon stdout/stderr |
| `/tmp/dashboard_pid.txt` | Running daemon PID |
| `config/pair_config.py` | Per-pair thresholds (editable) |

## Known Limitations

1. **Mean reversion fails in strong trends.** Designed for ranging/choppy markets.
2. **Coinbase perps return 403** — no short execution via Coinbase. Use Kalshi for directional down bets.
3. **Synced positions use current price as entry** — P&L may be under-reported for positions opened before a restart.
4. **BinanceUS data, Coinbase execution** — minor price differences possible.
5. **15m timeframe** — signals happen every few hours, not minutes. Be patient.
6. **Kalshi 15-min contracts** — thin liquidity on some assets. Wide spreads mean our smart pricing helps but fills aren't guaranteed.

## Session Checklist

Before every trading session:
- [ ] Read this file
- [ ] `algotrade_balances` — check both accounts
- [ ] `algotrade_errors` — any issues from last session?
- [ ] `algotrade_trade_history` — review recent trades
- [ ] `algotrade_signals` — what's firing right now?
- [ ] `algotrade_start` with dry-run for 1 cycle to confirm
- [ ] Switch to live when confident
- [ ] Monitor with `algotrade_status` periodically
- [ ] After session: `algotrade_performance` for review
