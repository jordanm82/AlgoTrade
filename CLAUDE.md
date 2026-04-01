# AlgoTrade — Claude Trading Operations Manual

Read this file completely before every trading session. It is your operational context.

## CRITICAL CODING RULES

**NO FALLBACK CODE. EVER.** If the correct data path fails, raise an error or skip the action. NEVER silently substitute different/stale/partial data. Every major bug in this project came from fallback code that hid errors and produced wrong predictions. Specifically:
- If a data fetch fails → skip the trade, do NOT use cached/stale data
- If a snapshot can't be built → return None, do NOT use raw partial candles
- `except Exception: pass` is BANNED — always log the error
- Never write `if X is None: use Y` where Y is a different data source
- The only acceptable fallback is "do nothing" (skip), NEVER "do the wrong thing"

**TRAINING/LIVE PARITY IS SACRED.** Every data path in live must exactly match training:
- Same price source (Coinbase+Bitstamp average for distance, NOT Coinbase-only)
- Same timing (minute 0 in both training and live, NOT minute 5 for training and minute 0 for live)
- Same labels (Kalshi settlement results, NOT Coinbase candle direction)
- Same strike source (Kalshi floor_strike, NOT Coinbase candle open)
- If you change ANY data path in live, you MUST retrain the model to match

## System Overview

Automated crypto trading system. Two exchanges: **Coinbase** (spot crypto) and **Kalshi** (15-minute prediction markets). Interact via **MCP tools** (`algotrade_*`).

## Architecture

```
dashboard_rich.py         — Rich terminal dashboard (primary UI)
cli/kalshi_daemon.py      — Kalshi K15 prediction daemon (minute-0 entry + active management)
exchange/kalshi.py        — Kalshi REST client
exchange/kalshi_ws.py     — Kalshi WebSocket (real-time contract prices + fill detection)
data/fetcher.py           — Coinbase via CCXT (market data)
data/brti_proxy.py        — Coinbase+Bitstamp averaged price (BRTI proxy)
data/brti_display.py      — CF Benchmarks scrape (dashboard display only, NOT for trading)
data/indicators.py        — RSI, BB, MACD, ATR, EMA
strategy/strategies/kalshi_predictor_v3.py — Strike-relative LogReg model
strategy/snapshot.py      — Minute-3 snapshot builder
config/settings.py        — API keys, risk limits
config/production.py      — Production settings
models/knn_kalshi.pkl     — Trained model (LogReg, strike-relative)
```

## Kalshi K15 UpDown Strategy (V3)

### Model — Strike-Relative LogReg

Predicts the actual Kalshi question: "Will price close above strike?"

**Training data sources (must match live exactly):**
- **Labels:** Kalshi settled market result (yes/no) — ground truth
- **Strike:** Kalshi floor_strike — actual settlement strike
- **Price at minute 0:** Coinbase + Bitstamp 5m candle OPEN averaged
- **Indicators:** Coinbase 15m/1h/4h candles
- **Training script:** `./venv/bin/python scripts/retrain_kalshi_labels.py`
- **Backtest script:** `./venv/bin/python scripts/backtest_kalshi_labels.py --days 30`
- Auto-triggered on daemon startup when model > 7 days old

**13 features:**
- `rsi_1h` (#1, +0.66) — hourly momentum direction
- `price_vs_ema` (#2, -0.39) — trend position
- `distance_from_strike` (#3, +0.29) — price vs strike at decision time
- Plus: macd_15m, norm_return, ema_slope, roc_5, macd_1h, hourly_return, trend_direction, vol_ratio, adx, rsi_4h

**Walk-forward validated:** 64% WR at minute 0, 0 losing days over 30 days on Kalshi ground truth.

### Execution Lifecycle — Minute-0 Entry + Active Management

| Time | State | Action |
|------|-------|--------|
| :00:03 | CONFIRMED | Bet immediately at window open (contracts ~50c) |
| :00-:04 | MONITORING | Stop loss active (50% of entry, checked every 15s) |
| :05:05 | CONFIRMATION | Recheck with 5m distance — hold or exit |
| :06-:09 | MONITORING | Holding confirmed position or already exited |
| :10+ | SETTLING | Cancel unfilled orders, await settlement |

**Stop loss:** 50% of entry price. Entry at 60c → stop at 30c. Checked every 15 seconds via WebSocket contract prices.

**Minute-5 confirmation:** Recompute model with actual distance_from_strike (price has moved by now). If market turned against us (> 0.1 ATR wrong direction) → sell position and cancel resting orders. If confirms → hold to settlement.

### Price Feeds (CRITICAL — must match training)

| Purpose | Source | Notes |
|---------|--------|-------|
| **Model input** | Coinbase+Bitstamp avg via CCXT | MUST match training. ~$5 from BRTI |
| **Dashboard PRICE column** | CF BRTI scrape | Display only, NOT for trading decisions. May be 30-60s stale |
| **Contract prices** | Kalshi WebSocket | Real-time YES/NO bid/ask. Used for stop loss + fill detection |
| **Strike** | Kalshi REST API (floor_strike) | Filtered to future markets only (prevents stale window match) |

**WARNING:** CF Benchmarks SSR scrape is NOT real-time. Never use for model input — caused 25% WR in live when used for distance calculation.

### Betting Rules

- Max entry price: **60c** (stop loss caps downside to 30c)
- Position sizing: **flat 5% of balance** per bet (CLI: `--maxsize=N`)
- Max concurrent bets: **configurable** (CLI: `--maxbets=N`, default 3)
- Highest confidence signals get priority when slots limited
- **No Kelly sizing, no tiered sizing** — consistent risk every bet

### Key Files

| File | Purpose |
|------|---------|
| `scripts/retrain_kalshi_labels.py` | Retrain model (Kalshi labels + multi-exchange price) |
| `scripts/backtest_kalshi_labels.py` | Backtest with same data sources as training |
| `data/store/trade_debug.jsonl` | Order lifecycle debug log (market select, fills, settlements) |
| `data/store/feature_log.jsonl` | Model prediction feature audit log |
| `models/knn_kalshi.pkl` | Trained model (auto-retrains when > 7 days old) |

### Running

```bash
# Dry-run (simulated bets, real market data)
python dashboard_rich.py --dry-run

# Live (real money)
python dashboard_rich.py --live --maxbets=1 --maxsize=2.5

# Demo (Kalshi demo exchange — tests order plumbing only, NOT model)
python dashboard_rich.py --demo
```

**WARNING:** Demo exchange has no real liquidity and prices don't track live. Only useful for testing order lifecycle, NOT for validating predictions.

### Series Tickers
`KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`

### Settlement
- Settles on **CF Benchmarks BRTI** — 60-second VWAP across Coinbase, Kraken, Bitstamp, Gemini, etc.
- Settlement check runs every 15 seconds with 15-second buffer after close_time
- Queries Kalshi `get_markets(status="settled")` for authoritative result
- Unfilled orders removed without W/L count

## Accounts

| Account | Purpose | Auth |
|---------|---------|------|
| **Coinbase** | Spot crypto + market data via CCXT | CDP API key (`cdp_api_key.json`) |
| **Kalshi** | K15 prediction markets | RSA key (`KalshiPrimaryKey.txt`, key ID in settings.py) |
| **Kalshi Demo** | Order lifecycle testing | RSA key (`KalshiDemoKeys.txt`, first line = API key ID) |

## Lessons Learned (from this session)

- **Train/live parity is everything.** If training uses 5m data at minute 5 but live decides at minute 1, WR drops from 75% to 54%. ALWAYS verify the exact same data path.
- **Price source matters.** Coinbase-only vs Coinbase+Bitstamp shifts distances enough to change predictions. The model MUST use the same price source as training.
- **Stale data kills WR.** CF Benchmarks SSR scrape was minutes stale — caused 25% WR in live. Use live API calls, never cached page data for trading decisions.
- **Fill timing is critical.** By minute 3-5, K15 contracts are already 70-90c. Minute 0 entry gets fills at ~50c.
- **Stop loss requires ticker field.** Dry-run bets must include `ticker` in the bet dict or stop loss silently skips them.
- **Kalshi market transitions take seconds.** At window boundary, query for future markets only (close_time > now) to avoid betting on the just-settled window.
- **Resting order cancel needs backstop.** Eval cycle is 50s — can miss minute 10. Dashboard 15s cycle is the guaranteed cancel path.
- **Demo exchange can't validate predictions.** Same strikes/settlements as production, but no real liquidity. Only tests order plumbing.
