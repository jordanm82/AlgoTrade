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
- Same features (all 33 M0 / 39 M10 features computed identically in training, backtest, and live)
- If you change ANY data path in live, you MUST retrain the model to match

**NO RESTING BUY ORDERS.** If the ask > MAX_ENTRY (60c), skip the bet. NEVER place a resting limit order below the market ask hoping it will fill — this is adverse selection. Fills on resting orders below market systematically lose because they only fill when the market turns against the bet.

## System Overview

Automated crypto trading system. Two exchanges: **Coinbase** (spot crypto) and **Kalshi** (15-minute prediction markets). Interact via **MCP tools** (`algotrade_*`).

## Architecture

```
dashboard_rich.py         — Rich terminal dashboard (primary UI)
cli/kalshi_daemon.py      — Kalshi K15 prediction daemon (minute-0 entry + M10 exit)
exchange/kalshi.py        — Kalshi REST client (unwraps order envelope)
exchange/kalshi_ws.py     — Kalshi WebSocket (real-time contract prices + fill detection)
data/fetcher.py           — Coinbase via CCXT (market data)
data/brti_proxy.py        — Coinbase+Bitstamp averaged price (BRTI proxy)
data/brti_display.py      — CF Benchmarks scrape (dashboard display only, NOT for trading)
data/indicators.py        — RSI, BB, MACD, ATR, EMA
strategy/strategies/kalshi_predictor_v3.py — Per-asset XGBoost model loader (33/39 features)
strategy/snapshot.py      — Minute-3 snapshot builder
config/settings.py        — API keys, risk limits
config/production.py      — Production settings
models/m0_{btc,eth,sol,xrp}.pkl — Per-asset XGBoost M0 entry models (33 features each)
models/m10_{btc,eth,sol,xrp}.pkl — Per-asset XGBoost M10 exit models (39 features each)
```

## Kalshi K15 UpDown Strategy (V3)

### Model — Per-Asset XGBoost with Cross-Asset Confluence + Regime Detection

Predicts the actual Kalshi question: "Will price close above strike?" Separate XGBoost model per asset (BTC, ETH, SOL, XRP). XGBoost replaces LogReg because it can learn conditional relationships (e.g., "below EMA in downtrend → NO" vs LogReg's fixed "below EMA → always YES").

**Training data sources (must match live exactly):**
- **Labels:** Kalshi settled market result (yes/no) — ground truth
- **Strike:** Kalshi floor_strike — actual settlement strike
- **Price at minute 0:** Coinbase + Bitstamp 5m candle OPEN averaged
- **Indicators:** Coinbase 15m/1h/4h candles (previous completed candle; in-progress candle dropped)
- **Training data:** 12 months synthetic data (~160K samples) + Kalshi settlements
- **Training script:** `./venv/bin/python scripts/retrain_kalshi_labels.py`
- **Backtest script:** `./venv/bin/python scripts/backtest_kalshi_labels.py --days 60 --threshold 65 --m10-threshold 80 --exit-min 10 --exit-max 25`
- Auto-triggered on daemon startup when model > 7 days old

**M0 entry model — 33 features (XGBoost, no rsi_15m):**

*Anti-overfitting:* subsample=0.7, colsample_bytree=0.7, learning_rate=0.03, max_depth=4, min_child_weight=50, early stopping on 15% validation set. Trained on 12 months synthetic data (~160K samples) + Kalshi settlements.

*Technical indicators (from previous completed 15m candle):*
- `price_vs_ema` — trend position relative to EMA
- `macd_15m`, `norm_return`, `ema_slope`, `roc_5`, `vol_ratio`, `adx`
- `hourly_return`, `trend_direction`

*Higher timeframe context (in-progress candle dropped):*
- `rsi_1h` — hourly momentum (continuation signal)
- `macd_1h`, `rsi_4h`

*Strike-relative:*
- `distance_from_strike` — price vs strike in ATR units

*Kalshi-specific (lookback from recent settlements):*
- `prev_result` — last window's settlement (yes=1/no=0)
- `prev_3_yes_pct` — % of last 3 windows that settled YES
- `streak_length` — consecutive same-direction settlements (signed)
- `strike_delta` — strike movement vs previous window (ATR-normalized)
- `strike_trend_3` — average strike movement over last 3 windows

*Cross-asset confluence:*
- `alt_rsi_avg` — average RSI 15m of other 3 assets
- `alt_rsi_1h_avg` — average RSI 1h of other 3 assets
- `alt_momentum_align` — directional consensus across assets
- `prev_result_consensus` — cross-asset settlement agreement
- `alt_distance_avg` — average distance from strike of other assets

*Regime detection:*
- `return_4h`, `return_12h` — multi-hour return context
- `price_vs_sma_1h` — price vs 1h 20-SMA in ATR units
- `lower_lows_4h` — count of lower lows in last 3 4h candles
- `trend_strength` — price vs 4h 10-SMA in ATR units

*Time + derived:*
- `hour_sin`, `hour_cos` — cyclical hour encoding
- `rsi_alignment` — 1h/4h RSI agreement (+1 aligned, -1 diverging)
- `atr_percentile` — current ATR vs 20-period range (volatility regime)
- `bbw` — Bollinger Band Width (ranging vs trending regime)

*Interaction features (let XGBoost learn conditional relationships):*
- `pve_x_trend` — price_vs_ema × trend_strength
- `pve_x_return12h` — price_vs_ema × return_12h
- `slope_x_trend` — ema_slope × trend_strength
- `slope_x_return12h` — ema_slope × return_12h

### M10 Confirmation Model

Separate per-asset XGBoost model trained on minute-10 data. Decides hold or exit at minute 10.
- **39 features** — same 33 as M0 plus `rsi_15m` and 5 intra-window 5m features, plus interaction features:
  - `rsi_15m` — available at minute 10 (candle nearly complete)
  - `price_move_atr` — total price move during minutes 0-10
  - `candle1_range_atr`, `candle2_range_atr` — range of two 5m sub-periods
  - `momentum_shift` — direction change between 5m sub-periods
  - `volume_acceleration` — volume ratio between 5m sub-periods
- **Top features:** `distance_from_strike` (0.25) and `price_move_atr` (0.12-0.14) dominate exit decisions
- **Threshold:** 80/20 — only exits on very high-conviction disagreement
- **Anti-overfitting:** same XGBoost settings as M0
- **Training script:** `./venv/bin/python scripts/train_per_asset_m10.py`

### Execution Lifecycle — Minute-0 Entry + M10 Exit

| Time | State | Action |
|------|-------|--------|
| :00:03 | CONFIRMED | Signal + reprice up to 60c for ~8s. If ask > 60c → skip (no resting orders) |
| :02-:09 | MONITORING | Hold position, no new entries |
| :10+ | M10 CONFIRM | Per-asset M10 model: hold or exit. Only exits at 80%+ confidence. Sell at bid (floor 10c) |
| :15 | SETTLEMENT | Kalshi settles. P&L computed from our entry price + result |

**No resting buy orders.** If the ask isn't ≤ 60c at minute 0, we skip. The reprice loop tries 4 times over ~8 seconds, then cancels if still unfilled.

**M10 exit:** If per-asset M10 disagrees with bet side at 80%+ confidence → sell at current bid. If bid < 10c → place resting sell at 10c floor. Bet is immediately removed from `_pending_bets` to prevent double-counting at settlement.

**Shutdown:** All resting orders cancelled on Ctrl+C, SIGTERM, window boundary, and via atexit handler.

### Price Feeds (CRITICAL — must match training)

| Purpose | Source | Notes |
|---------|--------|-------|
| **Model input** | Coinbase+Bitstamp avg via CCXT | MUST match training. ~$5 from BRTI |
| **Dashboard PRICE column** | CF BRTI scrape | Display only, NOT for trading decisions. May be 30-60s stale |
| **Contract prices** | Kalshi WebSocket | Real-time YES/NO bid/ask. Used for fill detection |
| **Strike** | Kalshi REST API (floor_strike) | Filtered to future markets only (prevents stale window match) |

**WARNING:** CF Benchmarks SSR scrape is NOT real-time. Never use for model input — caused 25% WR in live when used for distance calculation.

### Betting Rules

- **Entry threshold:** 65% confidence (model prob >= 65 for YES, <= 35 for NO) — all assets
- **Max entry price:** 60c — if ask > 60c, skip entirely (NO resting orders)
- **M10 exit threshold:** 80/20 — only exit when per-asset M10 disagrees at 80%+ confidence
- **Position sizing:** flat 5% of balance per bet (CLI: `--maxsize=N`)
- **Max concurrent bets:** configurable (CLI: `--maxbets=N`, default 3)
- **No duplicate positions:** won't bet on an asset that already has an open position from any window
- **No Kelly sizing, no tiered sizing** — consistent risk every bet

### Mid-Cycle Recovery

On startup, the daemon queries Kalshi for existing positions and resting orders (`_recover_positions()`):
- Rebuilds `_pending_bets` from open positions
- Sets `bet_placed=True` in `_kalshi_pending_signals` to prevent duplicate bets
- If loaded at minute 10+, runs M10 immediately on recovered positions
- Subscribes WebSocket to recovered tickers

### Key Files

| File | Purpose |
|------|---------|
| `scripts/train_xgboost_per_asset.py` | Train per-asset XGBoost M0 models (33 features) |
| `scripts/train_per_asset_m10.py` | Train per-asset XGBoost M10 models (39 features) |
| `scripts/backtest_kalshi_labels.py` | Backtest with M0 entry + M10 exit simulation |
| `data/store/trade_debug.jsonl` | Order lifecycle debug log (market select, fills, settlements) |
| `data/store/feature_log.jsonl` | Model prediction feature audit log |
| `models/m0_{asset}.pkl` | Per-asset XGBoost M0 entry models |
| `models/m10_{asset}.pkl` | Per-asset XGBoost M10 exit models |

### Running

```bash
# Dry-run (simulated bets, real market data)
python dashboard_rich.py --dry-run

# Live (real money)
python dashboard_rich.py --live --maxbets=3 --maxsize=5

# Demo (Kalshi demo exchange — tests order plumbing only, NOT model)
python dashboard_rich.py --demo

# Backtest (60 days, threshold 65, M10 threshold 80)
./venv/bin/python scripts/backtest_kalshi_labels.py --days 60 --threshold 65 --m10-threshold 80 --exit-min 10 --exit-max 25
```

**WARNING:** Demo exchange has no real liquidity and prices don't track live. Only useful for testing order lifecycle, NOT for validating predictions.

### Series Tickers
`KXBTC15M`, `KXETH15M`, `KXSOL15M`, `KXXRP15M`

### Settlement
- Settles on **CF Benchmarks BRTI** — 60-second VWAP across Coinbase, Kraken, Bitstamp, Gemini, etc.
- Settlement check runs every 15 seconds with 15-second buffer after close_time
- Queries Kalshi `get_markets(status="settled")` for authoritative result
- P&L computed from our known entry price + settlement result (NOT from Kalshi settlements API costs, which mix buy/sell transactions)
- Unfilled orders removed without W/L count

## Accounts

| Account | Purpose | Auth |
|---------|---------|------|
| **Coinbase** | Spot crypto + market data via CCXT | CDP API key (`cdp_api_key.json`) |
| **Kalshi** | K15 prediction markets | RSA key (`KalshiPrimaryKey.txt`, key ID in settings.py) |
| **Kalshi Demo** | Order lifecycle testing | RSA key (`KalshiDemoKeys.txt`, first line = API key ID) |

## Lessons Learned

- **Train/live parity is everything.** If training uses 5m data at minute 5 but live decides at minute 1, WR drops from 75% to 54%. ALWAYS verify the exact same data path.
- **Price source matters.** Coinbase-only vs Coinbase+Bitstamp shifts distances enough to change predictions. The model MUST use the same price source as training.
- **Stale data kills WR.** CF Benchmarks SSR scrape was minutes stale — caused 25% WR in live. Use live API calls, never cached page data for trading decisions.
- **Fill timing is critical.** By minute 3-5, K15 contracts are already 70-90c. Minute 0 entry gets fills at ~50c.
- **Adverse selection from resting orders.** Placing limit orders below market ask = systematically filling on losers. If the ask drops to our limit, the market moved against our bet. NEVER place resting buy orders.
- **Settlements API costs are unreliable.** The `yes_total_cost_dollars`/`no_total_cost_dollars` fields mix buy and sell transaction costs, producing impossible entry prices (>100c). Always compute P&L from our known entry price.
- **M10 double-counting.** When M10 places a resting sell, the bet MUST be removed from `_pending_bets` immediately. Otherwise settlement processes it again as a LOSS.
- **get_order_status envelope.** Kalshi API returns `{"order": {...}}` — must unwrap the `order` key or fill detection silently fails.
- **M10 timing.** Use `>= 10` not `== 10` for the minute check. Eval runs on 60s intervals and can miss exact minute 10.
- **Entry window.** Only enter at minute 0-1. Re-evaluating at minutes 2-4 with stale indicators + drifting price generates bad signals.
- **Kalshi market transitions take seconds.** At window boundary, query for future markets only (close_time > now) to avoid betting on the just-settled window.
- **Demo exchange can't validate predictions.** Same strikes/settlements as production, but no real liquidity. Only tests order plumbing.
- **Losing days are NO-heavy choppy days.** The model's NO side underperforms on sideways markets near the strike. Higher thresholds (65) filter out weak signals and reduce losing days.
- **In-progress candle leaks future data.** CCXT returns the current (incomplete) candle as the last row in 1h/4h data. Must drop it before computing indicators, or the model sees partial future data that training never had.
- **Per-asset models outperform unified.** Each asset has different volatility regimes and RSI behavior. Per-asset models with cross-asset confluence capture both asset-specific patterns and market-wide signals.
- **rsi_15m is NOT available at minute 0.** The 15m candle hasn't closed yet, so rsi_15m is from the prior candle and stale. M0 excludes it (29 features); M10 includes it (candle nearly complete at minute 10).
- **Settlement P&L must use known entry price.** Kalshi settlements API cost fields mix buy/sell transactions. Always track entry_price at fill time and compute P&L as (payout - entry_price).
- **LogReg cannot learn conditional relationships.** `price_vs_ema × -0.98` ALWAYS pushes YES when price is below EMA, regardless of trend. XGBoost can learn "IF below EMA AND downtrend THEN NO." This was the root cause of persistent YES bias in bearish markets.
- **Synthetic training data must compute ALL features identically to live.** Defaulting confluence features to 50/0 in synthetic data trained the model to expect neutral confluence. In live, real values (63 vs 50) caused a -0.76 logit shift systematically biasing predictions.
- **XGBoost anti-overfitting is critical.** Use subsample=0.7, colsample=0.7, learning_rate=0.03, max_depth=4, min_child_weight=50, and early stopping on a validation set. Without these, XGBoost memorizes training data.
