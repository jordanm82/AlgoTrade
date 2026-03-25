# Kalshi Multi-Timeframe Evaluation System v2

**Date:** 2026-03-25
**Supersedes:** `2026-03-25-kalshi-multi-timeframe-eval-design.md` (v1 used full 5m re-scoring which was validated as non-viable — predictor produces ~25% WR on 5m candles vs 62.8% on 15m)

**Goal:** Keep proven 15m scoring as directional base, add a 5m confirmation booster that promotes near-threshold signals when real-time price action confirms the predicted direction.

## Problem

The system scores Kalshi predictions once per 15-minute candle close, producing a base confidence score. Three issues:

1. **Stale leading indicators** — order book and trade flow are fetched once and frozen for 15 minutes.
2. **No mid-window promotion** — a signal at conf=26 (4 points below threshold 30) never gets re-evaluated even if the move develops strongly in the predicted direction.
3. **No confirmation requirement** — the system bets immediately on candle close without verifying the move develops.

## Key Finding (from v1)

**The predictor does NOT work on 5m candles.** RSI(14), MACD(12/26), BB(20), StochRSI(14) are all calibrated for 15m timeframes. On 5m data, these indicators produce ~25% WR (worse than random). The 15m scoring (62.8% WR) must remain the directional signal source.

However, 5m price action (candle direction, volume, momentum, ATR distance) IS useful as a **confirmation layer** — it tells us whether the 15m prediction is developing in real-time.

## Design

### 1. Architecture: 15m Base + 5m Booster

```
15m candle close → full predictor score → base confidence (e.g. 26)
    ↓
Every 5 min → check 5m confirmation factors → booster (+12 to -10)
    ↓
boosted_confidence = base + booster → if >= threshold → bet
```

The 15m score provides the **directional edge**. The 5m booster provides **timing and confirmation**.

### 2. 5m Confirmation Booster

Computed from the most recent closed 5m candle + recent 5m history. Uses raw price action, not indicator re-scoring.

**Boost factors (additive, +3 each):**

| Factor | Points | Logic |
|--------|--------|-------|
| **5m candle direction** | +3 | Last closed 5m candle moved in predicted direction (close > open for UP, close < open for DOWN) |
| **5m volume surge** | +3 | Volume of last closed 5m candle > 1.5x its 20-period SMA |
| **ATR distance** | +3 | Price has moved > 0.5x ATR(14) in predicted direction since 15m window opened |
| **5m MACD crossover aligned** | +3 | 5m MACD histogram crossed from negative to positive (for UP) or positive to negative (for DOWN) in the last 2 candles |

**Penalty factors (stronger than boosts):**

| Factor | Points | Logic |
|--------|--------|-------|
| **5m MACD crossover against** | -5 | 5m MACD histogram just crossed AGAINST predicted direction — momentum reversing |
| **5m RSI divergence** | -5 | Price made new high/low on 5m but RSI didn't follow (bearish/bullish divergence against predicted direction) |

**Range: -10 to +12.** Max boost = +12, max penalty = -10.

**Boundary behavior:** A 15m signal at conf=18 with perfect +12 boost = 30 (barely clears BTC threshold). Signals below ~20 base confidence shouldn't bet even with perfect confirmation. A signal at conf=35 with -10 penalty = 25 (drops below threshold — bad warning signs killed a decent signal).

### 3. Signal Lifecycle (Revised)

The lifecycle from v1 stays, but the scoring model changes:

| State | Minute | What Happens |
|-------|--------|-------------|
| **SETUP** | 0 (on 15m candle close) | Full 15m predictor score. Sets direction + base confidence. Refreshes leading indicators. No betting. |
| **CONFIRMED** | 5 | Compute 5m booster from first closed 5m candle. `boosted_conf = base + booster`. If boosted >= threshold AND direction matches setup → bet. Also re-fetch leading indicators. |
| **DOUBLE_CONFIRMED** | 10 | Compute 5m booster from second closed 5m candle. Same logic. Highest conviction if both 5m candles confirm. |
| **LAST_LOOK** | 12 | 1m momentum check (existing `check_1m_momentum()`). Elevated threshold (per-asset + 10). Uses last computed boosted confidence. |
| **EXPIRED** | 13+ | No new bets. |

**Key change from v1:** SETUP runs the full 15m predictor (not the 5m predictor). CONFIRMED/DOUBLE_CONFIRMED compute the 5m booster (not a full re-score). The booster requires `add_indicators()` on 5m data for the MACD histogram and RSI checks, but this is computed once per 5m eval and is only used for the booster's 6 factor checks — not for full confidence scoring.

### 4. Booster Implementation

New method on `KalshiPredictor`:

```python
def compute_5m_booster(self, df_5m: pd.DataFrame, direction: str,
                        window_open_price: float) -> int:
    """Compute 5m confirmation booster for a 15m base signal.

    Returns a score from -10 to +12 that adds to the base 15m confidence.
    Positive = 5m price action confirms the predicted direction.
    Negative = 5m price action contradicts the predicted direction.

    Args:
        df_5m: 5m OHLCV with indicators (at least RSI, MACD hist)
        direction: "UP" or "DOWN" from the 15m base signal
        window_open_price: Price at the start of the 15m window
    """
```

**Requires:** `df_5m` must have `add_indicators()` applied (needs `rsi`, `macd_hist`, `atr`, `vol_sma_20` columns).

Booster logic (uses last 2 closed 5m candles):
1. Get last 2 closed 5m candles (`last` = iloc[-1], `prev` = iloc[-2])
2. **Candle direction** (+3): `last` candle close > open for UP, close < open for DOWN
3. **Volume surge** (+3): `last` candle volume > 1.5x `vol_sma_20`
4. **ATR distance** (+3): `abs(last.close - window_open_price) > 0.5 * last.atr` AND movement is in predicted direction
5. **MACD crossover aligned** (+3): `prev.macd_hist` and `last.macd_hist` have different signs, AND `last.macd_hist` is in predicted direction (positive for UP, negative for DOWN)
6. **MACD crossover against** (-5): same crossover detection but `last.macd_hist` is AGAINST predicted direction
7. **RSI divergence against** (-5): for UP prediction: `last.close > prev.close` BUT `last.rsi < prev.rsi` (price up, momentum fading). For DOWN prediction: `last.close < prev.close` BUT `last.rsi > prev.rsi` (price down, momentum fading). Simple 2-candle divergence — no swing point detection needed.
8. Return sum, clamped to range [-10, +12]

Note: MACD checks 5 and 6 are mutually exclusive (a crossover is either aligned or against, not both).

### 5. Daemon Changes

**`_kalshi_eval()` changes:**

- **SETUP:** Fetch 15m candles (limit=200) + `add_indicators()`. Run the full 15m predictor with `score(df_15m, market_data, df_1h)`. Store base confidence + direction. Also fetch 5m candles (limit=200) + `add_indicators()` and cache in `_kalshi_5m_dataframes` for the booster. Capture `window_open_price` from the 15m scoring candle's close price (last 15m candle close = this window's open). Re-fetch leading indicators (order book + trade flow) for the 15m scorer.
- **CONFIRMED/DOUBLE_CONFIRMED:** Don't re-score on 15m. Fetch latest 5m candles + `add_indicators()`, update cache. Run `compute_5m_booster(df_5m, direction, window_open_price)`. Compute `boosted_conf = base_conf + booster`. If boosted >= threshold → actionable. No need to re-fetch leading indicators (the booster uses only 5m OHLCV + indicators).
- **LAST_LOOK:** Unchanged from v1 — 1m momentum check with elevated threshold. Uses last `boosted_conf`.

**15m data caching:** 15m candles are only fetched at SETUP (once per 15m window). The result is used for `score()` and then the base confidence is cached in the pending signal state. No 15m DataFrame cache needed — just the score.

**New state fields:**
```python
{
    "direction": "UP",
    "base_conf": 26,            # from 15m predictor at SETUP
    "boosted_conf": 35,         # base + latest booster
    "last_booster": 9,          # last computed booster value
    "last_5m_conf": 35,         # alias for boosted_conf (used by LAST_LOOK)
    "window_open_price": 71000, # 15m candle close price = window open price
    "setup_time": timestamp,
    "confirmed": False,
    "bet_placed": False,
    "window_start": timestamp,
}
```

### 6. Backtest Changes

The backtest needs to simulate the booster, not full 5m re-scoring.

**Function signature change:** `simulate_lifecycle(asset, df_15m, df_5m, df_1h, predictor, threshold)` — takes BOTH 15m and 5m DataFrames.

**For each 15-minute window:**
1. **SETUP:** Score on 15m data using `predictor.score(df_15m[:boundary], df_1h=...)`. Record base confidence + direction. Capture `window_open_price` from 15m candle close.
2. **At minute 5:** Find the first 5m candle that closed within this window. Run `predictor.compute_5m_booster(df_5m[:candle], direction, window_open_price)`. If `base + booster >= threshold` → record bet at minute 5.
3. **At minute 10:** If no bet yet, find the second 5m candle. Compute booster again. If `base + booster >= threshold` → record bet at minute 10.
4. **Actual direction:** Compare 15m window open price to close price.
5. Compare total results to 15m baseline (single-shot scoring at candle close).

The backtest should report:
- How many signals were promoted (started below threshold, booster pushed above)
- How many signals were demoted (started above threshold, penalty pushed below)
- Entry minute distribution (min 5 vs min 10)
- Side-by-side WR: 15m baseline vs 15m+booster

**Note:** The number of base signals (SETUP) will match the 15m baseline exactly, since both use the same 15m predictor. The difference is that the booster allows near-threshold signals to enter and can also prevent some above-threshold signals from entering (if penalties fire).

## Files to Modify

| File | Change |
|------|--------|
| `strategy/strategies/kalshi_predictor.py` | Add `compute_5m_booster()` method |
| `cli/live_daemon.py` | Update `_kalshi_eval()` SETUP to use 15m scoring. Update CONFIRMED/DOUBLE_CONFIRMED to use booster instead of full re-score. Add `window_open_price` and `boosted_conf` to pending signal state. |
| `backtest_kalshi.py` | Rewrite `simulate_lifecycle()` to use 15m base + 5m booster model |
| `tests/test_kalshi.py` | Add tests for `compute_5m_booster()` |

## Files NOT Modified

- `exchange/kalshi.py` — API client unchanged
- `data/indicators.py` — already works on any timeframe
- `config/production.py` — thresholds unchanged
- `dashboard.py` — lifecycle state display already implemented

## Success Criteria

- 5m booster backtest shows **more bets** than 15m baseline (catching near-threshold signals)
- Win rate stays **>= 60%** (booster promotes good signals, penalties kill bad ones)
- Profit factor >= 3.0
- Booster values are tunable via backtest sweep (+3/+3/+3/+3/-5/-5 can be adjusted)
- MACD crossover penalty demonstrably prevents some losing bets
