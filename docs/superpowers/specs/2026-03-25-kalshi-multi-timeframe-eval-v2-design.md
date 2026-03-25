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

**Key change from v1:** SETUP runs the full 15m predictor (not the 5m predictor). CONFIRMED/DOUBLE_CONFIRMED compute the 5m booster (not a full re-score). The booster is lightweight — it reads the cached 5m candle data and computes 4 boost checks + 2 penalty checks. No `add_indicators()` call needed on 5m data, just raw OHLCV + simple computations.

**Exception:** The booster DOES need 5m MACD histogram and 5m RSI for the crossover/divergence checks. These require `add_indicators()` on 5m data. However, this is computed once per 5m eval (not per-candle like v1 attempted) and is only used for the booster, not for full scoring.

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

Booster logic:
1. Get last closed 5m candle
2. Check candle direction (+3 if aligned)
3. Check volume vs SMA (+3 if > 1.5x)
4. Check ATR distance from window open (+3 if > 0.5x ATR in predicted direction)
5. Check MACD histogram crossover (+3 if aligned, -5 if against)
6. Check RSI divergence (-5 if diverging against)
7. Return sum (clamped to -10..+12)

### 5. Daemon Changes

**`_kalshi_eval()` changes:**

- **SETUP:** Instead of scoring on 5m data, run the full 15m predictor on 15m candles (same as old `_kalshi_cycle()` did). Store base confidence + direction. Also fetch and cache 5m candles with indicators for the booster.
- **CONFIRMED/DOUBLE_CONFIRMED:** Don't re-score. Fetch latest 5m candles, run `compute_5m_booster()`, add to base confidence. If `base + booster >= threshold` → actionable.
- **LAST_LOOK:** Unchanged from v1 — 1m momentum check with elevated threshold.

**New state field:**
```python
{
    "direction": "UP",
    "base_conf": 26,           # from 15m predictor
    "boosted_conf": 35,        # base + latest booster
    "last_booster": 9,         # last computed booster value
    "window_open_price": 71000, # price at window start for ATR distance
    ...existing fields...
}
```

### 6. Backtest Changes

The backtest needs to simulate the booster, not full 5m re-scoring.

For each 15-minute window:
1. Score at window boundary using 15m data up to that point (the old way — this works at 62.8% WR)
2. At minute 5: compute 5m booster from first 5m candle. If `base + booster >= threshold` → bet at minute 5.
3. At minute 10: compute 5m booster from second 5m candle. If `base + booster >= threshold` → bet at minute 10.
4. Compare to 15m baseline (single-shot scoring).

The backtest for the booster is straightforward because we have both 15m and 5m historical data. No order book simulation needed.

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
