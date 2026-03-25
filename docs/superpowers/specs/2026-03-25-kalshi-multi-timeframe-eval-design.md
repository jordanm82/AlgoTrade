# Kalshi Multi-Timeframe Evaluation System

**Date:** 2026-03-25
**Goal:** Replace the single 15-minute scoring cycle with a progressive 5-minute evaluation system that accumulates confirmation within each 15-minute Kalshi contract window, enabling more high-confidence bets.

## Problem

The current system scores Kalshi predictions once per 15-minute candle close. This has three issues:

1. **Stale data** — leading indicators (order book, trade flow) are fetched once and frozen for 15 minutes. Opportunities that develop mid-window are missed entirely.
2. **All-or-nothing** — a signal at conf=26 (below threshold 30) at candle close might have been conf=40 five minutes later. We never re-check.
3. **No confirmation** — we bet on the first candle close score without waiting to see if the move actually develops. The first few minutes of a 15m window provide zero confirmation.

## Design

### 1. Evaluation Cadence

Replace the single 15m Kalshi cycle with a dedicated 5-minute evaluation cycle.

**New constants:**
```python
KALSHI_CUTOFF_MINUTES = 13     # no new bets after minute 13 of a 15m window
KALSHI_LASTLOOK_MINUTE = 12    # elevated threshold window
KALSHI_THRESHOLD_BOOST = 10    # added to per-asset threshold for last-look entries
```

**Timer alignment:** Instead of an elapsed-time interval, `_kalshi_eval()` fires on every 60-second tick but only runs when `datetime.now().minute % 5 == 1` (i.e., at minutes :01, :06, :11 of each 15m window). The 1-minute offset ensures the preceding 5m candle is fully closed before we score it.

**LAST_LOOK trigger:** Since `12 % 5 == 2`, the standard timer condition won't fire at minute 12. LAST_LOOK gets its own additional check: `current_minute % 15 == 12`. The full trigger condition is:
```python
should_eval = (current_minute % 5 == 1 and now - self._last_kalshi_eval >= 240) \
           or (current_minute % 15 == 12 and now - self._last_kalshi_eval >= 50)
```
The `>= 50` guard on LAST_LOOK prevents double-firing within the same minute.

**Data sources per evaluation:**
- 5m candles (limit=200) with full `add_indicators()` — RSI, StochRSI, ROC, MACD, BB, ATR all computed on 5m data. **The last candle is dropped if its timestamp matches the current 5-minute boundary** (partial candle protection).
- 1h candles (limit=50) with `add_indicators()` — MTF confirmation (unchanged)
- Order book + trade flow — fresh leading indicators every 5 minutes
- 1m candles (limit=5) — only fetched at minute 12 for last-look momentum check

**API rate impact:** 5 assets x 4 calls (5m OHLCV, 1h OHLCV, orderbook, trade flow) = 20 API calls every 5 minutes + 5 additional 1m OHLCV calls at minute 12. BinanceUS rate limits allow 1200 requests/minute; we use ~4/minute on average. Well within limits.

**Removed:** The old `_kalshi_cycle()` called from `signal_cycle()` every 15 minutes. All Kalshi evaluation moves to the new `_kalshi_eval()` on the tick-aligned timer.

### 2. Signal Lifecycle

A Kalshi signal has a lifecycle within each 15-minute contract window. The window boundaries are fixed clock times (:00, :15, :30, :45). The minute within the window is computed as `current_minute % 15`.

**States:**

| State | Minute | Data | Threshold | Action |
|-------|--------|------|-----------|--------|
| SETUP | 0-4 | 5m candles + OB + flow | N/A | Score and store direction + base confidence. NO BETTING. |
| CONFIRMED | 5-9 | 5m candles + OB + flow | Normal per-asset | If confidence >= threshold AND direction matches setup → place bet |
| DOUBLE_CONFIRMED | 10-11 | 5m candles + OB + flow | Normal per-asset | If confidence >= threshold AND direction held through both 5m candles → place bet |
| LAST_LOOK | 12 | Cached 5m lagging scores + 1m momentum | Per-asset + 10 | If 5m confidence >= elevated threshold AND 2 of last 3 1m candles confirm direction → place bet |
| EXPIRED | 13-14 | None | N/A | No new bets. Too close to settlement. |

**Lifecycle rules:**
- Once a bet is placed on an asset in a window, no additional bets on that asset until the next window
- If direction flips between evaluations (e.g. SETUP=UP, CONFIRMED=DOWN), the signal dies — no resurrecting
- If `score()` returns `None` during CONFIRMED/DOUBLE_CONFIRMED, the pending signal **survives** — treated as a skip for this evaluation, not a kill. Only an explicit direction flip kills it.
- Each evaluation refreshes leading indicators regardless of state
- SETUP uses the most recent fully closed 5m candle (eval fires at minute 1/6/11, so the preceding candle is guaranteed closed)
- **Window boundary detection:** Each pending signal stores `window_start` (current time rounded down to nearest 15-minute boundary). On every evaluation, compute the current window start. If it doesn't match the stored `window_start`, clear the signal. This is robust against timing drift — no reliance on detecting "minute 0" exactly.

**State stored per asset:**
```python
{
    "direction": "UP",         # direction from SETUP
    "base_conf": 26,           # confidence at SETUP
    "last_5m_conf": 35,        # confidence from most recent 5m evaluation (for LAST_LOOK)
    "setup_time": timestamp,   # when SETUP was recorded
    "confirmed": False,        # has a 5m confirmation occurred
    "bet_placed": False,       # has a bet been placed this window
    "window_start": timestamp, # start of the 15m window (rounded to :00/:15/:30/:45)
}
```

### 3. Predictor Changes

**No changes to `score()`.** It already accepts any DataFrame — 5m candles work identically to 15m candles. Indicators naturally reflect the faster timeframe.

**New method: `check_1m_momentum()`**

```python
def check_1m_momentum(self, df_1m: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
    """Check if recent 1m candles confirm the direction.

    Returns True if at least 2 of the last `lookback` candles moved in
    the expected direction (close > open for UP, close < open for DOWN).
    Simple majority rule avoids single noisy 1m candle killing the signal.
    """
```

This is a lightweight directional check, not a full scoring run. No indicators computed on 1m data.

**Per-asset thresholds unchanged:** BTC/XRP: 30, ETH/SOL/BNB: 35.
**Elevated thresholds for last-look:** BTC/XRP: 40, ETH/SOL/BNB: 45.

These will be re-validated via 5m backtest.

### 4. Daemon Architecture

**Run loop — wall-clock aligned Kalshi eval:**

The Kalshi eval fires on every 60-second tick when `minute % 5 == 1` (offset by 1 minute to ensure the preceding 5m candle is fully closed). This guarantees evaluations at minutes 1, 6, 11 of each 15m window regardless of daemon start time.

```python
while self._running:
    self.tick()

    now = time.time()
    current_minute = datetime.now(timezone.utc).minute

    # Existing 15m signal cycle (spot trading)
    if now - last_signal_time >= SIGNAL_INTERVAL:
        self.signal_cycle()    # spot only, no Kalshi
        last_signal_time = now

    # Wall-clock aligned Kalshi eval at :01, :06, :11 + LAST_LOOK at :12, :27, :42, :57
    should_eval = (current_minute % 5 == 1 and now - self._last_kalshi_eval >= 240) \
               or (current_minute % 15 == 12 and now - self._last_kalshi_eval >= 50)
    if should_eval:
        self._kalshi_eval()
        self._last_kalshi_eval = now
```

The `>= 240` guard prevents double-firing on the 5-min cycle. The `>= 50` guard prevents double-firing on the LAST_LOOK minute.

**New state in `__init__`:**
```python
self._kalshi_pending_signals = {}   # {asset: {direction, base_conf, ...}}
self._last_kalshi_eval = 0          # timestamp
self._kalshi_5m_dataframes = {}     # {symbol: DataFrame} — overwritten on each SETUP/CONFIRMED/DOUBLE_CONFIRMED eval, read without re-fetch at LAST_LOOK, cleared at window boundary
```

**`_kalshi_eval()` method flow:**

For each asset in KALSHI_PAIRS:
1. Compute `minute_in_window = datetime.now(timezone.utc).minute % 15`
2. Compute `current_window_start` (round down to nearest :00/:15/:30/:45). If pending signal's `window_start` doesn't match → clear it (new window).
3. Fetch 5m candles + indicators (drop last candle if partial), cache in `_kalshi_5m_dataframes`
4. Fetch 1h candles + indicators
5. Fetch order book + trade flow
6. Based on `minute_in_window`:
   - **0-4 (SETUP):** Score with 5m data. Store direction + confidence in `_kalshi_pending_signals` with `window_start`. Only write if no pending signal exists for the current window (prevents overwriting on duplicate ticks). Log but don't bet.
   - **5-9 (CONFIRMED):** Score with fresh 5m data. Update `last_5m_conf`. If `score()` returns None → skip (signal survives). If direction flipped → kill signal. If confidence >= threshold AND direction matches → mark as actionable.
   - **10-11 (DOUBLE_CONFIRMED):** Same as CONFIRMED. The double confirmation is implicit — if the signal survived two 5m candles, it's stronger.
   - **12 (LAST_LOOK):** Don't re-score lagging. Fetch 1m candles (limit=5). Call `check_1m_momentum()`. If momentum confirms AND `last_5m_conf` >= threshold + KALSHI_THRESHOLD_BOOST → mark as actionable.
   - **13-14 (EXPIRED):** Skip.
7. Collect all actionable signals, sort by confidence descending, execute top N up to concurrency limit via `_kalshi_execute_bet()`.
8. Store predictions for dashboard display (include lifecycle state for observability).

**`_kalshi_execute_bet()` — extracted execution helper:**

The ~180 lines of bet execution logic (balance check, market discovery, orderbook pricing, order placement, fill verification) currently inline in `_kalshi_cycle()` is extracted into a standalone `_kalshi_execute_bet(symbol, series_ticker, signal, market_data)` method. This keeps `_kalshi_eval()` focused on lifecycle management and prevents it from becoming a 300+ line method.

**`signal_cycle()` change:** Remove the `_kalshi_cycle()` call. Kalshi evaluation is now fully handled by `_kalshi_eval()`.

**Dashboard observability:** The predictions stored for dashboard display include the lifecycle state (SETUP, CONFIRMED, DOUBLE_CONFIRMED, LAST_LOOK, EXPIRED) so the operator can see where each asset is in its evaluation window.

### 5. Backtest Methodology

**Data requirements:**
- 5m candles for all assets (limit ~26,000 per asset for 90 days, fetched in batches)
- 1h candles (existing)
- 1m candles (limit ~130,000 per asset for 90 days — large but feasible, fetched in batches)

**Simulation logic:**

For each 15-minute window in the backtest period:
1. Identify the three 5m candles within the window (at minutes 0, 5, 10)
2. **Minute 0**: Score using 5m data up to that point. Record as SETUP.
3. **Minute 5**: Score using 5m data including the first closed 5m candle. If confidence >= threshold AND direction matches SETUP → record as a bet entry at minute 5.
4. **Minute 10**: If no bet yet, score again. If confidence >= threshold AND direction held → record as bet entry at minute 10.
5. **Minute 12**: If no bet yet, check 1m momentum (2 of 3 candles confirm). If yes AND 5m confidence >= threshold + 10 → record as bet entry at minute 12.
6. **Result**: Did the 15m window close higher or lower than it opened? Compare to the bet direction.

Only the FIRST qualifying entry per window per asset counts — no double-betting.

**Metrics:** Same as current backtest — WR, bets, ROI, PF, max drawdown, per-asset WR.

**Comparison:** Run side-by-side with the existing 15m backtest on the same 90-day period. The 5m system should show more bets (catching mid-window entries) with comparable or better WR.

**1m data note:** For the last-look simulation, we need 1m candles. This is the largest data fetch (~130K candles per asset over 90 days). If fetching is too slow, the last-look can be excluded from the initial backtest and tested separately on a shorter period (30 days).

**Leading indicator gap:** The backtest cannot simulate order book and trade flow data (not available historically). The scorer runs with `market_data=None`, using `max_possible = _MAX_LAGGING + _MAX_MTF = 135` instead of the live system's `_MAX_RAW = 200`. This means backtest confidence scores are systematically higher than live scores for the same raw signal strength. This is a known limitation shared with the current 15m backtest and does not affect relative comparisons between the 15m and 5m systems.

## Files to Modify

| File | Change |
|------|--------|
| `strategy/strategies/kalshi_predictor.py` | Add `check_1m_momentum()` method |
| `cli/live_daemon.py` | Add wall-clock aligned Kalshi eval timer. Add `_kalshi_eval()` with lifecycle state machine. Add `_kalshi_pending_signals` state. Extract `_kalshi_execute_bet()` from existing execution logic. Remove `_kalshi_cycle()` call from `signal_cycle()`. Update run loop. |
| `backtest_kalshi.py` | Rewrite to simulate 5m evaluation lifecycle with 5m + 1m candle data, progressive confirmation gates |
| `tests/test_kalshi.py` | Add tests for `check_1m_momentum()`, lifecycle state transitions |

## Files NOT Modified

- `exchange/kalshi.py` — API client unchanged
- `data/indicators.py` — already works on any timeframe
- `config/production.py` — per-asset thresholds unchanged (re-tuned via backtest, not code change)
- `config/pair_config.py` — not involved in Kalshi

## Post-Implementation

- Update CLAUDE.md with new evaluation cadence and signal lifecycle
- Re-run per-asset threshold sweep on 5m data to validate/re-tune thresholds

## Success Criteria

- WR >= 60% on 3-month 5m backtest (comparable to 15m baseline of 62.8%)
- More total bets than the 15m system (catching mid-window entries)
- Profit factor >= 3.0
- No regressions in per-asset performance
- System correctly avoids betting before minute 5 (no unconfirmed bets)
- Last-look entries at minute 12 have higher WR than primary entries (elevated threshold ensures quality)
