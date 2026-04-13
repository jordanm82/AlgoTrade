# Chop Detection Logging (v1) — Design

**Date:** 2026-04-13
**Status:** Approved, ready for implementation plan
**Scope:** `cli/kalshi_daemon.py` only. No model changes. No gating logic.

## Motivation

Directional-chop days (market grinds sideways, RSI flips repeatedly signaling mean reversion that never completes) produce clusters of losses for the K15 predictor. The `rsi_1h` / `rsi_4h` features are the dominant M0 drivers and have a structural bias toward mean reversion. When the 15m or 1h range is compressed, those signals fire and lose.

Before we add any corrective logic, we need data: does a chop signal computed at bet-placement actually predict losses? And if so, which timeframe (15m vs 1h) and scope (per-asset vs market-wide) is most predictive? v1 answers those questions by logging chop metrics on every bet and carrying them through to settlement.

## Non-goals

- No threshold bump, no bet-blocking, no model retraining. v1 is telemetry only.
- No new feature added to the model — chop metrics live outside the feature vector.
- No new data fetches, no new indicators, no new candle requests. Reuses cached dataframes.

## Design

### Metrics

Eight fields computed at bet-placement time:

| Field | Timeframe | Scope | Source |
|---|---|---|---|
| `bbw_15m` | 15m | per-asset | `(bb_upper - bb_lower) / sma_20 * 100` on the bet's asset |
| `atr_pct_15m` | 15m | per-asset | rolling-20 min/max normalization of ATR, same formula as existing `atr_percentile` |
| `bbw_1h` | 1h | per-asset | same BBW formula on 1h df |
| `atr_pct_1h` | 1h | per-asset | same rolling-20 formula on 1h ATR |
| `bbw_15m_mkt` | 15m | market | arithmetic mean of `bbw_15m` across BTC, ETH, SOL, XRP |
| `atr_pct_15m_mkt` | 15m | market | arithmetic mean of `atr_pct_15m` |
| `bbw_1h_mkt` | 1h | market | arithmetic mean of `bbw_1h` |
| `atr_pct_1h_mkt` | 1h | market | arithmetic mean of `atr_pct_1h` |

All computations use the existing filtered dataframes (in-progress candle already dropped via `.iloc[:-1]` in the fetch paths). The rolling-20 lookback matches the existing `atr_percentile` computation at `cli/kalshi_daemon.py:326-331`.

### Helper

New method on `KalshiDaemon`:

```python
def _compute_chop_metrics(self, asset: str) -> dict:
    """Compute BBW/ATR chop metrics for `asset` plus market-wide averages.

    Returns a dict with the 8 fields above. Missing-data fields are None
    (never substituted with stale or default values — see CLAUDE.md fallback rule).
    Never raises; never blocks the caller.
    """
```

Implementation notes:
- Reads from `self._kalshi_cached_dataframes[symbol]` for 15m and `self._kalshi_cached_dataframes[f"{symbol}_1h"]` for 1h.
- Filters 15m df to `< current_window_start` for parity with model inference (training ignores in-progress window).
- Per-asset field is `None` if that asset's df is missing, empty, or has fewer than 20 rows.
- Market-wide field is the mean of whichever of the 4 per-asset values are non-None. If fewer than 2 are available, the market-wide field is `None` (one asset isn't a "market").
- BBW requires `bb_upper`, `bb_lower`, `sma_20` columns. 15m df has these from `add_indicators`; confirm 1h df does too when loaded (already called via `add_indicators(df_1h)` in daemon fetch path).

### Call sites

One call at bet placement, in `_kalshi_execute_bet` (cli/kalshi_daemon.py:2438). Compute the dict once, then:

1. **Merge into the `MARKET_SELECT` event** emitted before order placement. The 8 fields become top-level keys on that JSON line.
2. **Stash on the `_pending_bets` entry** under key `chop_metrics` so downstream events can re-emit without recomputation.

### Event schema changes

No new event types. Three existing events gain the 8 chop fields:

- **`MARKET_SELECT`** — fields computed fresh at emit time.
- **`SETTLED`** — fields copied from `bet["chop_metrics"]` so the settlement record contains both the decision-time chop values and the win/loss outcome. Enables a one-file join.
- **`M10_EXIT`** and **`M10_RESTING_SELL`** — same copy from `bet["chop_metrics"]`. Supports separate analysis of "did M10 exit correlate with chop at entry?".

`ORDER_PLACED`, `EARLY_EXIT`, `EXIT_REVIEW` do not get the fields. They can be joined via ticker if needed.

### Failure behavior

Follows the project's no-fallback rule:

- Missing dataframe → field is `None`, bet proceeds normally.
- Rolling window too short → field is `None`.
- Any exception inside `_compute_chop_metrics` is caught, logged via `print(colored(..., "yellow"))`, and returns a dict of all-`None` values. Never raises into the bet path.
- `bet["chop_metrics"]` is always a dict (possibly all-None), never missing — so downstream `.get("chop_metrics", {})` calls are safe even for recovered positions.

### Recovered positions

`_recover_positions()` on startup synthesizes `_pending_bets` entries for open positions. Those bets missed the entry-time compute. Set `chop_metrics` to an all-`None` dict on recovered entries. SETTLED/M10 logs for those bets will show `None` for chop fields — analysis filters them out.

### Dashboard

No dashboard changes in v1. The data is purely for offline analysis.

## Testing

- **Unit-test scope:** `_compute_chop_metrics` with (a) full data, (b) missing 1h df, (c) 15m df with <20 rows, (d) one asset missing from market, (e) all assets missing.
- **Integration:** one dry-run window with all 4 assets; verify `MARKET_SELECT` lines in `trade_debug.jsonl` contain the 8 fields with non-None values, and verify `SETTLED` lines 15 min later carry the same values.

## Analysis plan (post-v1, informational)

Once ~500 SETTLED bets are logged with chop metrics, run:

1. Bucket bets by each of the 8 chop fields (quartiles).
2. Compute WR per bucket. If any field shows a monotonic WR drop in high-chop buckets, it's a candidate gate.
3. Pick the field with the cleanest signal and the threshold that recovers WR by ≥3 pts with acceptable bet-count loss.
4. v2 spec: implement a threshold bump keyed on that field. v1 remains unchanged.

## Out of scope (for v1)

- Switching-frequency signal (last N settlement YES/NO alternation count).
- VPIN (volume-synchronized probability of informed trading).
- Chop-score composite formula.
- Per-asset threshold bumps.
- Live dashboard chop indicator.

These are candidates for v2+ after v1 logs validate which raw signal is worth gating on.
