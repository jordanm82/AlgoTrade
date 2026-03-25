# Kalshi V2 Continuation Predictor

**Date:** 2026-03-25
**Goal:** Build a trend/continuation-based Kalshi 15m predictor as a separate scoring module (v2) alongside the existing mean-reversion predictor (v1). Continuation is the primary signal; mean-reversion indicators become overextension penalties.

## Problem

The v1 predictor uses mean-reversion indicators (RSI, BB, StochRSI) as primary signals. It achieves 63% WR by catching reversals from extremes. But on any given 15-minute candle, the most common outcome is **continuation of the current direction** — especially with momentum behind it. V1 is completely blind to these setups.

Kalshi 15m contracts ask "will price be above/below X in 15 minutes?" — fundamentally a continuation question. A predictor designed around continuation should capture a larger opportunity set.

## Architecture

**Two-layer scoring model:**

```
Layer 1: Trend Continuation Score (0-100)
    9 components measuring direction, strength, persistence, volume
    ↓
Layer 2: Mean-Reversion Penalty (-45 to 0)
    4 penalty components checking for overextension
    ↓
Raw Score = Trend Score + Penalties + Leading Indicators + MTF
    ↓
Confidence = normalized 0-100
```

**Swappable predictor:** V2 is a new file (`kalshi_predictor_v2.py`) with the same interface as v1. The daemon selects which predictor to use via a `--predictor v1|v2` flag. All infrastructure (lifecycle, execution, concurrency, dashboard) stays identical.

## Design

### 1. Trend Continuation Scoring (Primary Layer, max 100 points)

**Trend Direction (0-25 points):**

| Component | Max | Logic |
|-----------|-----|-------|
| Price vs EMA-12 | 10 | Price > EMA-12: +10 UP. Price < EMA-12: +10 DOWN. Within 0.1%: 0. |
| Price vs SMA-20 | 10 | Price > SMA-20: +10 UP. Price < SMA-20: +10 DOWN. Within 0.1%: 0. |
| EMA-12 vs SMA-20 | 5 | EMA above SMA: +5 UP. EMA below SMA: +5 DOWN. |

**Trend Strength (0-30 points):**

| Component | Max | Logic |
|-----------|-----|-------|
| ADX | 15 | ADX > 40: +15. ADX > 30: +10. ADX > 20: +5. ADX < 20: 0. |
| MACD histogram | 15 | Histogram growing in trend direction for 2+ candles: +15. Growing 1 candle: +8. Positive but shrinking: +3. |

**Trend Persistence (0-30 points):**

| Component | Max | Logic |
|-----------|-----|-------|
| Consecutive candles | 10 | 4+ candles same direction (close >= open for UP, close <= open for DOWN — dojis count as continuation): +10. 3: +7. 2: +3. |
| Higher highs/higher lows | 10 | Compare last 3 candles' highs and lows to their predecessors. 3 consecutive higher-highs AND higher-lows: +10. 2 consecutive: +5. For DOWN: lower-lows AND lower-highs. |
| ROC-5 aligned | 10 | ROC > 1.0% in trend direction: +10. ROC > 0.3%: +5. |

**Volume Confirmation (0-15 points):**

| Component | Max | Logic |
|-----------|-----|-------|
| Volume trend | 8 | Volume increasing for 2+ candles (each > previous): +8. Flat (within 10% of previous): +3. Decreasing: 0. |
| Volume vs average | 7 | Volume > 2x vol_sma_20: +7. > 1.5x: +4. > 1x: +2. |

Direction determined by which side (UP vs DOWN) accumulates more points. **If UP and DOWN scores are equal, return None** (no signal) — same behavior as v1.

### 2. Mean-Reversion Penalty Layer (max -45 points)

Penalties only fire when a trend direction is established. They reduce confidence when the trend is overextended.

| Component | Penalty | Logic |
|-----------|---------|-------|
| RSI extreme | -15 | RSI > 80 (UP trend) or RSI < 20 (DOWN trend): -15. RSI > 75 / < 25: -8. |
| StochRSI extreme | -10 | StochRSI K > 95 (UP) or K < 5 (DOWN): -10. K > 90 / < 10: -5. |
| BB overextension | -10 | Price > 1% beyond BB upper (UP) or BB lower (DOWN): -10. At the band: -5. |
| RSI divergence | -10 | Price making higher high but RSI lower (UP), or lower low but RSI higher (DOWN), over last 3 candles: -10. Requires at least 4 candles of data (3 comparisons). |

Penalties are applied to the dominant direction's score, reducing it.

### 3. Leading Indicators + MTF (unchanged from v1)

Same 5 leading components (Order Book 0-20, Trade Flow 0-20, Large Trade Bias 0-10, Spread 0-5, Cross-Asset 0-10) and 1h Trend Alignment (-15 to +15).

These are signal-type agnostic — order book buy pressure confirms a continuation signal just as well as a reversal.

### 4. Signal Quality Filters (unchanged from v1)

Same 3 hard-gate filters: directional conflict, volatility regime (>90th percentile ATR), margin of victory (winner >= 1.5x loser).

### 5. Normalization and Confidence

**Max raw scores:**
- Trend layer: 100
- Penalties: -45
- Leading: 65
- MTF: 15
- **Max possible raw: 180** (100 trend + 0 penalty + 65 leading + 15 MTF)

**Normalization (adaptive, same pattern as v1):**
```python
_MAX_TREND = 100
_MAX_LEADING = 65
_MAX_MTF = 15

if has_leading:
    max_possible = _MAX_TREND + _MAX_LEADING + _MAX_MTF  # 180
elif df_1h is not None:
    max_possible = _MAX_TREND + _MAX_MTF  # 115
else:
    max_possible = _MAX_TREND  # 100

confidence = min(100, max(0, int(raw_score * 100 / max_possible)))
```
Negative raw scores (from heavy penalties) are clamped to 0 confidence.

**Per-asset thresholds:** Start with v1 values (BTC/XRP: 30, ETH/SOL/BNB: 35). Re-tune via backtest since confidence distributions will differ.

All point values are tunable via backtest sweep.

### 6. Integration — Swappable Predictor

**New file:** `strategy/strategies/kalshi_predictor_v2.py`

Same interface as v1:
```python
class KalshiPredictorV2:
    def score(self, df, market_data=None, df_1h=None) -> KalshiSignal | None
    def check_1m_momentum(self, df_1m, direction, lookback=3) -> bool
    def compute_5m_booster(self, df_5m, direction, window_open_price) -> int
    def _apply_filters(self, ...) -> bool
```

Returns the same `KalshiSignal` dataclass. All downstream code (daemon, dashboard, execution) works without changes.

**`check_1m_momentum()`:** Reuse v1 implementation unchanged — the 2-of-3 majority rule works for continuation (recent 1m candles moving in predicted direction).

**`compute_5m_booster()`:** Reuse v1 implementation unchanged. The v1 booster was inversely correlated with WR on mean-reversion signals, but for continuation signals the booster's checks (candle direction, volume, MACD crossover) should be positively correlated — they confirm the trend is continuing. The daemon currently does not call `compute_5m_booster()` (simplified eval uses 15m re-scoring with fresh leading indicators), but the method exists for interface compliance and potential future use.

**`_apply_filters()`:** Same filter logic — directional conflict, volatility regime, margin of victory. V2 will track `lagging_up`/`lagging_down` (from trend components) and `leading_up`/`leading_down` separately, same as v1.

**Component diagnostics:** Include penalty details in the `components` dict returned by `score()` (e.g. `components["penalty_rsi"] = {"score": -15, "value": 82}`) for dashboard visibility and backtest analysis.

**Daemon predictor selection:**

In `LiveDaemon.__init__()`:
```python
if self.kalshi_predictor_version == "v2":
    from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
    self.kalshi_predictor = KalshiPredictorV2()
else:
    from strategy.strategies.kalshi_predictor import KalshiPredictor
    self.kalshi_predictor = KalshiPredictor()
```

**CLI flag:** `--predictor v1|v2` on `dashboard.py` and `cli/live_daemon.py`.

**MCP param:** `predictor` on `algotrade_start` tool.

### 7. New Indicator: ADX

`data/indicators.py` needs ADX added to `add_indicators()`:

```python
adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
out["adx"] = adx_df.filter(like="ADX").iloc[:, 0]  # ADX line (named column lookup)
```

All other indicators (EMA-12, SMA-20, MACD, RSI, StochRSI, ROC-5, BB, ATR, vol_sma_20) are already computed.

### 8. Backtest Methodology

Same infrastructure as v1 backtest. Run both predictors on identical 90-day data.

**Comparison:**
1. v1 predictor → WR, bets, PF, per-asset (existing baseline)
2. v2 predictor with v1 thresholds → same metrics
3. v2 threshold sweep (25-55, step 5) → find optimal per-asset thresholds
4. v2 with optimized thresholds → final comparison against v1
5. v2 with vs without penalty layer → verify penalties improve WR

**What to watch for:**
- v2 should generate more signals (trends are more common than extremes)
- Per-asset breakdown may reveal some assets trend more vs mean-revert
- Penalty layer effectiveness: penalized signals should have lower WR than unpenalized

## Files to Create

| File | Purpose |
|------|---------|
| `strategy/strategies/kalshi_predictor_v2.py` | V2 continuation predictor with 9 trend components + 4 penalty components |
| `tests/test_kalshi_v2.py` | Tests for v2 predictor |

## Files to Modify

| File | Change |
|------|--------|
| `data/indicators.py` | Add ADX to `add_indicators()` |
| `cli/live_daemon.py` | Add `--predictor v1\|v2` flag, predictor selection in `__init__()` |
| `dashboard.py` | Pass `--predictor` flag through |
| `mcp_server.py` | Add `predictor` param to `algotrade_start` |
| `backtest_kalshi.py` | Add v2 comparison run |
| `tests/test_indicators.py` | Add test for ADX column |

## Files NOT Modified

- `strategy/strategies/kalshi_predictor.py` — v1 stays untouched
- `exchange/kalshi.py` — API client unchanged
- `config/production.py` — shared config
- `data/market_data.py` — shared leading indicators

## Success Criteria

- v2 WR >= 65% on 90-day backtest
- More total bets than v1 (capturing continuation setups v1 misses)
- Profit factor >= 3.0
- Penalty layer demonstrably improves WR vs no-penalty version
- v1 continues to function identically when selected
