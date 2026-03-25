# Kalshi V3 Strike-Relative Predictor

**Date:** 2026-03-25
**Goal:** Build a strike-relative probability predictor that answers the actual Kalshi question: "will price be above or below the strike at settlement?" Uses a statistically calibrated base probability from historical data, adjusted by real-time technical signals.

## Problem

V1 and V2 both answer the wrong question. Kalshi 15m contracts ask: "will the 60-second average price at settlement be >= the 60-second average price at open (the strike)?" Our predictors answer: "will the next candle go UP or DOWN?" These are fundamentally different questions.

V1 (mean-reversion, 63% WR) and V2 (continuation, 46% WR) both miss the key insight: **distance from the strike is the dominant variable.** If BTC is 2 ATR above the strike with 5 minutes left, it almost certainly stays above — regardless of what RSI says. If it's 0.1 ATR from the strike, technical signals matter a lot.

## Architecture

```
Kalshi API → strike price + contract prices
Market Data → current price + indicators
                ↓
Strike-Relative Probability Model:
    1. distance = (current_price - strike) / ATR
    2. base_probability = lookup_table[distance_bucket][time_bucket]
    3. adjusted_probability = base + technical_adjustments
    4. Compare adjusted_probability vs contract price
    5. Bet if edge > 5c margin
```

## Design

### 1. Probability Lookup Table

**Built from historical data.** For each historical 15m window:
- Open price = simulated strike
- 1m sub-candles = observations at each minute within the window
- Close >= open = YES outcome, close < open = NO outcome

**Table structure:** 2D grid of (distance_bucket, time_bucket) → historical win rate.

**Distance buckets (in ATR units) — assigned by nearest-center:**
Centers: `[-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0, +0.25, +0.5, +1.0, +1.5, +2.0, +3.0]`
Bucket assignment: `bucket = min(centers, key=lambda c: abs(distance - c))`. Values beyond ±3.0 are clamped to ±3.0.

**Time buckets (minutes remaining):**
`[14, 12, 10, 8, 6, 4, 2]`

Each cell contains the historical probability that price closes >= strike, given that at the observation time it was in that distance bucket with that many minutes remaining.

**Data source:** 3-6 months of 1m candles from BinanceUS. Fetched in weekly chunks (10,080 candles/week per asset) to avoid timeout issues.

**Building process:**
1. Fetch 1m candles for all assets
2. Group into 15m windows (every 15 candles)
3. For each 1m candle within each window:
   - Compute ATR from recent 15m candles (for distance normalization)
   - Compute distance = (current_1m_close - window_open) / ATR
   - Assign to (distance_bucket, time_bucket)
   - Record outcome (did window close >= open?)
4. Aggregate: win_rate per cell = count(YES) / count(all)
5. Save as JSON

**Storage:** `data/store/kalshi_prob_table.json`

**Refresh cadence:** Rebuild weekly or monthly. The underlying statistics (how far price moves in 15m relative to ATR) are stable across market regimes.

**ATR computation:** Aggregate 1m candles to 15m (resample), compute ATR(14) on the 15m series. Requires 14 prior 15m candles (~3.5 hours warmup) before the first valid observation.

**Minimum sample size:** Require at least 30 observations per cell. Cells with fewer observations use 0.5 (neutral) as the default — no interpolation from neighbors to keep it simple. In practice, 90 days of data across 5 assets provides ~45,000 15m windows × multiple time observations each, so most cells will have adequate samples.

**Data quality:** The build script should report any gaps in 1m data (missing candles) per asset before building.

### 2. Technical Adjustments

Applied to the base probability from the lookup table. Adjustments are in absolute probability points, not relative.

**Positive adjustments (price likely to stay on current side):**

| Signal | Adjustment | Logic |
|--------|-----------|-------|
| Order book confirms | +5% | OB imbalance > 0.2 supporting current side |
| Trade flow confirms | +5% | Net flow > 0.15 AND buy ratio aligned with current side |
| 1h trend aligned | +5% | 1h RSI + MACD agree with current position relative to strike |
| MACD momentum building | +3% | 15m MACD histogram growing in direction of current side |

**Negative adjustments (potential crossing):**

| Signal | Adjustment | Logic |
|--------|-----------|-------|
| RSI extreme | -8% | RSI > 75 when above strike, or RSI < 25 when below — overextension risk |
| RSI divergence | -8% | Price making new extreme vs strike but RSI reversing — momentum fading. Requires 3+ candles. |
| Order book opposes | -5% | OB imbalance > 0.2 against current side |
| Trade flow opposes | -5% | Net flow + buy ratio shifting against current side |
| 1h trend opposes | -5% | Hourly trend fighting current position |

**Range: -31% to +18%.** Clamped so final probability stays in [0.05, 0.95].

All adjustment values are tunable via backtest.

### 3. Bet Decision Logic

```python
# After computing adjusted_probability:
EDGE_MARGIN = 0.05  # require 5c edge over market price
MAX_YES_PRICE = 50  # hard cap on YES contracts (cents)
MAX_NO_PRICE = 50   # hard cap on NO contracts (cents)

if adjusted_probability >= 0.55:
    # Price likely stays above strike (or crosses above)
    yes_price = query_kalshi_orderbook(YES side)
    if yes_price < (adjusted_probability * 100) - (EDGE_MARGIN * 100):
        if yes_price <= MAX_YES_PRICE:
            → BET YES at yes_price

elif adjusted_probability <= 0.45:
    # Price likely stays below strike (or crosses below)
    no_price = query_kalshi_orderbook(NO side)
    no_probability = 1.0 - adjusted_probability  # e.g. 0.70 if adj_prob = 0.30
    if no_price < (no_probability * 100) - (EDGE_MARGIN * 100):
        if no_price <= MAX_NO_PRICE:
            → BET NO at no_price

else:
    → SKIP (too close to call, no edge)
```

**Key properties:**
- Bets BOTH sides — YES when probability is high, NO when probability is low
- Edge margin prevents betting when our estimate barely beats the market
- Max price caps prevent overpaying regardless of probability estimate
- The probability directly maps to fair contract price — no arbitrary confidence thresholds

### 4. Signal Output

**New dataclass:**
```python
@dataclass
class KalshiV3Signal:
    asset: str
    probability: float        # 0.0-1.0 that price closes >= strike
    recommended_side: str     # "YES", "NO", or "SKIP"
    max_price_cents: int      # max we should pay for the recommended side
    distance_atr: float       # distance from strike in ATR units
    base_prob: float          # from lookup table (before adjustments)
    adjustments: dict         # breakdown of each technical adjustment
    current_price: float
    strike_price: float
    minutes_remaining: float
```

### 5. Predictor Interface

**New file:** `strategy/strategies/kalshi_predictor_v3.py`

```python
class KalshiPredictorV3:
    def __init__(self, prob_table_path="data/store/kalshi_prob_table.json"):
        """Load pre-computed probability lookup table."""

    def predict(self, df: pd.DataFrame, strike_price: float,
                minutes_remaining: float, market_data: dict | None = None,
                df_1h: pd.DataFrame | None = None) -> KalshiV3Signal | None:
        """Compute strike-relative probability and bet recommendation.

        Args:
            df: 15m OHLCV with indicators
            strike_price: Kalshi contract floor_strike
            minutes_remaining: minutes until settlement
            market_data: order book + trade flow (optional)
            df_1h: 1h OHLCV with indicators for MTF (optional)
        """

    @staticmethod
    def build_probability_table(historical_1m: dict[str, pd.DataFrame],
                                 output_path: str = "data/store/kalshi_prob_table.json"):
        """Build lookup table from historical 1m data.

        Args:
            historical_1m: {asset_name: DataFrame of 1m candles}
        """
```

**Different interface from V1/V2.** V3 requires strike_price and minutes_remaining as inputs. The daemon's `_kalshi_eval()` must query Kalshi for the active contract to get these values.

### 6. Daemon Integration

**`_kalshi_eval()` changes for `--predictor v3`:**

**Contract discovery at SETUP:** Query `client.get_markets(series_ticker=..., status="open")` for each asset. Extract `floor_strike` and `close_time` from the first result. Use `close_time` field, fall back to `expiration_time` if missing. If no contract is available (market closed, gap between contracts), skip the asset for this window — log and continue.

**Strike is constant per window:** Once discovered at SETUP, the strike and close_time are cached in the pending signal dict for CONFIRMED/DOUBLE_CONFIRMED. No need to re-query Kalshi at each eval — the contract doesn't change within a 15m window.

**Pending signal state for V3:**
```python
{
    "strike_price": 70975.23,    # from floor_strike
    "close_time": datetime,       # from close_time
    "probability": 0.72,          # latest adjusted probability
    "recommended_side": "YES",    # latest recommendation
    "max_price_cents": 45,        # latest max price
    "distance_atr": 1.5,          # latest distance
    "bet_placed": False,
    "window_start": datetime,
}
```

**SETUP (minute 0-4):** Query Kalshi for contract, extract strike + close_time. Fetch 15m candles + indicators. Call `predictor.predict(df, strike, minutes_remaining, market_data, df_1h)`. Store signal in pending. No betting.

**CONFIRMED (minute 5-9):** Fetch fresh 15m candles + leading indicators. Compute `minutes_remaining` from cached `close_time`. Call `predictor.predict()` with fresh data + same strike. If `recommended_side` != "SKIP" → actionable.

**DOUBLE_CONFIRMED (minute 10-11):** Same as CONFIRMED with even less time remaining.

**LAST_LOOK (minute 12):** V3 already accounts for time remaining in its probability. Just re-run `predict()` with `minutes_remaining=3`. No need for `check_1m_momentum()` — V3's model handles the time dimension directly. If V3, skip the 1m momentum check.

**`_kalshi_execute_bet()` changes for V3:**
When signal is a `KalshiV3Signal`:
- `side` = `"yes"` if `recommended_side == "YES"` else `"no"`
- `fill_price` = `min(signal.max_price_cents, MAX_ENTRY_CENTS)` — use V3's max price directly instead of the orderbook pricing logic
- Still query the orderbook to verify we can actually get filled at that price — if best ask > max_price, skip
- Rest of execution (balance check, order placement, fill verification) stays the same

**Lifecycle still applies:** SETUP (minute 0-4, no bet), CONFIRMED (minute 5-9, bet if edge), etc. The wall-clock aligned timer and confirmation gate work the same way. V3 computes a probability at each evaluation instead of a confidence score.

**No-contract handling:** If `get_markets()` returns empty for an asset, the SETUP stores nothing for that asset. CONFIRMED sees no pending signal and skips. This is the existing "no SETUP" handling.

### 7. Backtest Methodology

**Phase 1: Build probability table (first 2 months of data)**

1. Fetch 1m candles for all assets, chunked by week
2. For each asset, group 1m candles into 15m windows
3. Compute ATR from 15m candles (need 15m data too, or compute from 1m aggregation)
4. For each 1m observation within each window: record (distance_bucket, time_bucket, outcome)
5. Aggregate into probability table
6. Save to JSON

**Phase 2: Walk-forward validation (last 1 month of data)**

1. Load the probability table from Phase 1
2. For each 15m window in the validation period:
   - Simulate strike = window open price
   - At minutes 5, 10: compute distance, look up base prob, apply technical adjustments
   - If adjusted_probability gives edge (>55% or <45%) at simulated contract price (use 50c as default): record bet
   - Check actual outcome
3. Report:
   - **Calibration:** When model says 70%, does it win ~70%?
   - **WR at different probability thresholds:** only bet when prob > 60%, > 70%, > 80%
   - **Edge:** average (probability - contract_price_paid) across all bets
   - **Comparison vs V1** on same validation data

**1m data fetching:** Chunked by week (~10,080 candles per asset per week). For 3 months = ~13 weeks = ~131,000 candles per asset. 5 assets = ~655,000 total 1m candles. Fetch sequentially with rate limiting.

## Files to Create

| File | Purpose |
|------|---------|
| `strategy/strategies/kalshi_predictor_v3.py` | V3 strike-relative predictor with probability table + technical adjustments |
| `tests/test_kalshi_v3.py` | Tests for V3 predictor |
| `data/store/kalshi_prob_table.json` | Pre-computed probability lookup table (generated, not checked in) |
| `scripts/build_prob_table.py` | Script to build/refresh the probability table from historical data |

## Files to Modify

| File | Change |
|------|--------|
| `cli/live_daemon.py` | Add `v3` to predictor version selection. Update `_kalshi_eval()` SETUP/CONFIRMED to query Kalshi strike + compute time remaining for V3. Update `_kalshi_execute_bet()` to accept V3's recommended_side + max_price. |
| `dashboard.py` | Show probability + distance in Kalshi predictions display (instead of confidence) |
| `mcp_server.py` | Add `v3` to predictor enum |
| `backtest_kalshi.py` | Add V3 walk-forward validation section |

## Files NOT Modified

| File | Reason |
|------|--------|
| `strategy/strategies/kalshi_predictor.py` | V1 stays untouched |
| `strategy/strategies/kalshi_predictor_v2.py` | V2 stays (shelved but available) |
| `exchange/kalshi.py` | `get_markets()` returns raw dicts that contain `floor_strike` and `close_time`. No parsing changes needed — the daemon extracts these fields directly from the dict. |
| `data/indicators.py` | All needed indicators already present (RSI, MACD, BB, ATR, etc.) |

## Success Criteria

- **Calibration:** When model predicts 70%, actual win rate is 65-75%
- **WR >= 65%** on walk-forward validation (bets taken when edge > 5c)
- **Positive edge:** average probability - average price paid > 5c across all bets
- **Both sides profitable:** YES bets and NO bets both have positive WR
- **More bets than V1:** captures "stay on current side" setups that V1 misses entirely
- V1 continues to function when selected
