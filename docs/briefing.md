# K15 UpDown — Development Briefing

## What This Is

An automated Kalshi 15-minute crypto prediction bot that bets YES/NO on whether BTC, ETH, SOL, XRP, and BNB will close above or below a target price within a 15-minute window. Uses a hybrid KNN machine learning model + statistical probability table for prediction, with tiered position sizing.

## How To Run

```bash
# Dry-run (no real money):
./venv/bin/python dashboard.py --dry-run --kalshi-only --predictor v3 --cycles 10

# Live (ONLY with explicit permission):
./venv/bin/python dashboard.py --live --kalshi-only --predictor v3 --cycles 10

# Retrain models:
./venv/bin/python scripts/build_knn_model.py --days 90 --k 50
./venv/bin/python scripts/build_prob_table.py --days 90
```

---

## Development History

### Phase 1: V1 Mean-Reversion Predictor

**Starting point:** An existing Kalshi predictor with 11 scoring components (RSI, BB, MACD, volume, order book, trade flow) that answered "will the next candle go UP or DOWN?"

**Optimizations applied:**
- Added 4 new indicators: StochRSI, ROC-5, ATR Move Ratio, 1-Hour Trend Alignment
- Added 3 signal quality filters: directional conflict, volatility regime, margin of victory
- Added per-asset confidence thresholds (BTC/XRP: 30, ETH/SOL/BNB: 35)
- Replaced DOGE with BNB (better WR)
- Separated Kalshi concurrency from spot trading (max 2 Kalshi bets)

**Result:** 56% WR → **63% WR** (+7pp). 1h trend alignment was the biggest single improvement.

### Phase 2: Multi-Timeframe Evaluation

**Goal:** Refresh predictions every 5 minutes instead of every 15.

**Attempts:**
1. **5m full re-scoring:** 25% WR — non-viable. RSI/MACD/BB are calibrated for 15m. On 5m data they produce noise.
2. **5m booster (+3 per factor):** Inversely correlated with WR. Trend confirmation HURTS a mean-reversion predictor — the move has already happened.
3. **5m penalty-only:** Fired on <1% of bets. No meaningful improvement.

**Final approach:** Keep 15m scoring, refresh leading indicators (order book, trade flow) every 5 minutes. Wall-clock aligned eval at minutes :01, :06, :11, :12.

**Key finding:** The predictor's mean-reversion nature (RSI, BB, StochRSI) means trend confirmation signals are counterproductive.

### Phase 3: V2 Continuation Predictor

**Hypothesis:** Maybe trend/continuation is better than mean-reversion for 15m predictions.

**Built:** 9 trend components (price vs MAs, ADX, MACD trend, consecutive candles, HH/HL, ROC, volume) + 4 mean-reversion penalty components.

**Result:** **46% WR** — worse than a coin flip. Shelved.

**Key finding:** Neither mean-reversion NOR continuation reliably predicts the next 15m candle direction. The question itself was wrong.

### Phase 4: V3 Strike-Relative Model (The Breakthrough)

**The insight:** V1 and V2 both answered "will price go UP or DOWN?" — but Kalshi asks "will price close above or below THIS SPECIFIC STRIKE PRICE?" These are fundamentally different questions.

**Approach:**
1. Get the strike price from Kalshi API (`floor_strike`)
2. Compute distance from strike in ATR units
3. Look up historical probability from a pre-computed table
4. Adjust with real-time technical signals
5. Compare our probability against the contract price
6. Bet when we have edge

**Built probability table** from 90 days of 1m candle data (604,730 observations). The table shows clear monotonic signal: +1 ATR above strike with 2 min left = 99% probability.

**Initial results:** 68.8% WR on 90-day walk-forward. All 5 assets above 65%.

### Phase 5: The P&L Problem

**Discovery:** 90% WR at minute 10 produced NEGATIVE P&L (-$5.68 over 61 bets). Why? Contracts cost 80-90c at that point. Win 10-20c per win, lose 80-90c per loss. Need 85%+ WR just to break even at 85c entry.

**The math:**
- At 50c contracts: 55% WR = +10% ROI
- At 80c contracts: 90% WR = barely breakeven
- The market efficiently prices the distance — no free money in "wait and buy the obvious"

**Key finding:** Profit requires EARLY entry (minute 0-3) when contracts are near 50c, with prediction accuracy above 55%.

### Phase 6: KNN Machine Learning Model

**Research:** Studied "151 Trading Strategies" (SSRN paper) which described neural networks on BTC 15-minute intervals using multi-timeframe features.

**Single-feature analysis showed:**
- Most indicators at minute 0 are essentially 50/50 (no edge)
- 1h RSI > 70 → 62% UP (best single feature, but rare)
- MACD alignment across 3 timeframes = 0% edge (useless)

**KNN approach:** Instead of hand-tuning rules, use K-Nearest Neighbors to find historically similar setups across 12 features simultaneously:
1. RSI (15m), StochRSI (15m), MACD (15m)
2. Normalized returns, volume ratio, BB position
3. EMA slope, ADX, ROC-5
4. RSI (1h), MACD (1h), RSI (4h)

**Walk-forward results (train 60 days, test 30 days):**

| Config | WR | Bets/month | ROI @50c |
|--------|-----|-----------|---------|
| K=50 conf≥55% | 59.2% | 9,866 | +18.4% |
| K=50 conf≥60% | 61.3% | 6,178 | +22.6% |
| K=50 conf≥65% | 62.7% | 3,173 | +25.5% |

**Optimal config:** K=50 at ≥60% confidence — 61.3% WR on 6,178 bets with +22.6% ROI.

**Critical finding:** Raw KNN probability is stronger than KNN + V3 adjustments. The hand-tuned adjustments (momentum gate, wrong-side cap, OB/flow) were HURTING the KNN signal because KNN already captures all multi-timeframe context internally.

### Phase 7: Production Hardening

**Price source fix:** BinanceUS has a $15-30 spread vs CF Benchmarks BRTI (Kalshi's settlement source). Switched to Coinbase price for distance calculation (<$1 spread).

**Wrong-side cap:** Base probability capped at 50% when price is on wrong side of strike. Prevents the 0.0 ATR bucket's bullish bias from betting YES when below target.

**Momentum gate:** -15% penalty when 3 consecutive candles move toward the strike (against our bet). Prevents betting YES on a rapidly falling price.

**Market orders → Limit orders:** Kalshi rejects market orders with 400. Limit orders at the ask give instant-fill behavior.

**Settlement tracking:** Queries Kalshi API for authoritative `result` and `expiration_value` — no approximation from Coinbase price.

**Tiered position sizing:**

| Confidence | Max Risk |
|-----------|----------|
| 70%+ | 10% of balance |
| 65-69% | 7.5% |
| 60-64% | 5% |
| 55-59% | 2.5% |

---

## Current Architecture

```
Kalshi API → strike price + contract prices + settlement results
Coinbase API → live price (closer to BRTI than BinanceUS)
BinanceUS → 15m/1h/4h candles for indicator computation
    ↓
KNN Model (early entry, near strike):
    12 features across 3 timeframes → probability of UP
    ↓
Probability Table (late entry, far from strike):
    distance_ATR × time_remaining → base probability
    + technical adjustments → adjusted probability
    ↓
Bet Decision:
    probability >= 60% → YES
    probability <= 40% → NO
    otherwise → SKIP
    ↓
Position Sizing:
    tiered by confidence (2.5% - 10% of balance)
    ↓
Execution:
    limit order at ask price for instant fill
    max 2 concurrent bets
    hold to settlement
```

## Key Files

| File | Purpose |
|------|---------|
| `strategy/strategies/kalshi_predictor_v3.py` | V3 predictor: KNN + probability table + bet decision |
| `cli/live_daemon.py` | Trading daemon with lifecycle state machine |
| `dashboard.py` | ASCII dashboard with Kalshi-only mode |
| `scripts/build_knn_model.py` | Train KNN model on historical data |
| `scripts/build_prob_table.py` | Build probability lookup table |
| `models/knn_kalshi.pkl` | Trained KNN model (refresh periodically) |
| `data/store/kalshi_prob_table.json` | Probability table (refresh periodically) |
| `exchange/kalshi.py` | Kalshi REST API client |
| `k15.py` | Rich interactive dashboard (WIP — input rendering issues) |

## Lessons Learned

1. **Ask the right question.** V1/V2 failed because they predicted "up or down?" when Kalshi asks "above or below this strike?" Reframing the question improved WR from 46-63% to 68%.

2. **The market prices distance efficiently.** Late-entry bets (minute 10+) have 90% WR but negative P&L because contracts cost 80-90c. The edge is in EARLY entry when contracts are near 50c.

3. **ML beats hand-tuned rules.** KNN with 12 features outperforms any single indicator or hand-crafted rule combination. The raw KNN signal is stronger than KNN + manual adjustments.

4. **Mean-reversion ≠ continuation ≠ strike-relative.** Three fundamentally different questions that require different models. Mixing them (like adding trend confirmation to a mean-reversion predictor) makes things worse.

5. **Price source matters enormously.** A $15 BinanceUS-BRTI spread caused wrong-side bets. Using Coinbase ($1 spread) fixed it.

6. **Backtest integrity is critical.** Our first V3 backtests showed 75% WR — inflated by data leakage (using candle close as both current price and settlement). Proper 5m backtesting with no future data showed the real numbers.

7. **P&L, not WR, is the metric.** 90% WR means nothing if contracts cost 90c and you only win 10c. The ratio of win_profit / loss_cost × WR determines profitability.
