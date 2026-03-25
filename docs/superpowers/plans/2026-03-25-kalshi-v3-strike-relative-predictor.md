# Kalshi V3 Strike-Relative Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strike-relative probability predictor that answers the actual Kalshi question — "will price close above or below this specific strike?" — using a historical probability lookup table adjusted by real-time technical signals.

**Architecture:** Pre-computed 2D probability table (distance_ATR x time_remaining) from historical 1m data provides base probability. Real-time technical signals (OB, trade flow, RSI, MACD, 1h trend) adjust the probability ±31%. Bet decision compares adjusted probability against actual Kalshi contract price with a 5c edge margin. Bets both YES and NO sides.

**Tech Stack:** Python 3, pandas, pandas_ta, numpy, pytest, JSON

**Spec:** `docs/superpowers/specs/2026-03-25-kalshi-v3-strike-relative-predictor-design.md`

---

### Task 1: Build probability table script

**Files:**
- Create: `scripts/build_prob_table.py`
- Test: manual run + inspect output

This is the foundation — a standalone script that fetches historical 1m data, computes the probability table, and saves it as JSON.

- [ ] **Step 1: Create the script**

Create `scripts/build_prob_table.py`:

```python
#!/usr/bin/env python3
"""Build the Kalshi V3 probability lookup table from historical 1m candle data.

For each historical 15m window, the open price = simulated strike.
At each 1m mark, compute distance from strike in ATR units.
Record outcome: did the 15m window close >= open?
Aggregate into a 2D table: P(close >= strike | distance_bucket, time_bucket).

Usage:
    ./venv/bin/python scripts/build_prob_table.py [--days 90] [--output data/store/kalshi_prob_table.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators

ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
    "BNB": "BNB/USDT",
}

# Distance buckets (in ATR units)
DISTANCE_BINS = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
# Time buckets (minutes remaining in the 15m window)
TIME_BINS = [14, 12, 10, 8, 6, 4, 2]


def fetch_1m_chunked(fetcher: DataFetcher, symbol: str, days: int) -> pd.DataFrame:
    """Fetch 1m candles in weekly chunks to avoid timeouts."""
    all_frames = []
    now_ms = int(time.time() * 1000)
    chunk_days = 7
    candle_ms = 60 * 1000  # 1 minute

    total_chunks = (days + chunk_days - 1) // chunk_days
    for chunk in range(total_chunks):
        start_day = days - (chunk + 1) * chunk_days
        end_day = days - chunk * chunk_days
        if start_day < 0:
            start_day = 0

        since = now_ms - end_day * 24 * 60 * 60 * 1000
        until = now_ms - start_day * 24 * 60 * 60 * 1000

        batch_since = since
        while batch_since < until:
            try:
                df = fetcher.ohlcv(symbol, "1m", limit=1000, since=batch_since)
                if df is None or df.empty:
                    break
                all_frames.append(df)
                last_ts = int(df.index[-1].timestamp() * 1000)
                batch_since = last_ts + candle_ms
                if len(df) < 1000:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"    Warning: {e}")
                batch_since += 1000 * candle_ms
                time.sleep(1)

        print(f"    Chunk {chunk+1}/{total_chunks}: {len(all_frames)} batches so far")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def compute_15m_atr(df_1m: pd.DataFrame) -> pd.Series:
    """Compute ATR from 1m candles aggregated to 15m for distance normalization."""
    # Resample 1m to 15m
    df_15m = df_1m.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    if len(df_15m) < 15:
        return pd.Series(dtype=float)

    # Compute ATR(14) on 15m
    import pandas_ta as ta
    atr = ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
    return atr


def bucket_distance(distance_atr: float) -> float:
    """Assign a distance value to the nearest bucket."""
    closest = min(DISTANCE_BINS, key=lambda b: abs(b - distance_atr))
    return closest


def build_table(df_1m: pd.DataFrame, asset_name: str) -> dict:
    """Build probability observations from 1m candles for one asset."""
    observations = []  # list of (distance_bucket, time_bucket, outcome)

    # Compute 15m ATR series
    atr_series = compute_15m_atr(df_1m)
    if atr_series.empty:
        print(f"  {asset_name}: insufficient data for ATR")
        return {}

    # Group 1m candles into 15m windows
    # Window starts at :00, :15, :30, :45
    df_1m = df_1m.copy()
    df_1m["window_start"] = df_1m.index.floor("15min")
    df_1m["minute_in_window"] = (df_1m.index - df_1m["window_start"]).total_seconds() / 60

    windows = df_1m.groupby("window_start")

    for window_start, group in windows:
        if len(group) < 14:  # need nearly complete window
            continue

        # Strike = open of the window (first 1m candle open)
        strike = float(group.iloc[0]["open"])
        # Outcome = did window close >= strike?
        window_close = float(group.iloc[-1]["close"])
        outcome = 1 if window_close >= strike else 0

        # Get ATR for this window (from the 15m ATR series)
        # Find the nearest 15m ATR value at or before this window
        atr_mask = atr_series.index <= window_start
        if atr_mask.sum() == 0:
            continue
        atr_val = float(atr_series.loc[atr_mask].iloc[-1])
        if atr_val <= 0:
            continue

        # For each 1m candle in the window, record an observation
        for _, row in group.iterrows():
            minute = int(row["minute_in_window"])
            minutes_remaining = 14 - minute
            if minutes_remaining < 1 or minutes_remaining > 14:
                continue

            # Find nearest time bucket
            closest_time = min(TIME_BINS, key=lambda t: abs(t - minutes_remaining))

            current_price = float(row["close"])
            distance = (current_price - strike) / atr_val
            dist_bucket = bucket_distance(distance)

            observations.append({
                "distance": dist_bucket,
                "time": closest_time,
                "outcome": outcome,
            })

    return observations


def aggregate_observations(all_observations: list[dict]) -> dict:
    """Aggregate observations into probability table."""
    table = {}

    for obs in all_observations:
        key = f"{obs['distance']}_{obs['time']}"
        if key not in table:
            table[key] = {"wins": 0, "total": 0}
        table[key]["total"] += 1
        table[key]["wins"] += obs["outcome"]

    # Convert to probabilities
    prob_table = {}
    for key, counts in table.items():
        dist_str, time_str = key.split("_")
        if counts["total"] >= 30:  # minimum sample size
            prob = counts["wins"] / counts["total"]
        else:
            prob = 0.5  # insufficient data, default to 50%
        prob_table[key] = {
            "probability": round(prob, 4),
            "sample_size": counts["total"],
            "distance": float(dist_str),
            "time_remaining": int(float(time_str)),
        }

    return prob_table


def main():
    parser = argparse.ArgumentParser(description="Build Kalshi V3 probability table")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--output", default="data/store/kalshi_prob_table.json")
    args = parser.parse_args()

    fetcher = DataFetcher()
    all_observations = []

    for asset_name, symbol in ASSETS.items():
        print(f"\nFetching {asset_name} 1m candles ({args.days} days)...")
        df_1m = fetch_1m_chunked(fetcher, symbol, args.days)
        if df_1m.empty:
            print(f"  {asset_name}: no data")
            continue
        print(f"  {asset_name}: {len(df_1m)} candles ({df_1m.index[0]} to {df_1m.index[-1]})")

        print(f"  Building observations...")
        obs = build_table(df_1m, asset_name)
        if isinstance(obs, list):
            all_observations.extend(obs)
            print(f"  {asset_name}: {len(obs)} observations")

    print(f"\nTotal observations: {len(all_observations)}")
    print("Aggregating into probability table...")

    prob_table = aggregate_observations(all_observations)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(prob_table, f, indent=2)

    print(f"\nSaved probability table to {output_path}")
    print(f"Cells: {len(prob_table)}")

    # Print summary
    print("\nProbability Table Summary:")
    print(f"{'Distance':>10} | " + " | ".join(f"{t:>4}min" for t in TIME_BINS))
    print("-" * 70)
    for dist in DISTANCE_BINS:
        row = []
        for t in TIME_BINS:
            key = f"{dist}_{t}"
            cell = prob_table.get(key, {})
            prob = cell.get("probability", 0.5)
            n = cell.get("sample_size", 0)
            row.append(f"{prob:.0%}({n:>3})" if n >= 30 else "  ---  ")
        print(f"{dist:>+10.2f} | " + " | ".join(row))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script on 30 days first (faster, validates structure)**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python scripts/build_prob_table.py --days 30
```

Verify: JSON file created at `data/store/kalshi_prob_table.json`, probability table summary printed.

- [ ] **Step 3: Run on 90 days for the full table**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python scripts/build_prob_table.py --days 90
```

This will take longer (~15-30 min for 1m data fetching). Run in background if needed.

- [ ] **Step 4: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add scripts/build_prob_table.py && git commit -m "feat: add probability table builder for V3 strike-relative predictor

Fetches historical 1m candles, groups into 15m windows, computes
distance-from-strike in ATR units at each minute, records outcome.
Aggregates into 2D probability table (distance x time_remaining).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create `KalshiPredictorV3` with probability model

**Files:**
- Create: `strategy/strategies/kalshi_predictor_v3.py`
- Create: `tests/test_kalshi_v3.py`

- [ ] **Step 1: Create test file**

Create `tests/test_kalshi_v3.py`:

```python
# tests/test_kalshi_v3.py
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch
from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3, KalshiV3Signal


@pytest.fixture
def sample_prob_table(tmp_path):
    """Create a simple probability table for testing."""
    table = {}
    # Above strike = high probability
    for dist in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for t in [14, 12, 10, 8, 6, 4, 2]:
            key = f"{dist}_{t}"
            # Higher distance + less time = higher probability
            base = 0.5 + dist * 0.1 + (14 - t) * 0.01
            table[key] = {"probability": min(0.95, round(base, 4)),
                          "sample_size": 100, "distance": dist, "time_remaining": t}
    # Below strike = low probability of closing above
    for dist in [-0.5, -1.0, -1.5, -2.0, -3.0]:
        for t in [14, 12, 10, 8, 6, 4, 2]:
            key = f"{dist}_{t}"
            base = 0.5 + dist * 0.1 + (14 - t) * 0.01
            table[key] = {"probability": max(0.05, round(base, 4)),
                          "sample_size": 100, "distance": dist, "time_remaining": t}
    # At strike = 50%
    for t in [14, 12, 10, 8, 6, 4, 2]:
        key = f"0.0_{t}"
        table[key] = {"probability": 0.5, "sample_size": 100, "distance": 0.0, "time_remaining": t}
    # Near strike
    for dist in [0.25, -0.25]:
        for t in [14, 12, 10, 8, 6, 4, 2]:
            key = f"{dist}_{t}"
            base = 0.5 + dist * 0.1
            table[key] = {"probability": round(base, 4),
                          "sample_size": 100, "distance": dist, "time_remaining": t}

    path = tmp_path / "test_prob_table.json"
    with open(path, "w") as f:
        json.dump(table, f)
    return str(path)


def _make_df(n=50, close=100.0, rsi=50.0, stochrsi_k=50.0, macd_hist=0.0,
             atr=2.0, bb_lower=98.0, bb_upper=102.0):
    """Build a minimal DataFrame for V3 testing."""
    return pd.DataFrame({
        "close": np.full(n, close), "open": np.full(n, close),
        "high": np.full(n, close + 1), "low": np.full(n, close - 1),
        "volume": np.full(n, 1000.0),
        "rsi": np.full(n, rsi), "stochrsi_k": np.full(n, stochrsi_k),
        "macd_hist": np.full(n, macd_hist),
        "atr": np.full(n, atr), "vol_sma_20": np.full(n, 1000.0),
        "bb_lower": np.full(n, bb_lower), "bb_upper": np.full(n, bb_upper),
        "ema_12": np.full(n, close), "sma_20": np.full(n, close),
        "roc_5": np.full(n, 0.0), "adx": np.full(n, 25.0),
    }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))


class TestV3BaseProbability:
    """Tests for probability lookup."""

    def test_above_strike_high_probability(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=104.0, atr=2.0)  # 2 ATR above strike at 100
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=10)
        assert signal is not None
        assert signal.probability > 0.6
        assert signal.distance_atr > 1.5

    def test_below_strike_low_probability(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=96.0, atr=2.0)  # 2 ATR below strike
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=10)
        assert signal is not None
        assert signal.probability < 0.4

    def test_at_strike_near_50(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=100.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=10)
        assert signal is not None
        assert 0.4 <= signal.probability <= 0.6

    def test_less_time_more_certainty(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=103.0, atr=2.0)  # 1.5 ATR above
        sig_early = predictor.predict(df, strike_price=100.0, minutes_remaining=14)
        sig_late = predictor.predict(df, strike_price=100.0, minutes_remaining=2)
        assert sig_late.probability > sig_early.probability  # more certain with less time


class TestV3TechnicalAdjustments:
    """Tests for technical signal adjustments."""

    def test_rsi_extreme_reduces_probability(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df_normal = _make_df(close=103.0, atr=2.0, rsi=55.0)
        df_extreme = _make_df(close=103.0, atr=2.0, rsi=82.0)
        sig_normal = predictor.predict(df_normal, strike_price=100.0, minutes_remaining=10)
        sig_extreme = predictor.predict(df_extreme, strike_price=100.0, minutes_remaining=10)
        assert sig_extreme.probability < sig_normal.probability

    def test_no_adjustments_when_far_from_strike(self, sample_prob_table):
        """When very far from strike, adjustments are small relative to base."""
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=106.0, atr=2.0, rsi=80.0)  # 3 ATR above, RSI high
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=4)
        # Even with RSI penalty, probability should still be high
        assert signal.probability > 0.6


class TestV3BetDecision:
    """Tests for bet recommendation logic."""

    def test_high_prob_recommends_yes(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=104.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=6)
        assert signal.recommended_side in ("YES", "SKIP")
        if signal.probability >= 0.55:
            assert signal.recommended_side == "YES"

    def test_low_prob_recommends_no(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=96.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=6)
        assert signal.recommended_side in ("NO", "SKIP")

    def test_near_50_recommends_skip(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=100.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=10)
        assert signal.recommended_side == "SKIP"

    def test_max_price_cents_set_correctly(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=104.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=6)
        if signal.recommended_side == "YES":
            # max price should be probability * 100 - margin
            assert signal.max_price_cents <= int(signal.probability * 100)
            assert signal.max_price_cents <= 50  # hard cap


class TestV3SignalFields:
    """Tests for signal output completeness."""

    def test_signal_has_all_fields(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(close=103.0, atr=2.0)
        signal = predictor.predict(df, strike_price=100.0, minutes_remaining=10)
        assert signal is not None
        assert isinstance(signal.probability, float)
        assert isinstance(signal.recommended_side, str)
        assert isinstance(signal.max_price_cents, int)
        assert isinstance(signal.distance_atr, float)
        assert isinstance(signal.base_prob, float)
        assert isinstance(signal.adjustments, dict)
        assert isinstance(signal.current_price, float)
        assert signal.strike_price == 100.0
        assert signal.minutes_remaining == 10

    def test_insufficient_data_returns_none(self, sample_prob_table):
        predictor = KalshiPredictorV3(prob_table_path=sample_prob_table)
        df = _make_df(n=5)
        assert predictor.predict(df, strike_price=100.0, minutes_remaining=10) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi_v3.py -v
```

- [ ] **Step 3: Implement `KalshiPredictorV3`**

Create `strategy/strategies/kalshi_predictor_v3.py`:

```python
# strategy/strategies/kalshi_predictor_v3.py
"""V3 Kalshi predictor: strike-relative probability model.

Answers the actual Kalshi question: "will price close above or below
this specific strike price?" Uses a pre-computed probability lookup
table calibrated from historical data, adjusted by real-time signals.
"""
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

DISTANCE_BINS = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TIME_BINS = [14, 12, 10, 8, 6, 4, 2]

EDGE_MARGIN = 0.05    # require 5% edge over implied contract price
MAX_BET_PRICE = 50    # never pay more than 50c for either side


@dataclass
class KalshiV3Signal:
    asset: str
    probability: float          # 0.0-1.0 that price closes >= strike
    recommended_side: str       # "YES", "NO", or "SKIP"
    max_price_cents: int        # max we should pay for recommended side
    distance_atr: float         # distance from strike in ATR units
    base_prob: float            # from lookup table before adjustments
    adjustments: dict           # breakdown of each technical adjustment
    current_price: float
    strike_price: float
    minutes_remaining: float


class KalshiPredictorV3:
    """Strike-relative probability predictor for Kalshi 15m contracts."""

    def __init__(self, prob_table_path: str = "data/store/kalshi_prob_table.json"):
        self._prob_table = {}
        try:
            with open(prob_table_path) as f:
                self._prob_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # empty table — will return 0.5 for everything

    def predict(self, df: pd.DataFrame, strike_price: float,
                minutes_remaining: float, market_data: dict | None = None,
                df_1h: pd.DataFrame | None = None) -> KalshiV3Signal | None:
        """Compute strike-relative probability and bet recommendation."""
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        current_price = float(last["close"])
        atr = float(last.get("atr", 0)) if pd.notna(last.get("atr")) else 0

        if atr <= 0 or strike_price <= 0:
            return None

        # 1. Compute distance from strike in ATR units
        distance_atr = (current_price - strike_price) / atr

        # 2. Look up base probability
        base_prob = self._lookup_probability(distance_atr, minutes_remaining)

        # 3. Apply technical adjustments
        adjustments = self._compute_adjustments(
            df, current_price, strike_price, distance_atr, market_data, df_1h
        )
        total_adjustment = sum(adjustments.values())

        # 4. Compute adjusted probability
        adjusted_prob = max(0.05, min(0.95, base_prob + total_adjustment))

        # 5. Determine bet recommendation
        recommended_side, max_price = self._decide_bet(adjusted_prob)

        return KalshiV3Signal(
            asset="",
            probability=round(adjusted_prob, 4),
            recommended_side=recommended_side,
            max_price_cents=max_price,
            distance_atr=round(distance_atr, 3),
            base_prob=round(base_prob, 4),
            adjustments=adjustments,
            current_price=current_price,
            strike_price=strike_price,
            minutes_remaining=minutes_remaining,
        )

    def _lookup_probability(self, distance_atr: float, minutes_remaining: float) -> float:
        """Look up base probability from the pre-computed table."""
        # Find nearest distance bucket
        dist_bucket = min(DISTANCE_BINS, key=lambda b: abs(b - distance_atr))
        # Clamp to table range
        dist_bucket = max(DISTANCE_BINS[0], min(DISTANCE_BINS[-1], dist_bucket))

        # Find nearest time bucket
        time_bucket = min(TIME_BINS, key=lambda t: abs(t - minutes_remaining))

        key = f"{dist_bucket}_{time_bucket}"
        cell = self._prob_table.get(key, {})
        return cell.get("probability", 0.5)  # default 50% if cell missing

    def _compute_adjustments(self, df: pd.DataFrame, current_price: float,
                              strike_price: float, distance_atr: float,
                              market_data: dict | None, df_1h: pd.DataFrame | None) -> dict:
        """Compute technical signal adjustments to base probability."""
        adjustments = {}
        last = df.iloc[-1]
        above_strike = current_price >= strike_price

        # --- Positive adjustments (price likely stays on current side) ---

        # Order book confirms (+5%)
        if market_data:
            ob = market_data.get("order_book", {})
            imbalance = ob.get("imbalance", 0)
            if (above_strike and imbalance > 0.2) or (not above_strike and imbalance < -0.2):
                adjustments["ob_confirms"] = 0.05
            elif (above_strike and imbalance < -0.2) or (not above_strike and imbalance > 0.2):
                adjustments["ob_opposes"] = -0.05

        # Trade flow confirms (+5%)
        if market_data:
            tf = market_data.get("trade_flow", {})
            net_flow = tf.get("net_flow", 0)
            buy_ratio = tf.get("buy_ratio", 0.5)
            if (above_strike and net_flow > 0.15 and buy_ratio > 0.55) or \
               (not above_strike and net_flow < -0.15 and buy_ratio < 0.45):
                adjustments["flow_confirms"] = 0.05
            elif (above_strike and net_flow < -0.15) or (not above_strike and net_flow > 0.15):
                adjustments["flow_opposes"] = -0.05

        # 1h trend aligned (+5%)
        if df_1h is not None and len(df_1h) >= 20:
            last_1h = df_1h.iloc[-1]
            rsi_1h = float(last_1h.get("rsi", 50)) if pd.notna(last_1h.get("rsi")) else 50
            macd_1h = float(last_1h.get("macd_hist", 0)) if pd.notna(last_1h.get("macd_hist")) else 0
            trend_1h_up = rsi_1h > 60 and macd_1h > 0
            trend_1h_down = rsi_1h < 40 and macd_1h < 0
            if (above_strike and trend_1h_up) or (not above_strike and trend_1h_down):
                adjustments["1h_aligned"] = 0.05
            elif (above_strike and trend_1h_down) or (not above_strike and trend_1h_up):
                adjustments["1h_opposes"] = -0.05

        # MACD momentum building (+3%)
        macd_hist = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        if len(df) >= 2:
            prev_hist = float(df.iloc[-2].get("macd_hist", 0)) if pd.notna(df.iloc[-2].get("macd_hist")) else 0
            if above_strike and macd_hist > prev_hist and macd_hist > 0:
                adjustments["macd_building"] = 0.03
            elif not above_strike and macd_hist < prev_hist and macd_hist < 0:
                adjustments["macd_building"] = 0.03

        # --- Negative adjustments (potential crossing) ---

        # RSI extreme (-8%)
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
        if above_strike and rsi > 75:
            adjustments["rsi_extreme"] = -0.08
        elif not above_strike and rsi < 25:
            adjustments["rsi_extreme"] = -0.08

        # RSI divergence (-8%)
        if len(df) >= 4:
            c1, c2, c3 = float(df.iloc[-3]["close"]), float(df.iloc[-2]["close"]), current_price
            r1 = float(df.iloc[-3].get("rsi", 50)) if pd.notna(df.iloc[-3].get("rsi")) else 50
            r2 = float(df.iloc[-2].get("rsi", 50)) if pd.notna(df.iloc[-2].get("rsi")) else 50
            r3 = rsi
            if above_strike and c3 > c2 > c1 and r3 < r2 < r1:
                adjustments["rsi_divergence"] = -0.08
            elif not above_strike and c3 < c2 < c1 and r3 > r2 > r1:
                adjustments["rsi_divergence"] = -0.08

        return adjustments

    def _decide_bet(self, adjusted_prob: float) -> tuple[str, int]:
        """Decide bet side and max price from adjusted probability."""
        if adjusted_prob >= 0.55:
            # Bet YES — price likely closes above strike
            fair_price = int(adjusted_prob * 100)
            max_price = min(MAX_BET_PRICE, fair_price - int(EDGE_MARGIN * 100))
            if max_price >= 5:  # minimum viable price
                return "YES", max_price
        elif adjusted_prob <= 0.45:
            # Bet NO — price likely closes below strike
            no_prob = 1.0 - adjusted_prob
            fair_price = int(no_prob * 100)
            max_price = min(MAX_BET_PRICE, fair_price - int(EDGE_MARGIN * 100))
            if max_price >= 5:
                return "NO", max_price

        return "SKIP", 0
```

- [ ] **Step 4: Run all V3 tests**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi_v3.py -v
```

- [ ] **Step 5: Run V1 tests to verify no regressions**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi.py -v
```

- [ ] **Step 6: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add strategy/strategies/kalshi_predictor_v3.py tests/test_kalshi_v3.py && git commit -m "feat: add KalshiPredictorV3 — strike-relative probability model

Answers 'will price close above/below strike?' using pre-computed
probability table + real-time technical adjustments. Bets both YES
and NO sides when edge > 5c margin.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `--predictor v3` to daemon with strike-aware eval

**Files:**
- Modify: `cli/live_daemon.py`
- Modify: `dashboard.py`
- Modify: `mcp_server.py`

- [ ] **Step 1: Add V3 to predictor selection in `__init__`**

Add V3 import branch:
```python
elif predictor_version == "v3":
    from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
    self.kalshi_predictor = KalshiPredictorV3()
```

- [ ] **Step 2: Add V3 path to `_kalshi_eval()` SETUP and CONFIRMED states**

For V3, the eval needs to:
1. Query Kalshi API for the active contract's `floor_strike` and `close_time`
2. Compute `minutes_remaining`
3. Call `predictor.predict(df, strike, minutes_remaining, market_data, df_1h)`
4. Use `signal.recommended_side` and `signal.max_price_cents`

Add a V3-specific scoring block inside the SETUP and CONFIRMED states:

```python
if self.kalshi_predictor_version == "v3":
    # V3: query Kalshi for strike price + compute time remaining
    self._init_kalshi_client()
    if self.kalshi_client:
        try:
            markets = self.kalshi_client.get_markets(series_ticker=series_ticker, status="open")
            if markets:
                market = markets[0]
                strike = market.get("floor_strike", 0)
                close_time_str = market.get("close_time", "")
                if close_time_str and strike:
                    from datetime import datetime
                    close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                    mins_left = max(0, (close_time - now_utc).total_seconds() / 60)
                    signal = self.kalshi_predictor.predict(
                        df_15m, strike_price=strike,
                        minutes_remaining=mins_left,
                        market_data=market_data, df_1h=df_1h
                    )
        except Exception as e:
            print(colored(f"  [V3] Kalshi API error: {e}", "yellow"))
else:
    # V1/V2: existing score() call
    signal = self.kalshi_predictor.score(df_15m, market_data=market_data, df_1h=df_1h)
```

- [ ] **Step 3: Update `_kalshi_execute_bet()` for V3 signals**

V3's signal has `recommended_side` and `max_price_cents` instead of direction + confidence threshold. Update the execution method to accept these when the signal is a `KalshiV3Signal`:

```python
# At the start of _kalshi_execute_bet, detect V3 signal type
from strategy.strategies.kalshi_predictor_v3 import KalshiV3Signal
if isinstance(signal, KalshiV3Signal):
    side = "yes" if signal.recommended_side == "YES" else "no"
    # V3 provides max_price directly — use it instead of the old pricing logic
    fill_price = min(signal.max_price_cents, MAX_ENTRY_CENTS)
    # ... continue with existing order placement using fill_price
```

- [ ] **Step 4: Update CLI/MCP to accept v3**

Add `"v3"` to the choices in argparse and MCP schema (already have v1/v2).

- [ ] **Step 5: Update dashboard for V3 signal display**

Show probability + distance instead of confidence:
```python
if hasattr(pred, 'probability'):
    # V3 display
    lines.append(f"    {asset:<5} {rec_side:<5} prob={prob:.0%} dist={dist_atr:+.1f}ATR ...")
```

- [ ] **Step 6: Verify**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -c "
from cli.live_daemon import LiveDaemon
d = LiveDaemon(dry_run=True, predictor_version='v3')
print(type(d.kalshi_predictor).__name__)
"
```
Expected: `KalshiPredictorV3`

- [ ] **Step 7: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py dashboard.py mcp_server.py && git commit -m "feat: add --predictor v3 with strike-aware Kalshi evaluation

V3 queries Kalshi API for strike price + time remaining, computes
probability, recommends YES/NO/SKIP with max price.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Build backtest for V3

**Files:**
- Modify: `backtest_kalshi.py`

- [ ] **Step 1: Add V3 walk-forward validation section**

Add to `main()` after the V2 section:

```python
# ── V3 Strike-Relative Predictor ──
print("\n" + "=" * 60)
print("V3 STRIKE-RELATIVE PREDICTOR")
print("=" * 60)

from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3

# Check if probability table exists
prob_table_path = "data/store/kalshi_prob_table.json"
if not Path(prob_table_path).exists():
    print("  Probability table not found. Run: ./venv/bin/python scripts/build_prob_table.py")
else:
    predictor_v3 = KalshiPredictorV3(prob_table_path=prob_table_path)

    v3_results = []
    for asset_name, df_15m in data_15m.items():
        df_ind = add_indicators(df_15m.copy())
        df_1h = data_1h.get(asset_name)
        df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None

        for i in range(50, len(df_ind) - 1):
            # Simulate: strike = current candle open, settlement = next candle close
            strike = float(df_ind.iloc[i]["open"])
            current_close = float(df_ind.iloc[i]["close"])
            next_close = float(df_ind.iloc[i + 1]["close"])
            actual_above = next_close >= strike

            # Simulate minutes remaining (use 10 as mid-window average)
            for mins_left in [10, 6]:
                signal = predictor_v3.predict(
                    df_ind.iloc[:i+1], strike_price=strike,
                    minutes_remaining=mins_left
                )
                if signal is None or signal.recommended_side == "SKIP":
                    continue

                correct = (signal.recommended_side == "YES" and actual_above) or \
                          (signal.recommended_side == "NO" and not actual_above)

                v3_results.append({
                    "asset": asset_name,
                    "side": signal.recommended_side,
                    "probability": signal.probability,
                    "distance_atr": signal.distance_atr,
                    "correct": correct,
                    "mins_left": mins_left,
                })
                break  # only first qualifying entry per window

    if v3_results:
        wins = sum(1 for r in v3_results if r["correct"])
        total = len(v3_results)
        wr = wins / total * 100
        yes_results = [r for r in v3_results if r["side"] == "YES"]
        no_results = [r for r in v3_results if r["side"] == "NO"]
        yes_wr = sum(1 for r in yes_results if r["correct"]) / len(yes_results) * 100 if yes_results else 0
        no_wr = sum(1 for r in no_results if r["correct"]) / len(no_results) * 100 if no_results else 0

        print(f"\n  Overall: {wr:.1f}% WR ({total} bets)")
        print(f"  YES bets: {yes_wr:.1f}% WR ({len(yes_results)} bets)")
        print(f"  NO bets:  {no_wr:.1f}% WR ({len(no_results)} bets)")

        # Calibration check
        print(f"\n  Calibration (predicted prob vs actual win rate):")
        for prob_min in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            prob_max = prob_min + 0.05
            bucket = [r for r in v3_results if prob_min <= r["probability"] < prob_max]
            bucket_no = [r for r in v3_results if prob_min <= (1 - r["probability"]) < prob_max and r["side"] == "NO"]
            both = bucket + bucket_no
            if both:
                actual_wr = sum(1 for r in both if r["correct"]) / len(both) * 100
                print(f"    Predicted {prob_min:.0%}-{prob_max:.0%}: actual {actual_wr:.1f}% ({len(both)} bets)")

        # Per-asset
        print(f"\n  Per-asset:")
        for asset in ASSETS:
            a_results = [r for r in v3_results if r["asset"] == asset]
            if a_results:
                a_wr = sum(1 for r in a_results if r["correct"]) / len(a_results) * 100
                print(f"    {asset}: {a_wr:.1f}% WR ({len(a_results)} bets)")
```

- [ ] **Step 2: Run backtest**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python backtest_kalshi.py --days 30
```

- [ ] **Step 3: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add backtest_kalshi.py && git commit -m "feat: add V3 walk-forward validation to backtest

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run backtests, validate, iterate

- [ ] **Step 1: Build probability table (90 days)**
- [ ] **Step 2: Run 30-day backtest with V3**
- [ ] **Step 3: Check calibration — predicted probabilities vs actual win rates**
- [ ] **Step 4: Tune adjustment values if needed**
- [ ] **Step 5: Run 90-day validation (walk-forward: first 60 days build table, last 30 validate)**
- [ ] **Step 6: Compare V3 vs V1 side-by-side**
- [ ] **Step 7: Commit any tuning changes**

---

### Task 6: Update CLAUDE.md

- [ ] **Step 1: Add V3 predictor documentation to Strategy 3 section**

Document: strike-relative model, probability table, how `--predictor v3` works, both YES and NO betting.

- [ ] **Step 2: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add CLAUDE.md && git commit -m "docs: add V3 strike-relative predictor to CLAUDE.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
