# Kalshi V2 Continuation Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trend/continuation-based Kalshi predictor (v2) as a swappable alternative to the existing mean-reversion predictor (v1), targeting 65% WR on 15m crypto contracts.

**Architecture:** New `KalshiPredictorV2` class with same `score()` interface as v1. Two-layer scoring: 9 trend continuation components (0-100) minus 4 mean-reversion penalty components (0 to -45), plus existing leading indicators and MTF. Daemon selects predictor via `--predictor v1|v2` flag.

**Tech Stack:** Python 3, pandas, pandas_ta, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-kalshi-v2-continuation-predictor-design.md`

---

### Task 1: Add ADX to indicators module

**Files:**
- Modify: `data/indicators.py`
- Test: `tests/test_indicators.py`

- [ ] **Step 1: Write failing test**

Add to `TestIndicators` class in `tests/test_indicators.py`:

```python
def test_adds_adx(self, sample_ohlcv):
    df = add_indicators(sample_ohlcv)
    assert "adx" in df.columns
    valid = df["adx"].dropna()
    assert len(valid) > 0
    assert (valid >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_indicators.py::TestIndicators::test_adds_adx -v`

- [ ] **Step 3: Add ADX to `add_indicators()`**

In `data/indicators.py`, add after the `vol_sma_20` line:

```python
# ADX (Average Directional Index)
adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
out["adx"] = adx_df.filter(like="ADX").iloc[:, 0]
```

- [ ] **Step 4: Run all indicator tests**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_indicators.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add data/indicators.py tests/test_indicators.py && git commit -m "feat: add ADX indicator for V2 continuation predictor

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create `KalshiPredictorV2` with trend continuation scoring

**Files:**
- Create: `strategy/strategies/kalshi_predictor_v2.py`
- Test: `tests/test_kalshi_v2.py`

- [ ] **Step 1: Create test file with helper and initial tests**

Create `tests/test_kalshi_v2.py`:

```python
# tests/test_kalshi_v2.py
import pytest
import pandas as pd
import numpy as np
from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
from strategy.strategies.kalshi_predictor import KalshiSignal


def _make_df(n=50, close=100.0, open_price=None, rsi=50.0, stochrsi_k=50.0,
             bb_lower=98.0, bb_middle=100.0, bb_upper=102.0,
             macd_hist=0.0, volume=1000.0, vol_sma_20=1000.0,
             atr=2.0, roc_5=0.0, adx=25.0,
             ema_12=100.0, sma_20=100.0,
             close_trend=None, macd_hist_trend=None, volume_trend=None):
    """Build a synthetic DataFrame for V2 predictor testing."""
    if open_price is None:
        open_price = close
    closes = np.full(n, close)
    opens = np.full(n, open_price)
    macd_hists = np.full(n, macd_hist)
    volumes = np.full(n, volume)

    if close_trend is not None:
        for i, v in enumerate(close_trend):
            closes[n - len(close_trend) + i] = v
            opens[n - len(close_trend) + i] = v - 0.5  # slight green candle
    if macd_hist_trend is not None:
        for i, v in enumerate(macd_hist_trend):
            macd_hists[n - len(macd_hist_trend) + i] = v
    if volume_trend is not None:
        for i, v in enumerate(volume_trend):
            volumes[n - len(volume_trend) + i] = v

    highs = closes + 1.0
    lows = closes - 1.0
    if close_trend is not None:
        for i, v in enumerate(close_trend):
            idx = n - len(close_trend) + i
            highs[idx] = v + 1.0
            lows[idx] = v - 1.0

    df = pd.DataFrame({
        "close": closes, "open": opens, "high": highs, "low": lows,
        "volume": volumes, "rsi": np.full(n, rsi),
        "stochrsi_k": np.full(n, stochrsi_k),
        "bb_lower": np.full(n, bb_lower), "bb_middle": np.full(n, bb_middle),
        "bb_upper": np.full(n, bb_upper),
        "macd_hist": macd_hists, "vol_sma_20": np.full(n, vol_sma_20),
        "atr": np.full(n, atr), "roc_5": np.full(n, roc_5),
        "adx": np.full(n, adx),
        "ema_12": np.full(n, ema_12), "sma_20": np.full(n, sma_20),
    }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
    return df


class TestV2TrendDirection:
    """Tests for trend direction components."""

    def test_price_above_both_mas_gives_up(self):
        """Price above EMA-12 and SMA-20 gives UP direction points."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=25.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["price_vs_ema"]["up"] == 10
        assert signal.components["price_vs_sma"]["up"] == 10

    def test_price_below_both_mas_gives_down(self):
        """Price below EMA-12 and SMA-20 gives DOWN direction points."""
        df = _make_df(close=95.0, ema_12=100.0, sma_20=101.0, adx=25.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.direction == "DOWN"

    def test_ema_above_sma_gives_up_bonus(self):
        """EMA-12 above SMA-20 gives +5 UP."""
        df = _make_df(close=105.0, ema_12=101.0, sma_20=99.0, adx=25.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["ema_vs_sma"]["up"] == 5


class TestV2TrendStrength:
    """Tests for trend strength components."""

    def test_strong_adx_gives_high_points(self):
        """ADX > 40 gives 15 points."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=45.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["adx"]["score"] == 15

    def test_weak_adx_gives_zero(self):
        """ADX < 20 gives 0 points (no trend)."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=15.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["adx"]["score"] == 0

    def test_macd_growing_gives_high_points(self):
        """MACD histogram growing for 2+ candles gives 15."""
        df = _make_df(
            close=105.0, ema_12=100.0, sma_20=99.0, adx=25.0,
            macd_hist_trend=[0.1, 0.3, 0.5, 0.8],
        )
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["macd_trend"]["score"] == 15


class TestV2TrendPersistence:
    """Tests for trend persistence components."""

    def test_consecutive_green_candles(self):
        """4 consecutive green candles gives 10 points."""
        df = _make_df(
            close=104.0, ema_12=100.0, sma_20=99.0, adx=25.0,
            close_trend=[100.0, 101.0, 102.0, 103.0, 104.0],
        )
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["consecutive"]["score"] >= 7  # at least 3+ candles

    def test_roc_aligned_gives_points(self):
        """ROC > 1.0% in trend direction gives 10."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=25.0, roc_5=1.5)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["roc_aligned"]["score"] == 10


class TestV2Penalties:
    """Tests for mean-reversion penalty layer."""

    def test_rsi_extreme_penalizes_up_trend(self):
        """RSI > 80 in UP trend gives -15 penalty."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=30.0, rsi=82.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["penalty_rsi"]["score"] == -15

    def test_stochrsi_extreme_penalizes(self):
        """StochRSI K > 95 gives -10 penalty."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=30.0, stochrsi_k=96.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["penalty_stochrsi"]["score"] == -10

    def test_no_penalty_when_not_extreme(self):
        """RSI at 60 in UP trend gives 0 penalty."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=30.0, rsi=60.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["penalty_rsi"]["score"] == 0


class TestV2Integration:
    """Integration tests."""

    def test_strong_uptrend_high_confidence(self):
        """Perfect UP setup: price above MAs, strong ADX, growing MACD, consecutive candles."""
        df = _make_df(
            close=108.0, ema_12=104.0, sma_20=102.0, adx=35.0,
            roc_5=1.5, rsi=65.0, stochrsi_k=70.0,
            close_trend=[100.0, 102.0, 104.0, 106.0, 108.0],
            macd_hist_trend=[0.1, 0.3, 0.5, 0.8, 1.2],
            volume_trend=[900, 1000, 1100, 1200, 1400],
        )
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.confidence >= 50

    def test_no_trend_returns_low_confidence(self):
        """No trend (ADX < 20, price at MAs) returns low or no confidence."""
        df = _make_df(close=100.0, ema_12=100.0, sma_20=100.0, adx=12.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        if signal is not None:
            assert signal.confidence < 15

    def test_overextended_trend_gets_penalized(self):
        """Strong UP trend with RSI=85 gets penalty, lower confidence."""
        df_no_penalty = _make_df(
            close=108.0, ema_12=104.0, sma_20=102.0, adx=35.0,
            roc_5=1.5, rsi=65.0, stochrsi_k=70.0,
        )
        df_with_penalty = _make_df(
            close=108.0, ema_12=104.0, sma_20=102.0, adx=35.0,
            roc_5=1.5, rsi=85.0, stochrsi_k=96.0,
        )
        predictor = KalshiPredictorV2()
        sig1 = predictor.score(df_no_penalty)
        sig2 = predictor.score(df_with_penalty)
        assert sig1 is not None and sig2 is not None
        assert sig2.confidence < sig1.confidence

    def test_insufficient_data_returns_none(self):
        """Fewer than 20 candles returns None."""
        df = _make_df(n=10)
        predictor = KalshiPredictorV2()
        assert predictor.score(df) is None

    def test_confidence_capped_at_100(self):
        """Confidence never exceeds 100."""
        df = _make_df(
            close=120.0, ema_12=105.0, sma_20=100.0, adx=50.0,
            roc_5=3.0, rsi=65.0,
            close_trend=[100.0, 105.0, 110.0, 115.0, 120.0],
            macd_hist_trend=[0.5, 1.0, 2.0, 3.0, 4.0],
            volume_trend=[1000, 1500, 2000, 2500, 3000],
        )
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.confidence <= 100

    def test_same_interface_as_v1(self):
        """V2 returns KalshiSignal with same fields as V1."""
        df = _make_df(close=105.0, ema_12=100.0, sma_20=99.0, adx=30.0)
        predictor = KalshiPredictorV2()
        signal = predictor.score(df)
        assert signal is not None
        assert hasattr(signal, 'asset')
        assert hasattr(signal, 'direction')
        assert hasattr(signal, 'confidence')
        assert hasattr(signal, 'components')
        assert hasattr(signal, 'price')
        assert hasattr(signal, 'rsi')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi_v2.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `KalshiPredictorV2`**

Create `strategy/strategies/kalshi_predictor_v2.py`. The full implementation follows the spec's two-layer model:

```python
# strategy/strategies/kalshi_predictor_v2.py
"""V2 Kalshi predictor: trend/continuation primary, mean-reversion penalties.

Scores confidence for 15-minute crypto direction predictions using:
- Layer 1: 9 trend continuation components (0-100 points)
- Layer 2: 4 mean-reversion penalty components (0 to -45 points)
- Leading indicators (same as V1): order book, trade flow, etc.
- 1-hour trend alignment (same as V1)

Designed to capture trend continuation setups that V1 (mean-reversion) misses.
"""
import pandas as pd
import numpy as np
from strategy.strategies.kalshi_predictor import KalshiSignal

# Trend layer max
_MAX_TREND = 100
# Leading indicators max (same as V1)
_MAX_LEADING = 65
# MTF max
_MAX_MTF = 15


class KalshiPredictorV2:
    """Trend/continuation scorer for Kalshi 15m predictions."""

    def score(self, df: pd.DataFrame, market_data: dict | None = None,
              df_1h: pd.DataFrame | None = None) -> KalshiSignal | None:
        if df is None or len(df) < 20:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        close = float(last["close"])
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50

        up_score = 0
        down_score = 0
        trend_up = 0  # for filter tracking
        trend_down = 0
        leading_up = 0
        leading_down = 0
        components = {}

        # ═══════════════════════════════════════════════════════
        # LAYER 1: TREND CONTINUATION (9 components, max 100)
        # ═══════════════════════════════════════════════════════

        # 1. Price vs EMA-12 (0-10)
        ema_12 = float(last.get("ema_12", close)) if pd.notna(last.get("ema_12")) else close
        pve_up, pve_down = 0, 0
        pct_from_ema = abs(close - ema_12) / ema_12 * 100 if ema_12 > 0 else 0
        if pct_from_ema > 0.1:
            if close > ema_12:
                pve_up = 10
            else:
                pve_down = 10
        up_score += pve_up; down_score += pve_down
        trend_up += pve_up; trend_down += pve_down
        components["price_vs_ema"] = {"up": pve_up, "down": pve_down}

        # 2. Price vs SMA-20 (0-10)
        sma_20 = float(last.get("sma_20", close)) if pd.notna(last.get("sma_20")) else close
        pvs_up, pvs_down = 0, 0
        pct_from_sma = abs(close - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
        if pct_from_sma > 0.1:
            if close > sma_20:
                pvs_up = 10
            else:
                pvs_down = 10
        up_score += pvs_up; down_score += pvs_down
        trend_up += pvs_up; trend_down += pvs_down
        components["price_vs_sma"] = {"up": pvs_up, "down": pvs_down}

        # 3. EMA-12 vs SMA-20 (0-5)
        evs_up, evs_down = 0, 0
        if ema_12 > sma_20:
            evs_up = 5
        elif ema_12 < sma_20:
            evs_down = 5
        up_score += evs_up; down_score += evs_down
        trend_up += evs_up; trend_down += evs_down
        components["ema_vs_sma"] = {"up": evs_up, "down": evs_down}

        # 4. ADX (0-15)
        adx_val = float(last.get("adx", 0)) if pd.notna(last.get("adx")) else 0
        adx_score = 0
        if adx_val > 40:
            adx_score = 15
        elif adx_val > 30:
            adx_score = 10
        elif adx_val > 20:
            adx_score = 5
        # ADX confirms the dominant direction
        if up_score > down_score:
            up_score += adx_score
            trend_up += adx_score
        elif down_score > up_score:
            down_score += adx_score
            trend_down += adx_score
        components["adx"] = {"score": adx_score, "value": adx_val}

        # 5. MACD histogram trend (0-15)
        macd_score = 0
        macd_hist = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        prev_hist = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
        if len(df) >= 3:
            prev2_hist = float(df.iloc[-3].get("macd_hist", 0)) if pd.notna(df.iloc[-3].get("macd_hist")) else 0
            # Growing for 2+ candles
            if macd_hist > prev_hist > prev2_hist and macd_hist > 0:
                macd_score = 15  # growing UP
            elif macd_hist < prev_hist < prev2_hist and macd_hist < 0:
                macd_score = 15  # growing DOWN
            elif (macd_hist > prev_hist and macd_hist > 0) or (macd_hist < prev_hist and macd_hist < 0):
                macd_score = 8   # growing 1 candle
            elif (macd_hist > 0 and up_score > down_score) or (macd_hist < 0 and down_score > up_score):
                macd_score = 3   # positive but not growing
        if up_score > down_score:
            up_score += macd_score
            trend_up += macd_score
        else:
            down_score += macd_score
            trend_down += macd_score
        components["macd_trend"] = {"score": macd_score, "value": macd_hist}

        # 6. Consecutive candles (0-10)
        consec_score = 0
        if len(df) >= 5:
            up_count = 0
            down_count = 0
            for j in range(-1, -6, -1):
                c = float(df.iloc[j]["close"])
                o = float(df.iloc[j]["open"])
                if c >= o:
                    up_count += 1
                    if down_count > 0:
                        break
                else:
                    down_count += 1
                    if up_count > 0:
                        break
            count = max(up_count, down_count)
            if count >= 4:
                consec_score = 10
            elif count >= 3:
                consec_score = 7
            elif count >= 2:
                consec_score = 3
            if up_count > down_count:
                up_score += consec_score
                trend_up += consec_score
            else:
                down_score += consec_score
                trend_down += consec_score
        components["consecutive"] = {"score": consec_score}

        # 7. Higher highs / higher lows (0-10)
        hh_score = 0
        if len(df) >= 4:
            hh_count = 0
            hl_count = 0
            ll_count = 0
            lh_count = 0
            for j in range(-1, -4, -1):
                if float(df.iloc[j]["high"]) > float(df.iloc[j-1]["high"]):
                    hh_count += 1
                if float(df.iloc[j]["low"]) > float(df.iloc[j-1]["low"]):
                    hl_count += 1
                if float(df.iloc[j]["low"]) < float(df.iloc[j-1]["low"]):
                    ll_count += 1
                if float(df.iloc[j]["high"]) < float(df.iloc[j-1]["high"]):
                    lh_count += 1
            if hh_count >= 3 and hl_count >= 3:
                hh_score = 10
                up_score += hh_score; trend_up += hh_score
            elif hh_count >= 2 and hl_count >= 2:
                hh_score = 5
                up_score += hh_score; trend_up += hh_score
            elif ll_count >= 3 and lh_count >= 3:
                hh_score = 10
                down_score += hh_score; trend_down += hh_score
            elif ll_count >= 2 and lh_count >= 2:
                hh_score = 5
                down_score += hh_score; trend_down += hh_score
        components["hh_hl"] = {"score": hh_score}

        # 8. ROC-5 aligned (0-10)
        roc_score = 0
        roc_val = float(last.get("roc_5", 0)) if pd.notna(last.get("roc_5")) else 0
        if roc_val > 1.0:
            roc_score = 10
            up_score += roc_score; trend_up += roc_score
        elif roc_val > 0.3:
            roc_score = 5
            up_score += roc_score; trend_up += roc_score
        elif roc_val < -1.0:
            roc_score = 10
            down_score += roc_score; trend_down += roc_score
        elif roc_val < -0.3:
            roc_score = 5
            down_score += roc_score; trend_down += roc_score
        components["roc_aligned"] = {"score": roc_score, "value": roc_val}

        # 9. Volume confirmation (0-15: trend 0-8 + vs avg 0-7)
        vol_trend_score = 0
        vol_avg_score = 0
        vol = float(last.get("volume", 0))
        vol_sma = float(last.get("vol_sma_20", 0)) if pd.notna(last.get("vol_sma_20")) else 0
        if len(df) >= 3:
            v1 = float(df.iloc[-3]["volume"])
            v2 = float(df.iloc[-2]["volume"])
            v3 = vol
            if v3 > v2 > v1:
                vol_trend_score = 8
            elif abs(v3 - v2) / max(v2, 1) < 0.1:
                vol_trend_score = 3
        if vol_sma > 0:
            if vol > vol_sma * 2:
                vol_avg_score = 7
            elif vol > vol_sma * 1.5:
                vol_avg_score = 4
            elif vol > vol_sma:
                vol_avg_score = 2
        vol_total = vol_trend_score + vol_avg_score
        if up_score > down_score:
            up_score += vol_total; trend_up += vol_total
        else:
            down_score += vol_total; trend_down += vol_total
        components["volume"] = {"trend": vol_trend_score, "vs_avg": vol_avg_score, "total": vol_total}

        # ═══════════════════════════════════════════════════════
        # LAYER 2: MEAN-REVERSION PENALTIES (max -45)
        # ═══════════════════════════════════════════════════════
        dominant_is_up = up_score > down_score

        # Penalty 1: RSI extreme (-15)
        rsi_penalty = 0
        if dominant_is_up and rsi > 80:
            rsi_penalty = -15
        elif dominant_is_up and rsi > 75:
            rsi_penalty = -8
        elif not dominant_is_up and rsi < 20:
            rsi_penalty = -15
        elif not dominant_is_up and rsi < 25:
            rsi_penalty = -8
        if dominant_is_up:
            up_score += rsi_penalty
        else:
            down_score += rsi_penalty
        components["penalty_rsi"] = {"score": rsi_penalty, "value": rsi}

        # Penalty 2: StochRSI extreme (-10)
        stoch_penalty = 0
        stochrsi_k = float(last.get("stochrsi_k", 50)) if pd.notna(last.get("stochrsi_k")) else 50
        if dominant_is_up and stochrsi_k > 95:
            stoch_penalty = -10
        elif dominant_is_up and stochrsi_k > 90:
            stoch_penalty = -5
        elif not dominant_is_up and stochrsi_k < 5:
            stoch_penalty = -10
        elif not dominant_is_up and stochrsi_k < 10:
            stoch_penalty = -5
        if dominant_is_up:
            up_score += stoch_penalty
        else:
            down_score += stoch_penalty
        components["penalty_stochrsi"] = {"score": stoch_penalty, "value": stochrsi_k}

        # Penalty 3: BB overextension (-10)
        bb_penalty = 0
        bb_upper = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0
        bb_lower = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
        if bb_upper > 0 and dominant_is_up:
            if close > bb_upper * 1.01:
                bb_penalty = -10
            elif close > bb_upper:
                bb_penalty = -5
        if bb_lower > 0 and not dominant_is_up:
            if close < bb_lower * 0.99:
                bb_penalty = -10
            elif close < bb_lower:
                bb_penalty = -5
        if dominant_is_up:
            up_score += bb_penalty
        else:
            down_score += bb_penalty
        components["penalty_bb"] = {"score": bb_penalty}

        # Penalty 4: RSI divergence (-10)
        div_penalty = 0
        if len(df) >= 4:
            c1, c2, c3 = float(df.iloc[-3]["close"]), float(df.iloc[-2]["close"]), close
            r1 = float(df.iloc[-3].get("rsi", 50)) if pd.notna(df.iloc[-3].get("rsi")) else 50
            r2 = float(df.iloc[-2].get("rsi", 50)) if pd.notna(df.iloc[-2].get("rsi")) else 50
            r3 = rsi
            if dominant_is_up:
                if c3 > c2 > c1 and r3 < r2 < r1:
                    div_penalty = -10
            else:
                if c3 < c2 < c1 and r3 > r2 > r1:
                    div_penalty = -10
        if dominant_is_up:
            up_score += div_penalty
        else:
            down_score += div_penalty
        components["penalty_divergence"] = {"score": div_penalty}

        # ═══════════════════════════════════════════════════════
        # LEADING INDICATORS (same as V1, components 9-13)
        # ═══════════════════════════════════════════════════════
        has_leading = market_data is not None
        ob = (market_data or {}).get("order_book", {})
        tf = (market_data or {}).get("trade_flow", {})
        cross = (market_data or {}).get("cross_asset", {})

        # Order Book Imbalance (0-20)
        ob_up, ob_down = 0, 0
        imbalance = ob.get("imbalance", 0)
        if imbalance > 0.3: ob_up = 20
        elif imbalance > 0.15: ob_up = 10
        elif imbalance < -0.3: ob_down = 20
        elif imbalance < -0.15: ob_down = 10
        up_score += ob_up; down_score += ob_down
        leading_up += ob_up; leading_down += ob_down
        components["order_book"] = {"up": ob_up, "down": ob_down, "imbalance": imbalance}

        # Trade Flow (0-20)
        tf_up, tf_down = 0, 0
        net_flow = tf.get("net_flow", 0)
        buy_ratio = tf.get("buy_ratio", 0.5)
        if net_flow > 0.2 and buy_ratio > 0.55: tf_up = 20
        elif net_flow > 0.1: tf_up = 10
        elif net_flow < -0.2 and buy_ratio < 0.45: tf_down = 20
        elif net_flow < -0.1: tf_down = 10
        up_score += tf_up; down_score += tf_down
        leading_up += tf_up; leading_down += tf_down
        components["trade_flow"] = {"up": tf_up, "down": tf_down}

        # Large Trade Bias (0-10)
        lt_up, lt_down = 0, 0
        large_bias = tf.get("large_trade_bias", 0)
        if large_bias > 0.3: lt_up = 10
        elif large_bias < -0.3: lt_down = 10
        up_score += lt_up; down_score += lt_down
        leading_up += lt_up; leading_down += lt_down
        components["large_trade"] = {"up": lt_up, "down": lt_down}

        # Spread (0-5)
        spread_score = 0
        spread_pct = ob.get("spread_pct", 0)
        if spread_pct > 0.1: spread_score = 5
        if up_score > down_score:
            up_score += spread_score; leading_up += spread_score
        else:
            down_score += spread_score; leading_down += spread_score
        components["spread"] = {"score": spread_score}

        # Cross-Asset (0-10)
        ca_up, ca_down = 0, 0
        btc_dir = cross.get("market_direction", 0)
        if btc_dir < -1: ca_down = 10
        elif btc_dir > 1: ca_up = 10
        up_score += ca_up; down_score += ca_down
        leading_up += ca_up; leading_down += ca_down
        components["cross_asset"] = {"up": ca_up, "down": ca_down}

        # ═══════════════════════════════════════════════════════
        # MTF: 1-Hour Trend Alignment (-15 to +15)
        # ═══════════════════════════════════════════════════════
        mtf_score = 0
        if df_1h is not None and len(df_1h) >= 20:
            last_1h = df_1h.iloc[-1]
            rsi_1h = float(last_1h.get("rsi", 50)) if pd.notna(last_1h.get("rsi")) else 50
            macd_1h = float(last_1h.get("macd_hist", 0)) if pd.notna(last_1h.get("macd_hist")) else 0
            trend_1h_up = rsi_1h > 60 and macd_1h > 0
            trend_1h_down = rsi_1h < 40 and macd_1h < 0
            dom_up = up_score > down_score
            if trend_1h_up and dom_up: mtf_score = 15
            elif trend_1h_down and not dom_up: mtf_score = 15
            elif trend_1h_up and not dom_up: mtf_score = -15
            elif trend_1h_down and dom_up: mtf_score = -15
            if mtf_score > 0:
                if dom_up: up_score += mtf_score
                else: down_score += mtf_score
            elif mtf_score < 0:
                if dom_up: up_score += mtf_score
                else: down_score += mtf_score
        components["mtf"] = {"score": mtf_score}

        # ═══════════════════════════════════════════════════════
        # FILTERS (same as V1)
        # ═══════════════════════════════════════════════════════
        if self._apply_filters(up_score, down_score, trend_up, trend_down,
                               leading_up, leading_down, df, components):
            return None

        # ═══════════════════════════════════════════════════════
        # NORMALIZE AND RETURN
        # ═══════════════════════════════════════════════════════
        if has_leading:
            max_possible = _MAX_TREND + _MAX_LEADING + _MAX_MTF
        elif df_1h is not None:
            max_possible = _MAX_TREND + _MAX_MTF
        else:
            max_possible = _MAX_TREND

        if up_score > down_score and up_score > 0:
            confidence = min(100, max(0, int(up_score * 100 / max_possible)))
            return KalshiSignal(asset="", direction="UP", confidence=confidence,
                                components=components, price=close, rsi=rsi)
        elif down_score > up_score and down_score > 0:
            confidence = min(100, max(0, int(down_score * 100 / max_possible)))
            return KalshiSignal(asset="", direction="DOWN", confidence=confidence,
                                components=components, price=close, rsi=rsi)
        return None

    def check_1m_momentum(self, df_1m: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
        """Same as V1 — 2 of 3 candles in predicted direction."""
        if df_1m is None or len(df_1m) < lookback:
            return False
        recent = df_1m.iloc[-lookback:]
        if direction == "UP":
            confirming = (recent["close"] > recent["open"]).sum()
        else:
            confirming = (recent["close"] < recent["open"]).sum()
        return bool(confirming >= 2)

    def compute_5m_booster(self, df_5m: pd.DataFrame, direction: str,
                            window_open_price: float) -> int:
        """Same as V1 — available for future use."""
        if df_5m is None or len(df_5m) < 2:
            return 0
        last = df_5m.iloc[-1]
        prev = df_5m.iloc[-2]
        booster = 0
        is_up = direction == "UP"
        close = float(last["close"])
        open_price = float(last["open"])
        prev_close = float(prev["close"])
        candle_is_green = close > open_price
        if (is_up and candle_is_green) or (not is_up and not candle_is_green):
            booster += 3
        vol = float(last.get("volume", 0))
        vol_sma = float(last.get("vol_sma_20", 0)) if pd.notna(last.get("vol_sma_20")) else 0
        if vol_sma > 0 and vol > vol_sma * 1.5:
            booster += 3
        atr = float(last.get("atr", 0)) if pd.notna(last.get("atr")) else 0
        if atr > 0 and window_open_price > 0:
            distance = close - window_open_price
            if (is_up and distance > 0.5 * atr) or (not is_up and distance < -0.5 * atr):
                booster += 3
        macd_now = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else 0
        macd_prev = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
        crossed = (macd_now > 0 and macd_prev <= 0) or (macd_now < 0 and macd_prev >= 0)
        if crossed:
            aligned = (is_up and macd_now > 0) or (not is_up and macd_now < 0)
            if aligned: booster += 3
            else: booster -= 5
        rsi_now = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
        rsi_prev = float(prev.get("rsi", 50)) if pd.notna(prev.get("rsi")) else 50
        if is_up and close > prev_close and rsi_now < rsi_prev:
            booster -= 5
        elif not is_up and close < prev_close and rsi_now > rsi_prev:
            booster -= 5
        return max(-10, min(12, booster))

    def _apply_filters(self, up_score, down_score, trend_up, trend_down,
                       leading_up, leading_down, df, components):
        """Same filter logic as V1."""
        # Filter 1: Directional conflict
        trend_dir = "UP" if trend_up > trend_down else "DOWN"
        lead_dir = "UP" if leading_up > leading_down else "DOWN"
        trend_strength = max(trend_up, trend_down)
        lead_strength = max(leading_up, leading_down)
        if trend_dir != lead_dir and trend_strength >= 15 and lead_strength >= 15:
            components["filter_conflict"] = True
            return True

        # Filter 2: Volatility regime
        if "atr" in df.columns and len(df) >= 200:
            atr_series = df["atr"].dropna().tail(200)
            if len(atr_series) >= 50:
                current_atr = float(atr_series.iloc[-1])
                percentile = (atr_series < current_atr).sum() / len(atr_series) * 100
                if percentile > 90:
                    components["filter_volatility"] = True
                    return True

        # Filter 3: Margin of victory
        winner = max(up_score, down_score)
        loser = min(up_score, down_score)
        if loser > 0 and winner < loser * 1.5:
            components["filter_margin"] = True
            return True

        return False
```

- [ ] **Step 4: Run all V2 tests**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi_v2.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run V1 tests to verify no regressions**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi.py -v`
Expected: ALL PASS (V1 untouched)

- [ ] **Step 6: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add strategy/strategies/kalshi_predictor_v2.py tests/test_kalshi_v2.py && git commit -m "feat: add KalshiPredictorV2 — trend continuation scorer with penalty layer

9 trend components (direction, strength, persistence, volume) + 4 mean-reversion
penalties (RSI extreme, StochRSI extreme, BB overextension, RSI divergence).
Same interface as V1 for swappable integration.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `--predictor` flag to daemon, dashboard, and MCP

**Files:**
- Modify: `cli/live_daemon.py`
- Modify: `dashboard.py`
- Modify: `mcp_server.py`

- [ ] **Step 1: Add predictor version to LiveDaemon**

In `cli/live_daemon.py`, update `__init__` signature:

```python
def __init__(self, dry_run: bool = False, kalshi_only: bool = False, predictor_version: str = "v1"):
```

Add predictor selection after the existing predictor init (around line 94):

```python
# Kalshi predictor + client
self.kalshi_predictor_version = predictor_version
if predictor_version == "v2":
    from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
    self.kalshi_predictor = KalshiPredictorV2()
else:
    from strategy.strategies.kalshi_predictor import KalshiPredictor
    self.kalshi_predictor = KalshiPredictor()
```

Update the startup banner to show which predictor:

```python
print(colored(f"  Predictor: {'V2 Continuation' if predictor_version == 'v2' else 'V1 Mean-Reversion'}", "cyan"))
```

- [ ] **Step 2: Add CLI args**

In `cli/live_daemon.py` `main()`, add:

```python
parser.add_argument("--predictor", choices=["v1", "v2"], default="v1",
                    help="Kalshi predictor version: v1 (mean-reversion) or v2 (continuation)")
```

Pass to LiveDaemon: `LiveDaemon(dry_run=..., kalshi_only=..., predictor_version=args.predictor)`

In `dashboard.py` `main()`, add the same arg and pass through:

```python
parser.add_argument("--predictor", choices=["v1", "v2"], default="v1",
                    help="Kalshi predictor: v1 (mean-reversion) or v2 (continuation)")
```

Update Dashboard init: `Dashboard(dry_run=..., max_cycles=..., kalshi_only=..., predictor_version=args.predictor)`

Update Dashboard class to accept and pass it:

```python
def __init__(self, ..., predictor_version: str = "v1"):
    self.daemon = LiveDaemon(dry_run=dry_run, kalshi_only=kalshi_only, predictor_version=predictor_version)
```

- [ ] **Step 3: Add MCP parameter**

In `mcp_server.py`, update `algotrade_start` tool schema to add:

```python
"predictor": {
    "type": "string",
    "enum": ["v1", "v2"],
    "description": "Kalshi predictor: v1 (mean-reversion) or v2 (continuation)",
    "default": "v1",
},
```

Update `handle_start()` and the dispatcher to pass `predictor` through.

- [ ] **Step 4: Verify**

```bash
cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -c "from cli.live_daemon import LiveDaemon; d = LiveDaemon(dry_run=True, predictor_version='v2'); print(type(d.kalshi_predictor).__name__)"
```
Expected: `KalshiPredictorV2`

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py dashboard.py mcp_server.py && git commit -m "feat: add --predictor v1|v2 flag for swappable Kalshi predictor

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add V2 comparison to backtest

**Files:**
- Modify: `backtest_kalshi.py`

- [ ] **Step 1: Add V2 comparison section to `main()`**

After the existing iterations and summary, add a V2 comparison section:

```python
# ── V2 Continuation Predictor comparison ──
print("\n" + "=" * 60)
print("V2 CONTINUATION PREDICTOR (vs V1 baseline)")
print("=" * 60)

from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
predictor_v2 = KalshiPredictorV2()

results_v2 = []
for asset_name, df_15m in data_15m.items():
    df_1h = data_1h.get(asset_name)
    r = run_predictor_on_candles(asset_name, df_15m, df_1h, predictor_v2,
                                 use_filters=True, use_mtf=True)
    results_v2.extend(r)

# Per-asset breakdown with V1 thresholds
PER_ASSET_THRESH = {"BTC": 30, "ETH": 35, "SOL": 35, "XRP": 30, "BNB": 35}
for asset_name in ASSETS:
    r_asset = [r for r in results_v2 if r["asset"] == asset_name]
    thresh = PER_ASSET_THRESH.get(asset_name, 30)
    filtered = [r for r in r_asset if r["confidence"] >= thresh]
    if filtered:
        wins = sum(1 for r in filtered if r["correct"])
        wr = wins / len(filtered) * 100
        print(f"  {asset_name}: {wr:.1f}% WR ({len(filtered)} bets) @ threshold {thresh}")

# V2 threshold sweep
print(f"\n  V2 THRESHOLD SWEEP:")
print(f"  {'Thresh':>7} {'Bets':>6} {'Wins':>6} {'WR%':>7} {'PF':>6}")
print(f"  {'─' * 35}")
for t in [25, 30, 35, 40, 45, 50]:
    m = simulate_pnl(results_v2, t)
    if m['total_bets'] > 0:
        print(f"  {t:>7} {m['total_bets']:>6} {m['wins']:>6} {m['win_rate']:>6.1f}% {m['profit_factor']:>5.2f}")

# V2 without penalties (to measure penalty effectiveness)
print(f"\n  V2 WITHOUT PENALTIES:")
predictor_v2_no_pen = KalshiPredictorV2()
# Monkey-patch to skip penalties (set all penalty methods to return 0)
# ... implementer should run V2 with a flag or separate pass
```

- [ ] **Step 2: Run 30-day backtest**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python backtest_kalshi.py --days 30`

- [ ] **Step 3: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add backtest_kalshi.py && git commit -m "feat: add V2 continuation predictor comparison to backtest

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Run backtest and iterate

- [ ] **Step 1: Run 30-day V2 backtest**
- [ ] **Step 2: Run 90-day V2 backtest**
- [ ] **Step 3: Per-asset threshold sweep**
- [ ] **Step 4: Compare V2 vs V1 side-by-side**
- [ ] **Step 5: Tune component weights if WR < 65%**
- [ ] **Step 6: Commit adjustments**

---

### Task 6: Update CLAUDE.md

- [ ] **Step 1: Add V2 predictor documentation**

Add a new section after Strategy 3 describing V2: its components, when to use `--predictor v2`, the continuation vs reversion difference.

- [ ] **Step 2: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add CLAUDE.md && git commit -m "docs: add V2 continuation predictor to CLAUDE.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
