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
