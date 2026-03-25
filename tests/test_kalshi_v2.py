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
