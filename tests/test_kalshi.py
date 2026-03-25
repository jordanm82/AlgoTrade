# tests/test_kalshi.py
import pytest
from unittest.mock import patch, MagicMock
from exchange.kalshi import KalshiClient
from strategy.strategies.kalshi_predictor import KalshiPredictor, KalshiSignal
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# KalshiClient tests (unchanged)
# ---------------------------------------------------------------------------

class TestKalshiClient:
    def test_headers_include_required_fields(self):
        with patch("exchange.kalshi.serialization.load_pem_private_key") as mock_load:
            mock_key = MagicMock()
            mock_key.sign.return_value = b"fake_signature"
            mock_load.return_value = mock_key

            client = KalshiClient.__new__(KalshiClient)
            client.api_key_id = "test-key-id"
            client.base_url = "https://demo-trading-api.kalshi.com"
            client._private_key = mock_key

            headers = client._headers("GET", "/trade-api/v2/markets")
            assert "KALSHI-ACCESS-KEY" in headers
            assert "KALSHI-ACCESS-SIGNATURE" in headers
            assert "KALSHI-ACCESS-TIMESTAMP" in headers
            assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"

    def test_bet_btc_up_calls_correct_side(self):
        client = KalshiClient.__new__(KalshiClient)
        client.api_key_id = "test"
        client.base_url = "https://demo"
        client._private_key = MagicMock()
        client._private_key.sign.return_value = b"sig"

        with patch.object(client, "_post", return_value={"order": {"order_id": "123"}}) as mock_post:
            client.bet_btc_up("KXBTC-TEST", 5, price_cents=65)
            call_data = mock_post.call_args[0][1]
            assert call_data["side"] == "yes"
            assert call_data["type"] == "limit"
            assert call_data["yes_price"] == 65

    def test_bet_btc_down_calls_correct_side(self):
        client = KalshiClient.__new__(KalshiClient)
        client.api_key_id = "test"
        client.base_url = "https://demo"
        client._private_key = MagicMock()
        client._private_key.sign.return_value = b"sig"

        with patch.object(client, "_post", return_value={"order": {"order_id": "456"}}) as mock_post:
            client.bet_btc_down("KXBTC-TEST", 3, price_cents=40)
            call_data = mock_post.call_args[0][1]
            assert call_data["side"] == "no"
            assert call_data["no_price"] == 40


# ---------------------------------------------------------------------------
# KalshiPredictor tests
# ---------------------------------------------------------------------------

def _make_df(n=50, close=87000.0, rsi=50.0, bb_lower=86500.0, bb_middle=87000.0,
             bb_upper=87500.0, macd_hist=0.0, volume=1000.0, vol_sma_20=1000.0,
             close_trend=None, rsi_trend=None, macd_hist_trend=None,
             stochrsi_k=50.0, roc_5=0.0, atr=200.0):
    """Build a synthetic DataFrame for predictor testing.

    close_trend: list of last N close values (overrides tail of close column)
    rsi_trend:   list of last N RSI values (overrides tail of rsi column)
    macd_hist_trend: list of last N MACD histogram values
    """
    closes = np.full(n, close)
    rsis = np.full(n, rsi)
    macd_hists = np.full(n, macd_hist)
    volumes = np.full(n, volume)

    if close_trend is not None:
        for i, v in enumerate(close_trend):
            closes[n - len(close_trend) + i] = v
    if rsi_trend is not None:
        for i, v in enumerate(rsi_trend):
            rsis[n - len(rsi_trend) + i] = v
    if macd_hist_trend is not None:
        for i, v in enumerate(macd_hist_trend):
            macd_hists[n - len(macd_hist_trend) + i] = v

    df = pd.DataFrame({
        "close": closes,
        "high": closes + 100,
        "low": closes - 100,
        "open": closes,
        "volume": volumes,
        "rsi": rsis,
        "bb_lower": np.full(n, bb_lower),
        "bb_middle": np.full(n, bb_middle),
        "bb_upper": np.full(n, bb_upper),
        "macd_hist": macd_hists,
        "vol_sma_20": np.full(n, vol_sma_20),
        "stochrsi_k": np.full(n, stochrsi_k),
        "roc_5": np.full(n, roc_5),
        "atr": np.full(n, atr),
    }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
    return df


class TestKalshiPredictor:
    """Tests for the multi-signal confidence scorer."""

    def test_oversold_rsi_below_bb_gives_high_up_confidence(self):
        """RSI < 25 (30 pts) + price below BB lower (20 pts) = strong UP."""
        df = _make_df(rsi=22.0, close=86400.0, bb_lower=86500.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "UP"
        # RSI < 25 = 30pts, price < bb_lower = 20pts; normalized: 50*100/110 = 45
        assert signal.confidence >= 40
        assert signal.components["rsi"]["up"] == 30
        assert signal.components["bb"]["up"] == 20

    def test_overbought_rsi_above_bb_gives_high_down_confidence(self):
        """RSI > 75 (30 pts) + price above BB upper (20 pts) = strong DOWN."""
        df = _make_df(rsi=78.0, close=87600.0, bb_upper=87500.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "DOWN"
        # RSI > 75 = 30pts, price > bb_upper = 20pts; normalized: 50*100/110 = 45
        assert signal.confidence >= 40
        assert signal.components["rsi"]["down"] == 30
        assert signal.components["bb"]["down"] == 20

    def test_neutral_rsi_gives_low_or_no_confidence(self):
        """RSI at 50 should generate zero RSI points."""
        df = _make_df(rsi=50.0, close=87000.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        # With neutral RSI and price in middle of BB, might be None or very low
        if signal is not None:
            assert signal.confidence < 20
            assert signal.components["rsi"]["up"] == 0
            assert signal.components["rsi"]["down"] == 0

    def test_all_components_contribute(self):
        """Every signal component fires for a perfect UP setup."""
        # RSI < 25 = 30, BB below = 20, MACD hist positive+increasing = 15,
        # Volume 2x = 10, ROC > 1.5% = 10, RSI trend recovering = 10, StochRSI < 10 = 15
        # ATR move > 1.5x in UP direction = 10
        df = _make_df(
            rsi=22.0,
            close=86400.0,
            bb_lower=86500.0,
            macd_hist=5.0,
            volume=2500.0,
            vol_sma_20=1000.0,
            roc_5=2.0,
            stochrsi_k=8.0,
            atr=200.0,
            # MACD hist increasing
            macd_hist_trend=[2.0, 3.0, 4.0, 5.0],
            # RSI recovering from oversold
            rsi_trend=[18.0, 19.0, 20.0, 22.0],
        )
        # Big UP candle: move = 350, ratio = 1.75 > 1.5 → +10 ATR
        df.iloc[-1, df.columns.get_loc("open")] = df.iloc[-1]["close"] - 350
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "UP"
        # Check each component contributed
        assert signal.components["rsi"]["up"] == 30
        assert signal.components["bb"]["up"] == 20
        assert signal.components["macd"]["up"] == 15
        assert signal.components["volume"]["score"] == 10
        assert signal.components["roc"]["up"] == 10
        assert signal.components["rsi_trend"]["up"] == 10
        assert signal.components["stochrsi"]["up"] == 15
        assert signal.components["atr_move"]["score"] == 10
        # Total = 30 + 20 + 15 + 10 + 10 + 10 + 15 + 10 = 120
        assert signal.confidence == 100

    def test_confidence_capped_at_100(self):
        """Even with extreme signals, confidence does not exceed 100."""
        df = _make_df(
            rsi=15.0,
            close=86000.0,
            bb_lower=86500.0,
            macd_hist=10.0,
            volume=3000.0,
            vol_sma_20=1000.0,
            close_trend=[85800.0, 85900.0, 85950.0, 86000.0],
            macd_hist_trend=[5.0, 7.0, 8.0, 10.0],
            rsi_trend=[10.0, 12.0, 13.0, 15.0],
        )
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.confidence <= 100

    def test_insufficient_data_returns_none(self):
        """Fewer than 20 candles should return None."""
        df = _make_df(n=10)
        predictor = KalshiPredictor()
        assert predictor.score(df) is None

    def test_none_dataframe_returns_none(self):
        predictor = KalshiPredictor()
        assert predictor.score(None) is None

    def test_macd_negative_decreasing_gives_down(self):
        """MACD histogram negative and decreasing = 15 DOWN points."""
        df = _make_df(
            rsi=72.0,  # mild overbought for 20 pts
            macd_hist=-5.0,
            macd_hist_trend=[-2.0, -3.0, -4.0, -5.0],
        )
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "DOWN"
        assert signal.components["macd"]["down"] == 15

    def test_volume_confirms_dominant_direction(self):
        """High volume adds to whichever direction is already winning."""
        df = _make_df(
            rsi=22.0,  # strong oversold UP
            volume=2500.0,
            vol_sma_20=1000.0,
        )
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "UP"
        # Volume 2.5x avg = 10 pts, added to UP since UP is dominant
        assert signal.components["volume"]["score"] == 10

    def test_momentum_three_red_candles(self):
        """Strong negative ROC gives DOWN points (replaces old candle-color momentum)."""
        df = _make_df(rsi=68.0, roc_5=-2.0)  # mild overbought + strong negative ROC
        predictor = KalshiPredictor()
        signal = predictor.score(df)

        assert signal is not None
        assert signal.direction == "DOWN"
        assert signal.components["roc"]["down"] == 10

    def test_roc_strong_up_momentum(self):
        """ROC > 1.5% gives 10 UP points."""
        df = _make_df(rsi=50.0, roc_5=2.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["roc"]["up"] == 10

    def test_roc_moderate_up_momentum(self):
        """ROC between 0.5% and 1.5% gives 5 UP points."""
        df = _make_df(rsi=50.0, roc_5=0.8)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["roc"]["up"] == 5

    def test_roc_strong_down_momentum(self):
        """ROC < -1.5% gives 10 DOWN points."""
        df = _make_df(rsi=50.0, roc_5=-2.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["roc"]["down"] == 10

    def test_roc_neutral(self):
        """ROC between -0.5% and 0.5% gives 0 points."""
        df = _make_df(rsi=50.0, roc_5=0.1)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        if signal is not None:
            assert signal.components["roc"]["up"] == 0
            assert signal.components["roc"]["down"] == 0

    def test_stochrsi_oversold_gives_up_points(self):
        """StochRSI K < 10 gives 15 UP points."""
        df = _make_df(rsi=50.0, stochrsi_k=8.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["stochrsi"]["up"] == 15

    def test_stochrsi_overbought_gives_down_points(self):
        """StochRSI K > 90 gives 15 DOWN points."""
        df = _make_df(rsi=50.0, stochrsi_k=92.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["stochrsi"]["down"] == 15

    def test_stochrsi_moderate_oversold(self):
        """StochRSI K between 10-20 gives 8 UP points."""
        df = _make_df(rsi=50.0, stochrsi_k=15.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["stochrsi"]["up"] == 8

    def test_stochrsi_moderate_overbought(self):
        """StochRSI K between 80-90 gives 8 DOWN points."""
        df = _make_df(rsi=50.0, stochrsi_k=85.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["stochrsi"]["down"] == 8

    def test_atr_move_ratio_confirms_signal(self):
        """Candle move > 1.5x ATR in dominant direction gives +10."""
        df = _make_df(rsi=32.0, atr=200.0)  # RSI gives 10 UP
        # Override last candle to have big UP move (close > open by 350)
        df.iloc[-1, df.columns.get_loc("open")] = df.iloc[-1]["close"] - 350
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["atr_move"]["score"] == 10

    def test_atr_move_ratio_penalizes_overextension(self):
        """Candle move > 2.0x ATR against dominant direction gives -5 penalty."""
        df = _make_df(rsi=22.0, atr=200.0)  # RSI gives 30 UP
        # DOWN candle opposing dominant UP direction
        df.iloc[-1, df.columns.get_loc("open")] = df.iloc[-1]["close"] + 450
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["atr_move"]["score"] == -5

    def test_atr_move_ratio_no_effect_small_move(self):
        """Candle move < 1.5x ATR gives 0."""
        df = _make_df(rsi=32.0, atr=200.0)
        # Default _make_df sets open = close, so move is 0
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["atr_move"]["score"] == 0

    def test_signal_dataclass_fields(self):
        """KalshiSignal has all required fields."""
        sig = KalshiSignal(
            asset="BTC", direction="UP", confidence=65.0,
            components={"rsi": {"up": 20}}, price=87000.0, rsi=28.0,
        )
        assert sig.asset == "BTC"
        assert sig.direction == "UP"
        assert sig.confidence == 65.0
        assert sig.price == 87000.0
        assert sig.rsi == 28.0
        assert "rsi" in sig.components


# ---------------------------------------------------------------------------
# Enhanced predictor tests — leading indicators
# ---------------------------------------------------------------------------

class TestKalshiPredictorEnhanced:
    """Tests for leading indicator integration in the predictor."""

    def test_order_book_buy_pressure_boosts_up(self):
        """Strong order book buy imbalance adds UP points."""
        df = _make_df(rsi=32.0)  # mild oversold = 10 UP pts from RSI
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0.4, "spread_pct": 0.05},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["order_book"]["up"] == 20

    def test_order_book_sell_pressure_boosts_down(self):
        """Strong order book sell imbalance adds DOWN points."""
        df = _make_df(rsi=68.0)  # mild overbought = 10 DOWN pts
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": -0.35, "spread_pct": 0.05},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "DOWN"
        assert signal.components["order_book"]["down"] == 20

    def test_trade_flow_aggressive_buying(self):
        """Aggressive buying (net_flow > 0.2 and buy_ratio > 0.55) gives 20 UP."""
        df = _make_df(rsi=33.0)  # oversold
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.01},
            "trade_flow": {"net_flow": 0.25, "buy_ratio": 0.6, "large_trade_bias": 0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["trade_flow"]["up"] == 20

    def test_trade_flow_aggressive_selling(self):
        """Aggressive selling (net_flow < -0.2 and buy_ratio < 0.45) gives 20 DOWN."""
        df = _make_df(rsi=67.0)
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.01},
            "trade_flow": {"net_flow": -0.25, "buy_ratio": 0.38, "large_trade_bias": 0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "DOWN"
        assert signal.components["trade_flow"]["down"] == 20

    def test_large_trade_bias_whales_buying(self):
        """Whale buying (large_trade_bias > 0.3) gives 10 UP."""
        df = _make_df(rsi=33.0)
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.01},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0.5},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["large_trade"]["up"] == 10

    def test_wide_spread_confirms_direction(self):
        """Wide spread (> 0.1%) adds 5 to dominant direction."""
        df = _make_df(rsi=22.0)  # strong oversold
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.15},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.components["spread"]["score"] == 5

    def test_cross_asset_btc_dragging_alts_down(self):
        """BTC down >1% adds 10 DOWN for alts."""
        df = _make_df(rsi=50.0, close=87000.0)  # neutral lagging
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.01},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0},
            "cross_asset": {"market_direction": -2.5},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "DOWN"
        assert signal.components["cross_asset"]["down"] == 10

    def test_cross_asset_btc_lifting_alts_up(self):
        """BTC up >1% adds 10 UP for alts."""
        df = _make_df(rsi=50.0, close=87000.0)
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0, "spread_pct": 0.01},
            "trade_flow": {"net_flow": 0, "buy_ratio": 0.5, "large_trade_bias": 0},
            "cross_asset": {"market_direction": 2.0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["cross_asset"]["up"] == 10

    def test_no_market_data_backward_compatible(self):
        """Calling score() without market_data still works identically."""
        df = _make_df(rsi=22.0, close=86400.0, bb_lower=86500.0)
        predictor = KalshiPredictor()
        signal_old = predictor.score(df)
        signal_new = predictor.score(df, market_data=None)
        assert signal_old is not None
        assert signal_new is not None
        assert signal_old.direction == signal_new.direction
        assert signal_old.confidence == signal_new.confidence

    def test_enhanced_confidence_normalized(self):
        """With leading data, confidence is normalized to 0-100 scale."""
        df = _make_df(
            rsi=22.0, close=86400.0, bb_lower=86500.0,
            macd_hist=5.0, volume=2500.0, vol_sma_20=1000.0,
            roc_5=2.0, stochrsi_k=8.0, atr=200.0,
            macd_hist_trend=[2.0, 3.0, 4.0, 5.0],
            rsi_trend=[18.0, 19.0, 20.0, 22.0],
        )
        # Big UP candle: move = 350, ratio = 1.75 > 1.5 → +10 ATR
        df.iloc[-1, df.columns.get_loc("open")] = df.iloc[-1]["close"] - 350
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0.5, "spread_pct": 0.15},
            "trade_flow": {"net_flow": 0.3, "buy_ratio": 0.65, "large_trade_bias": 0.5},
            "cross_asset": {"market_direction": 2.0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.confidence <= 100
        # All lagging + leading components fire: 120 + 65 = 185 raw
        # _MAX_RAW = 200 (lagging 120 + leading 65 + MTF 15); no df_1h → MTF = 0
        # Normalized: 185 * 100 / 200 = 92
        assert signal.confidence == 92

    def test_leading_only_signal_works(self):
        """With neutral lagging but strong leading data, still produces signal."""
        df = _make_df(rsi=50.0, close=87000.0)  # completely neutral lagging
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0.5, "spread_pct": 0.2},
            "trade_flow": {"net_flow": 0.3, "buy_ratio": 0.65, "large_trade_bias": 0.5},
            "cross_asset": {"market_direction": 2.0},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"
        # Order book 20 + trade flow 20 + large trade 10 + spread 5 + cross 10 = 65
        # Normalized: 65 * 100 / 165 = 39
        assert signal.confidence >= 30


# ---------------------------------------------------------------------------
# MTF (1-hour trend alignment) tests
# ---------------------------------------------------------------------------

class TestKalshiPredictorMTF:
    def test_1h_trend_agrees_gives_bonus(self):
        """1h trend agreeing with 15m signal gives +15."""
        df = _make_df(rsi=22.0)  # strong UP signal from 15m
        df_1h = _make_df(n=50, rsi=65.0, macd_hist=5.0)  # 1h bullish
        predictor = KalshiPredictor()
        signal = predictor.score(df, df_1h=df_1h)
        assert signal is not None
        assert signal.direction == "UP"
        assert signal.components["mtf"]["score"] == 15

    def test_1h_trend_disagrees_gives_penalty(self):
        """1h trend opposing 15m signal gives -15 penalty."""
        df = _make_df(rsi=22.0)  # 15m says UP
        df_1h = _make_df(n=50, rsi=35.0, macd_hist=-5.0)  # 1h bearish
        predictor = KalshiPredictor()
        signal = predictor.score(df, df_1h=df_1h)
        assert signal is not None
        assert signal.components["mtf"]["score"] == -15

    def test_1h_trend_neutral_gives_zero(self):
        """1h trend neutral (RSI 40-60, MACD near zero) gives 0."""
        df = _make_df(rsi=22.0)
        df_1h = _make_df(n=50, rsi=50.0, macd_hist=0.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df, df_1h=df_1h)
        assert signal is not None
        assert signal.components["mtf"]["score"] == 0

    def test_no_1h_data_backward_compatible(self):
        """Calling score() without df_1h still works, MTF score = 0."""
        df = _make_df(rsi=22.0)
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.components["mtf"]["score"] == 0


# ---------------------------------------------------------------------------
# Signal quality filter tests
# ---------------------------------------------------------------------------

class TestKalshiFilters:
    """Tests for signal quality filters."""

    def test_directional_conflict_rejects_signal(self):
        """When lagging UP but leading DOWN (both >= 15), signal is rejected."""
        df = _make_df(rsi=22.0)  # lagging UP = 30 pts
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": -0.35, "spread_pct": 0.01},  # 20 DOWN
            "trade_flow": {"net_flow": -0.25, "buy_ratio": 0.38, "large_trade_bias": -0.4},  # 20+10 DOWN
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is None

    def test_no_conflict_when_aligned(self):
        """When lagging and leading agree, no conflict rejection."""
        df = _make_df(rsi=22.0)  # lagging UP
        predictor = KalshiPredictor()
        market_data = {
            "order_book": {"imbalance": 0.35, "spread_pct": 0.01},  # 20 UP
            "trade_flow": {"net_flow": 0.25, "buy_ratio": 0.6, "large_trade_bias": 0.4},
        }
        signal = predictor.score(df, market_data=market_data)
        assert signal is not None
        assert signal.direction == "UP"

    def test_volatility_too_high_rejects(self):
        """ATR in 90th+ percentile of its history rejects signal."""
        df = _make_df(n=250, rsi=22.0, atr=200.0)
        df.iloc[-1, df.columns.get_loc("atr")] = 2000.0
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is None

    def test_low_volatility_does_not_reject(self):
        """Low ATR should NOT reject — only high volatility spikes are filtered.
        Low-vol rejection was removed because it triggered false rejections
        during normal calm markets with limited data windows."""
        df = _make_df(n=250, rsi=22.0, atr=200.0)
        df.iloc[-1, df.columns.get_loc("atr")] = 5.0
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None  # should pass through, not rejected

    def test_margin_auto_passes_when_loser_zero(self):
        """When loser score is 0 or negative, margin filter auto-passes."""
        df = _make_df(rsi=22.0)  # 30 UP, 0 DOWN
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None
        assert signal.direction == "UP"

    def test_normal_volatility_passes(self):
        """ATR in normal range (10th-90th percentile) passes filter."""
        df = _make_df(n=250, rsi=22.0, atr=200.0)
        # Set a range of ATR values so percentile is meaningful
        atr_values = np.linspace(100, 300, 250)
        df["atr"] = atr_values  # last value is 300, percentile ~100%
        df.iloc[-1, df.columns.get_loc("atr")] = 200.0  # median = ~50th percentile
        predictor = KalshiPredictor()
        signal = predictor.score(df)
        assert signal is not None


# ---------------------------------------------------------------------------
# 1-minute momentum check tests
# ---------------------------------------------------------------------------

class TestKalshi1mMomentum:
    """Tests for 1-minute momentum check."""

    def test_up_momentum_confirmed(self):
        """2 of 3 green candles confirms UP."""
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0, 101.5, 103.0],
            "close": [101.0, 102.0, 101.0, 103.0, 104.0],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is True

    def test_down_momentum_confirmed(self):
        """2 of 3 red candles confirms DOWN."""
        df = pd.DataFrame({
            "open":  [104.0, 103.0, 102.0, 103.0, 101.0],
            "close": [103.0, 102.0, 103.0, 101.0, 100.0],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "DOWN", lookback=3) is True

    def test_momentum_not_confirmed(self):
        """All candles against direction fails."""
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [101.0, 100.5, 101.5, 102.0, 103.5],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        # Last 3: 102->101.5 DOWN, 103->102 DOWN, 104->103.5 DOWN
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is False

    def test_insufficient_data_returns_false(self):
        """Less than lookback candles returns False."""
        df = pd.DataFrame({
            "open": [100.0],
            "close": [101.0],
        }, index=pd.date_range("2026-01-01", periods=1, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is False


# ---------------------------------------------------------------------------
# 5-minute confirmation booster tests
# ---------------------------------------------------------------------------

class TestKalshi5mBooster:
    """Tests for 5m confirmation booster."""

    def _make_5m_df(self, n=10, close=100.0, open_price=99.0, volume=1000.0,
                     vol_sma_20=1000.0, atr=2.0, rsi=50.0, macd_hist=0.5,
                     prev_rsi=50.0, prev_macd_hist=0.5, prev_close=99.5, prev_open=99.0):
        """Build a small 5m DataFrame for booster testing."""
        import numpy as np
        closes = np.full(n, close)
        opens = np.full(n, open_price)
        volumes = np.full(n, volume)

        # Set last 2 candles explicitly
        closes[-2] = prev_close
        opens[-2] = prev_open
        closes[-1] = close
        opens[-1] = open_price

        df = pd.DataFrame({
            "close": closes,
            "open": opens,
            "volume": volumes,
            "vol_sma_20": np.full(n, vol_sma_20),
            "atr": np.full(n, atr),
            "rsi": np.full(n, rsi),
            "macd_hist": np.full(n, macd_hist),
        }, index=pd.date_range("2026-01-01", periods=n, freq="5min"))
        # Set prev candle indicators
        df.iloc[-2, df.columns.get_loc("rsi")] = prev_rsi
        df.iloc[-2, df.columns.get_loc("macd_hist")] = prev_macd_hist
        return df

    def test_all_boost_factors_fire(self):
        """All 4 boost factors give +12."""
        # UP prediction: green candle, high volume, ATR distance, MACD crossover aligned
        df = self._make_5m_df(
            close=103.0, open_price=101.0,    # green candle (+3)
            volume=2000.0, vol_sma_20=1000.0, # volume 2x avg (+3)
            atr=2.0,                          # ATR distance: 103-100=3 > 0.5*2=1 (+3)
            macd_hist=0.5, prev_macd_hist=-0.3, # MACD crossed positive (+3)
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.0)
        assert booster == 12

    def test_all_penalty_factors_fire(self):
        """Both penalty factors give -10."""
        # UP prediction but: MACD crossed negative (-5), RSI divergence (-5)
        # Use a doji/flat candle (close == open) so candle direction doesn't add +3
        df = self._make_5m_df(
            close=100.0, open_price=100.0,    # doji — no candle direction boost
            volume=500.0, vol_sma_20=1000.0,  # low volume
            atr=2.0,
            macd_hist=-0.3, prev_macd_hist=0.5,  # MACD crossed against UP (-5)
            rsi=48.0, prev_rsi=50.0,              # price up but RSI down = divergence (-5)
            prev_close=99.0, prev_open=99.5,      # prev close < current close (price up for divergence)
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.5)
        assert booster == -10

    def test_candle_direction_boost_up(self):
        """Green candle gives +3 for UP prediction."""
        df = self._make_5m_df(close=101.0, open_price=100.0)
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.0)
        assert booster >= 3  # at least candle direction fires

    def test_candle_direction_no_boost_wrong_direction(self):
        """Red candle gives 0 for UP prediction."""
        df = self._make_5m_df(close=99.0, open_price=100.0)
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.0)
        # Candle direction doesn't fire, but no penalty either (just 0 for this factor)
        assert booster <= 0 or booster < 3

    def test_volume_surge_boost(self):
        """Volume > 1.5x avg gives +3."""
        df = self._make_5m_df(
            close=101.0, open_price=100.0,      # green (+3)
            volume=2000.0, vol_sma_20=1000.0,   # 2x avg (+3)
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=101.0)
        # candle direction fires (+3), volume fires (+3), ATR distance may not
        assert booster >= 6

    def test_macd_crossover_against_penalty(self):
        """MACD crossing against direction gives -5."""
        df = self._make_5m_df(
            close=100.5, open_price=100.0,
            macd_hist=-0.5, prev_macd_hist=0.3,  # crossed negative while predicting UP
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.0)
        # Should have -5 penalty from MACD crossover against
        assert booster <= -2  # candle dir might give +3, net <= -2

    def test_rsi_divergence_penalty(self):
        """Price up but RSI down gives -5 for UP prediction."""
        df = self._make_5m_df(
            close=101.0, open_price=100.0,
            rsi=48.0, prev_rsi=52.0,            # RSI dropped while price went up
            prev_close=100.0, prev_open=99.5,   # prev close < current close (price up)
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=100.0)
        # RSI divergence should fire (-5)
        assert booster <= 1  # candle +3, divergence -5, maybe some others

    def test_insufficient_data_returns_zero(self):
        """Less than 2 candles returns 0."""
        df = pd.DataFrame({
            "close": [100.0], "open": [99.0], "volume": [1000.0],
            "vol_sma_20": [1000.0], "atr": [2.0], "rsi": [50.0], "macd_hist": [0.5],
        }, index=pd.date_range("2026-01-01", periods=1, freq="5min"))
        predictor = KalshiPredictor()
        assert predictor.compute_5m_booster(df, "UP", 100.0) == 0

    def test_clamped_to_range(self):
        """Result is always in [-10, +12]."""
        predictor = KalshiPredictor()
        # Even with extreme values, should clamp
        df = self._make_5m_df(
            close=110.0, open_price=100.0,
            volume=5000.0, vol_sma_20=1000.0,
            atr=1.0,
            macd_hist=5.0, prev_macd_hist=-5.0,
        )
        booster = predictor.compute_5m_booster(df, "UP", window_open_price=90.0)
        assert -10 <= booster <= 12

    def test_down_prediction_booster(self):
        """Booster works correctly for DOWN predictions."""
        # Red candle, high volume, MACD crossed negative, price below window open
        df = self._make_5m_df(
            close=97.0, open_price=99.0,        # red candle (+3)
            volume=2000.0, vol_sma_20=1000.0,   # volume (+3)
            atr=2.0,                            # 100-97=3 > 0.5*2=1 (+3)
            macd_hist=-0.5, prev_macd_hist=0.3, # MACD crossed negative (+3)
        )
        predictor = KalshiPredictor()
        booster = predictor.compute_5m_booster(df, "DOWN", window_open_price=100.0)
        assert booster == 12
