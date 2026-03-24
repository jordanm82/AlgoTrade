# tests/test_kalshi.py
import pytest
from unittest.mock import patch, MagicMock
from exchange.kalshi import KalshiClient
from strategy.strategies.kalshi_btc import KalshiBTCStrategy
import pandas as pd
import numpy as np


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


class TestKalshiBTCStrategy:
    @pytest.fixture
    def oversold_df(self):
        n = 50
        close = np.full(n, 87000.0)
        df = pd.DataFrame({
            "close": close, "high": close + 100, "low": close - 100,
            "open": close, "volume": [1000] * n,
            "rsi": [25.0] * n,  # oversold
            "bb_lower": [86500.0] * n,
            "bb_middle": [87000.0] * n,
            "bb_upper": [87500.0] * n,
        }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
        return df

    def test_oversold_recommends_yes(self, oversold_df):
        strat = KalshiBTCStrategy()
        market = {
            "ticker": "KXBTC-TEST-B86000",
            "yes_ask": 60, "no_ask": 40,
            "floor_strike": 86000,
            "_mins_to_expiry": 15,
        }
        result = strat.evaluate(oversold_df, market)
        assert result is not None
        assert result["side"] == "yes"
        assert result["edge_cents"] > 0

    def test_overbought_recommends_no(self):
        n = 50
        close = np.full(n, 87000.0)
        df = pd.DataFrame({
            "close": close, "high": close + 100, "low": close - 100,
            "open": close, "volume": [1000] * n,
            "rsi": [78.0] * n,
            "bb_lower": [86500.0] * n, "bb_middle": [87000.0] * n, "bb_upper": [87500.0] * n,
        }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
        strat = KalshiBTCStrategy()
        market = {
            "ticker": "KXBTC-TEST-B87000",
            "yes_ask": 55, "no_ask": 45,
            "floor_strike": 87000,
            "_mins_to_expiry": 15,
        }
        result = strat.evaluate(df, market)
        assert result is not None
        assert result["side"] == "no"

    def test_neutral_rsi_no_bet(self):
        n = 50
        close = np.full(n, 87000.0)
        df = pd.DataFrame({
            "close": close, "high": close + 100, "low": close - 100,
            "open": close, "volume": [1000] * n,
            "rsi": [50.0] * n,
            "bb_lower": [86500.0] * n, "bb_middle": [87000.0] * n, "bb_upper": [87500.0] * n,
        }, index=pd.date_range("2025-01-01", periods=n, freq="15min"))
        strat = KalshiBTCStrategy()
        market = {
            "ticker": "KXBTC-TEST-B87000",
            "yes_ask": 50, "no_ask": 50,
            "floor_strike": 87000, "_mins_to_expiry": 15,
        }
        result = strat.evaluate(df, market)
        assert result is None

    def test_too_close_to_expiry_no_bet(self, oversold_df):
        strat = KalshiBTCStrategy()
        market = {
            "ticker": "KXBTC-TEST-B86000",
            "yes_ask": 60, "no_ask": 40,
            "floor_strike": 86000,
            "_mins_to_expiry": 3,  # too close
        }
        result = strat.evaluate(oversold_df, market)
        assert result is None
