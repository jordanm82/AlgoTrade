# tests/test_kalshi_extended.py
"""Tests for new KalshiClient methods: action param, get_order_status, cancel_order_safe."""
import pytest
from unittest.mock import patch, MagicMock
from exchange.kalshi import KalshiClient
import requests


def _make_client():
    """Create a KalshiClient with mocked key."""
    client = KalshiClient.__new__(KalshiClient)
    client.api_key_id = "test"
    client.base_url = "https://demo"
    client._private_key = MagicMock()
    client._private_key.sign.return_value = b"sig"
    return client


class TestPlaceOrderAction:
    def test_default_action_is_buy(self):
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "yes", 10, price_cents=48, order_type="limit")
            data = mock.call_args[0][1]
            assert data["action"] == "buy"

    def test_sell_action(self):
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "yes", 10, price_cents=52,
                               order_type="limit", action="sell")
            data = mock.call_args[0][1]
            assert data["action"] == "sell"
            assert data["side"] == "yes"
            assert data["yes_price"] == 52

    def test_existing_callers_unaffected(self):
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "no", 5, price_cents=30, order_type="limit")
            data = mock.call_args[0][1]
            assert data["action"] == "buy"
            assert data["no_price"] == 30


class TestGetOrderStatus:
    def test_returns_order_data(self):
        client = _make_client()
        with patch.object(client, "_get", return_value={"order": {"status": "resting"}}) as mock:
            result = client.get_order_status("order-123")
            mock.assert_called_once_with("/trade-api/v2/portfolio/orders/order-123")
            assert result["status"] == "resting"


class TestCancelOrderSafe:
    def test_successful_cancel(self):
        client = _make_client()
        with patch.object(client, "_delete", return_value={"order": {"status": "cancelled"}}):
            result = client.cancel_order_safe("order-123")
            assert result["order"]["status"] == "cancelled"

    def test_404_returns_filled(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 404
        http_err = requests.HTTPError(response=resp)
        with patch.object(client, "_delete", side_effect=http_err):
            result = client.cancel_order_safe("order-123")
            assert result["status"] == "filled"

    def test_other_http_error_reraises(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 500
        http_err = requests.HTTPError(response=resp)
        with patch.object(client, "_delete", side_effect=http_err):
            with pytest.raises(requests.HTTPError):
                client.cancel_order_safe("order-123")
