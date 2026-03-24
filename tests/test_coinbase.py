import pytest
from unittest.mock import patch, MagicMock
from exchange.coinbase import CoinbaseExecutor

@pytest.fixture
def mock_client():
    with patch("exchange.coinbase.RESTClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        yield client

class TestCoinbaseExecutor:
    def test_market_buy_spot(self, mock_client):
        mock_client.market_order_buy.return_value = MagicMock(
            to_dict=lambda: {"success": True, "success_response": {"order_id": "123"}}
        )
        executor = CoinbaseExecutor("fake_key.json")
        result = executor.market_buy("BTC-USD", 500.0)
        mock_client.market_order_buy.assert_called_once()
        call_kwargs = mock_client.market_order_buy.call_args
        assert call_kwargs.kwargs["product_id"] == "BTC-USD"
        assert call_kwargs.kwargs["quote_size"] == "500.00"
        assert result["success"] is True

    def test_market_sell_spot(self, mock_client):
        mock_client.market_order_sell.return_value = MagicMock(
            to_dict=lambda: {"success": True, "success_response": {"order_id": "456"}}
        )
        executor = CoinbaseExecutor("fake_key.json")
        result = executor.market_sell("BTC-USD", 0.01)
        mock_client.market_order_sell.assert_called_once()
        call_kwargs = mock_client.market_order_sell.call_args
        assert call_kwargs.kwargs["base_size"] == "0.01000000"

    def test_open_perp_long(self, mock_client):
        mock_client.market_order_buy.return_value = MagicMock(
            to_dict=lambda: {"success": True, "success_response": {"order_id": "789"}}
        )
        executor = CoinbaseExecutor("fake_key.json")
        result = executor.open_perp_long("BTC-PERP-INTX", 0.01, leverage=3)
        call_kwargs = mock_client.market_order_buy.call_args
        assert call_kwargs.kwargs["leverage"] == "3"
        assert call_kwargs.kwargs["margin_type"] == "CROSS"

    def test_open_perp_short(self, mock_client):
        mock_client.market_order_sell.return_value = MagicMock(
            to_dict=lambda: {"success": True, "success_response": {"order_id": "012"}}
        )
        executor = CoinbaseExecutor("fake_key.json")
        result = executor.open_perp_short("BTC-PERP-INTX", 0.01, leverage=5)
        call_kwargs = mock_client.market_order_sell.call_args
        assert call_kwargs.kwargs["leverage"] == "5"

    def test_get_balances(self, mock_client):
        mock_client.get_accounts.return_value = {
            "accounts": [
                {"currency": "USD", "available_balance": {"value": "5000.00"}},
                {"currency": "BTC", "available_balance": {"value": "0.1"}},
            ]
        }
        executor = CoinbaseExecutor("fake_key.json")
        balances = executor.get_balances()
        assert "USD" in balances
        assert balances["USD"] == 5000.0

    def test_close_perp(self, mock_client):
        mock_client.close_position.return_value = MagicMock(
            to_dict=lambda: {"success": True}
        )
        executor = CoinbaseExecutor("fake_key.json")
        result = executor.close_perp("BTC-PERP-INTX")
        mock_client.close_position.assert_called_once()
