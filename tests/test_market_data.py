# tests/test_market_data.py
"""Tests for real-time market microstructure data helpers."""
import pytest
from unittest.mock import patch, MagicMock
from data.market_data import get_order_book_imbalance, get_trade_flow, get_cross_asset_momentum


class TestOrderBookImbalance:
    def test_buy_pressure(self):
        mock_book = {
            "bids": [[100, 10], [99, 8], [98, 5]],
            "asks": [[101, 2], [102, 1], [103, 1]],
        }
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_order_book.return_value = mock_book
            result = get_order_book_imbalance("BTC/USDT")
            assert result["imbalance"] > 0.5  # strong buy pressure
            assert result["spread_pct"] > 0
            assert result["bid_volume"] == 23
            assert result["ask_volume"] == 4

    def test_sell_pressure(self):
        mock_book = {
            "bids": [[100, 1], [99, 1]],
            "asks": [[101, 10], [102, 8], [103, 5]],
        }
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_order_book.return_value = mock_book
            result = get_order_book_imbalance("BTC/USDT")
            assert result["imbalance"] < -0.5  # strong sell pressure

    def test_balanced_book(self):
        mock_book = {
            "bids": [[100, 5], [99, 5]],
            "asks": [[101, 5], [102, 5]],
        }
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_order_book.return_value = mock_book
            result = get_order_book_imbalance("BTC/USDT")
            assert abs(result["imbalance"]) < 0.01

    def test_empty_book(self):
        mock_book = {"bids": [], "asks": []}
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_order_book.return_value = mock_book
            result = get_order_book_imbalance("BTC/USDT")
            assert result["imbalance"] == 0
            assert result["spread_pct"] == 0

    def test_wall_detection(self):
        # One massive bid level among small ones
        mock_book = {
            "bids": [[100, 100], [99, 1], [98, 1]],
            "asks": [[101, 1], [102, 1], [103, 1]],
        }
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_order_book.return_value = mock_book
            result = get_order_book_imbalance("BTC/USDT")
            assert result["wall_ratio"] > 3  # 100 is way above average


class TestTradeFlow:
    def test_net_buying(self):
        mock_trades = [
            {"side": "buy", "amount": 1.0} for _ in range(70)
        ] + [
            {"side": "sell", "amount": 1.0} for _ in range(30)
        ]
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_trades.return_value = mock_trades
            result = get_trade_flow("BTC/USDT")
            assert result["net_flow"] > 0
            assert result["buy_ratio"] > 0.6

    def test_net_selling(self):
        mock_trades = [
            {"side": "buy", "amount": 1.0} for _ in range(20)
        ] + [
            {"side": "sell", "amount": 1.0} for _ in range(80)
        ]
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_trades.return_value = mock_trades
            result = get_trade_flow("BTC/USDT")
            assert result["net_flow"] < 0
            assert result["buy_ratio"] < 0.3

    def test_large_trade_bias_buy(self):
        # avg_size = (20+1+1+1+1)/5 = 4.8, threshold 2x = 9.6 => 20.0 is large
        mock_trades = [
            {"side": "buy", "amount": 20.0},  # large (> 2x avg)
            {"side": "sell", "amount": 1.0},
            {"side": "sell", "amount": 1.0},
            {"side": "buy", "amount": 1.0},
            {"side": "buy", "amount": 1.0},
        ]
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_trades.return_value = mock_trades
            result = get_trade_flow("BTC/USDT")
            assert result["large_trade_bias"] > 0

    def test_large_trade_bias_sell(self):
        # avg_size = (20+1+1+1+1)/5 = 4.8, threshold 2x = 9.6 => 20.0 is large
        mock_trades = [
            {"side": "sell", "amount": 20.0},  # large (> 2x avg)
            {"side": "buy", "amount": 1.0},
            {"side": "buy", "amount": 1.0},
            {"side": "sell", "amount": 1.0},
            {"side": "sell", "amount": 1.0},
        ]
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_trades.return_value = mock_trades
            result = get_trade_flow("BTC/USDT")
            assert result["large_trade_bias"] < 0

    def test_empty_trades(self):
        with patch("data.market_data.exchange") as mock_ex:
            mock_ex.fetch_trades.return_value = []
            result = get_trade_flow("BTC/USDT")
            assert result["net_flow"] == 0
            assert result["buy_ratio"] == 0.5


class TestCrossAssetMomentum:
    def test_returns_all_pairs(self):
        def mock_ticker(sym):
            return {"last": 100, "percentage": 1.5, "quoteVolume": 1000000}

        with patch("data.market_data.exchange") as mock_ex, \
             patch("data.market_data.time"):
            mock_ex.fetch_ticker.side_effect = mock_ticker
            result = get_cross_asset_momentum()
            assert "BTC/USDT" in result
            assert "ETH/USDT" in result
            assert "SOL/USDT" in result
            assert "XRP/USDT" in result
            assert "btc_leads" in result
            assert "market_direction" in result

    def test_btc_leading_alts(self):
        tickers = {
            "BTC/USDT": {"last": 90000, "percentage": 3.0, "quoteVolume": 5e9},
            "ETH/USDT": {"last": 3000, "percentage": 1.0, "quoteVolume": 2e9},
            "SOL/USDT": {"last": 150, "percentage": 0.5, "quoteVolume": 1e9},
            "XRP/USDT": {"last": 0.6, "percentage": 0.8, "quoteVolume": 5e8},
        }

        with patch("data.market_data.exchange") as mock_ex, \
             patch("data.market_data.time"):
            mock_ex.fetch_ticker.side_effect = lambda sym: tickers[sym]
            result = get_cross_asset_momentum()
            # BTC at +3% vs alts averaging ~0.77% => btc_leads > 0
            assert result["btc_leads"] > 1.0
            assert result["market_direction"] == 3.0
