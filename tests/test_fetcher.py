import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.fetcher import DataFetcher

class TestBinanceData:
    def test_ohlcv_returns_dataframe(self, sample_ohlcv):
        """fetch_ohlcv should return a DataFrame with expected columns."""
        mock_data = [
            [1704067200000, 50000.0, 50200.0, 49800.0, 50100.0, 1234.5],
            [1704070800000, 50100.0, 50300.0, 49900.0, 50200.0, 1100.2],
        ]
        with patch("data.fetcher.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = mock_data
            mock_ccxt.binanceus.return_value = mock_exchange

            fetcher = DataFetcher()
            df = fetcher.ohlcv("BTC/USDT", "1h", limit=2, source="binance")

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert len(df) == 2
            assert df.index.name == "timestamp"

    def test_funding_rate_returns_dict(self):
        """fetch_funding_rate should return rate info."""
        mock_rate = {
            "symbol": "BTC/USDT:USDT",
            "fundingRate": 0.0001,
            "fundingTimestamp": 1704067200000,
            "nextFundingRate": 0.00012,
        }
        with patch("data.fetcher.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.fetch_funding_rate.return_value = mock_rate
            mock_ccxt.binanceusdm.return_value = mock_exchange

            fetcher = DataFetcher()
            rate = fetcher.funding_rate("BTC/USDT:USDT")

            assert rate["fundingRate"] == 0.0001
            assert "annualized" in rate

    def test_ticker_returns_dict(self):
        """fetch_ticker should return price and volume info."""
        mock_ticker = {
            "symbol": "BTC/USDT",
            "last": 50000.0,
            "bid": 49999.0,
            "ask": 50001.0,
            "baseVolume": 12345.6,
            "quoteVolume": 617280000.0,
            "percentage": 1.5,
        }
        with patch("data.fetcher.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ticker.return_value = mock_ticker
            mock_ccxt.binanceus.return_value = mock_exchange

            fetcher = DataFetcher()
            ticker = fetcher.ticker("BTC/USDT", source="binance")

            assert ticker["last"] == 50000.0

    def test_order_book_returns_bids_asks(self):
        """fetch_order_book should return bids and asks."""
        mock_book = {
            "bids": [[49999.0, 1.5], [49998.0, 2.0]],
            "asks": [[50001.0, 1.0], [50002.0, 3.0]],
            "timestamp": 1704067200000,
        }
        with patch("data.fetcher.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_exchange.fetch_order_book.return_value = mock_book
            mock_ccxt.binanceus.return_value = mock_exchange

            fetcher = DataFetcher()
            book = fetcher.order_book("BTC/USDT", depth=2, source="binance")

            assert len(book["bids"]) == 2
            assert len(book["asks"]) == 2
