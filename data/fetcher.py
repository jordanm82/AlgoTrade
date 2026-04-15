import ccxt
import pandas as pd
import time
from config.settings import DEFAULT_CANDLE_LIMIT


class DataFetcher:
    """Unified data fetcher — uses Coinbase as primary source (matches BRTI)."""

    def __init__(self):
        self._exchange = None  # lazy init Coinbase via CCXT

    def _get_exchange(self):
        if self._exchange is None:
            # Public market-data client for candles.
            # Candles do not require authenticated brokerage endpoints; using
            # public mode avoids auth-induced 401s on transaction_summary.
            self._exchange = ccxt.coinbase({
                "enableRateLimit": True,
            })
        return self._exchange

    def ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = DEFAULT_CANDLE_LIMIT,
        since: int | None = None, source: str = "coinbase"
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Coinbase. Returns DataFrame indexed by timestamp.

        Symbol format: 'ETH/USD', 'BTC/USD', etc.
        Also accepts 'ETH/USDT' — auto-converts to 'ETH/USD' for Coinbase.
        """
        # Normalize symbol: USDT → USD for Coinbase
        cb_symbol = symbol.replace("/USDT", "/USD")

        exchange = self._get_exchange()

        # Handle 4h timeframe — Coinbase doesn't have it, use 1h and resample
        if timeframe == "4h":
            return self._fetch_4h(cb_symbol, limit, since)

        import time as _time
        for attempt in range(3):
            try:
                raw = exchange.fetch_ohlcv(cb_symbol, timeframe, since=since, limit=limit)
                break
            except Exception:
                if attempt < 2:
                    _time.sleep(2 * (attempt + 1))
                else:
                    raise
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df

    def _fetch_4h(self, symbol: str, limit: int, since: int | None) -> pd.DataFrame:
        """Synthesize 4h candles from 1h data."""
        # Fetch 4x the 1h candles needed
        import time as _time
        h_limit = min(limit * 4, 1000)
        for attempt in range(3):
            try:
                raw = self._get_exchange().fetch_ohlcv(symbol, "1h", since=since, limit=h_limit)
                break
            except Exception:
                if attempt < 2:
                    _time.sleep(2 * (attempt + 1))
                else:
                    raise
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        # Resample to 4h
        resampled = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled.tail(limit)
