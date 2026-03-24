import ccxt
import pandas as pd
import time
from config.settings import DEFAULT_CANDLE_LIMIT, CDP_KEY_FILE


class DataFetcher:
    def __init__(self):
        self._spot = ccxt.binance({"enableRateLimit": True})
        self._futures = ccxt.binanceusdm({"enableRateLimit": True})
        self._coinbase = None  # lazy init

    def _get_coinbase(self):
        if self._coinbase is None:
            from coinbase.rest import RESTClient
            self._coinbase = RESTClient(key_file=str(CDP_KEY_FILE))
        return self._coinbase

    def ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = DEFAULT_CANDLE_LIMIT,
        since: int | None = None, source: str = "binance"
    ) -> pd.DataFrame:
        """Fetch OHLCV candles. Returns DataFrame indexed by timestamp."""
        if source == "binance":
            is_futures = ":" in symbol
            exchange = self._futures if is_futures else self._spot
            raw = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            return df
        elif source == "coinbase":
            return self._coinbase_ohlcv(symbol, timeframe, limit)
        else:
            raise ValueError(f"Unknown source: {source}")

    def _coinbase_ohlcv(self, product_id: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV from Coinbase Advanced Trade."""
        granularity_map = {
            "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE",
            "30m": "THIRTY_MINUTE", "1h": "ONE_HOUR", "2h": "TWO_HOUR",
            "6h": "SIX_HOUR", "1d": "ONE_DAY",
        }
        granularity = granularity_map.get(timeframe, "ONE_HOUR")

        now = int(time.time())
        seconds_per = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                       "1h": 3600, "2h": 7200, "6h": 21600, "1d": 86400}
        start = now - (limit * seconds_per.get(timeframe, 3600))

        client = self._get_coinbase()
        resp = client.get_candles(
            product_id=product_id, start=str(start), end=str(now), granularity=granularity
        )
        candles = resp["candles"]
        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["start"].astype(int), unit="s")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("timestamp").sort_index()
        return df

    def funding_rate(self, symbol: str) -> dict:
        """Fetch current funding rate for a futures symbol."""
        raw = self._futures.fetch_funding_rate(symbol)
        rate = raw["fundingRate"]
        raw["annualized"] = rate * 3 * 365 if rate else None
        return raw

    def funding_rates_all(self) -> list[dict]:
        """Fetch funding rates for all futures symbols."""
        raw = self._futures.fetch_funding_rates()
        results = []
        for symbol, data in raw.items():
            rate = data.get("fundingRate")
            data["annualized"] = rate * 3 * 365 if rate else None
            results.append(data)
        return sorted(results, key=lambda x: abs(x.get("annualized") or 0), reverse=True)

    def ticker(self, symbol: str, source: str = "binance") -> dict:
        """Fetch current ticker data."""
        if source == "binance":
            is_futures = ":" in symbol
            exchange = self._futures if is_futures else self._spot
            return exchange.fetch_ticker(symbol)
        elif source == "coinbase":
            client = self._get_coinbase()
            product = client.get_product(symbol)
            return product
        else:
            raise ValueError(f"Unknown source: {source}")

    def order_book(self, symbol: str, depth: int = 20, source: str = "binance") -> dict:
        """Fetch order book."""
        if source == "binance":
            is_futures = ":" in symbol
            exchange = self._futures if is_futures else self._spot
            return exchange.fetch_order_book(symbol, limit=depth)
        elif source == "coinbase":
            client = self._get_coinbase()
            return client.get_product_book(symbol, limit=depth)
        else:
            raise ValueError(f"Unknown source: {source}")
