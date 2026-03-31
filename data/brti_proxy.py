# data/brti_proxy.py
"""BRTI proxy — approximates CF Benchmarks Real Time Index.

BRTI = VWAP across Coinbase, Kraken, Bitstamp, Gemini, Bullish, Crypto.com, LMAX.
We use the top 3 (Coinbase, Kraken, Bitstamp) for a fast approximation.

Fetches in parallel (~3s for all assets), caches for 10 seconds.
Falls back to Coinbase-only if other exchanges fail.
"""
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import ccxt


class BRTIProxy:
    """Multi-exchange price aggregator approximating CF BRTI."""

    EXCHANGES = {
        "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
        "kraken": lambda: ccxt.kraken({"enableRateLimit": True}),
        "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
    }

    CACHE_TTL = 10  # seconds

    def __init__(self):
        self._exchanges = {}
        self._cache = {}  # {symbol: (price, timestamp)}
        self._lock = threading.Lock()

    def _get_exchange(self, name: str):
        if name not in self._exchanges:
            self._exchanges[name] = self.EXCHANGES[name]()
        return self._exchanges[name]

    def get_price(self, symbol: str) -> float | None:
        """Get BRTI-approximated price for a symbol (e.g., 'BTC/USD').

        Returns averaged price from 3 exchanges, cached for 10 seconds.
        """
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and time.time() - cached[1] < self.CACHE_TTL:
                return cached[0]

        # Fetch from all exchanges in parallel
        def _fetch(name):
            try:
                ex = self._get_exchange(name)
                ticker = ex.fetch_ticker(symbol)
                return ticker.get("last")
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_fetch, self.EXCHANGES.keys()))

        prices = [p for p in results if p is not None]
        if not prices:
            return None

        avg = sum(prices) / len(prices)

        with self._lock:
            self._cache[symbol] = (avg, time.time())

        return avg

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get BRTI prices for multiple symbols in one parallel batch.

        More efficient than calling get_price() per symbol.
        """
        # Check cache first
        results = {}
        to_fetch = []
        now = time.time()
        with self._lock:
            for sym in symbols:
                cached = self._cache.get(sym)
                if cached and now - cached[1] < self.CACHE_TTL:
                    results[sym] = cached[0]
                else:
                    to_fetch.append(sym)

        if not to_fetch:
            return results

        # Fetch all needed symbols from all exchanges in parallel
        def _fetch(args):
            name, sym = args
            try:
                ex = self._get_exchange(name)
                ticker = ex.fetch_ticker(sym)
                return (name, sym, ticker.get("last"))
            except Exception:
                return (name, sym, None)

        tasks = [(n, s) for s in to_fetch for n in self.EXCHANGES]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            fetch_results = list(pool.map(_fetch, tasks))

        # Average per symbol
        from collections import defaultdict
        by_sym = defaultdict(list)
        for name, sym, price in fetch_results:
            if price is not None:
                by_sym[sym].append(price)

        with self._lock:
            for sym in to_fetch:
                prices = by_sym.get(sym, [])
                if prices:
                    avg = sum(prices) / len(prices)
                    results[sym] = avg
                    self._cache[sym] = (avg, time.time())

        return results
