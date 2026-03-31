# data/brti_proxy.py
"""BRTI proxy — real-time multi-exchange price averaging.

BRTI settles on a VWAP of Coinbase, Kraken, Bitstamp, Gemini, and others.
We average the top 3 constituent exchanges for a ~$5 approximation.

Each call is a LIVE API fetch — no stale SSR scraping.
Parallel fetch, 10-second cache to avoid rate limits.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import ccxt


class BRTIProxy:
    """Multi-exchange real-time price averaging (BRTI constituent exchanges)."""

    CACHE_TTL = 10  # seconds

    def __init__(self):
        self._exchanges = {}
        self._cache = {}  # {symbol: (price, timestamp)}
        self._lock = threading.Lock()

    def _get_exchange(self, name: str):
        if name not in self._exchanges:
            constructors = {
                "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
                "kraken": lambda: ccxt.kraken({"enableRateLimit": True}),
                "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
            }
            if name in constructors:
                self._exchanges[name] = constructors[name]()
        return self._exchanges.get(name)

    def get_price(self, symbol: str) -> float | None:
        """Get real-time averaged price for a symbol (e.g., 'BTC/USD').

        Averages Coinbase + Kraken + Bitstamp last trade prices.
        Cached for 10 seconds. Each underlying call is a live API fetch.
        """
        # Normalize symbol
        sym = symbol.replace("/USDT", "/USD")

        with self._lock:
            cached = self._cache.get(sym)
            if cached and time.time() - cached[1] < self.CACHE_TTL:
                return cached[0]

        def _fetch(name):
            try:
                ex = self._get_exchange(name)
                if ex is None:
                    return None
                ticker = ex.fetch_ticker(sym)
                return ticker.get("last")
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_fetch, ["coinbase", "kraken", "bitstamp"]))

        prices = [p for p in results if p is not None]
        if not prices:
            return None

        avg = sum(prices) / len(prices)

        with self._lock:
            self._cache[sym] = (avg, time.time())
            # Also cache the USDT variant
            usdt_sym = sym.replace("/USD", "/USDT")
            self._cache[usdt_sym] = (avg, time.time())

        return avg

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get prices for multiple symbols in one parallel batch."""
        # Deduplicate and normalize
        unique = set()
        sym_map = {}
        for s in symbols:
            normalized = s.replace("/USDT", "/USD")
            unique.add(normalized)
            sym_map[s] = normalized

        # Check cache
        results = {}
        to_fetch = set()
        now = time.time()
        with self._lock:
            for orig_sym in symbols:
                norm = sym_map[orig_sym]
                cached = self._cache.get(norm)
                if cached and now - cached[1] < self.CACHE_TTL:
                    results[orig_sym] = cached[0]
                else:
                    to_fetch.add(norm)

        if not to_fetch:
            return results

        # Fetch all symbols from all exchanges in parallel
        def _fetch(args):
            name, sym = args
            try:
                ex = self._get_exchange(name)
                if ex is None:
                    return (sym, None)
                ticker = ex.fetch_ticker(sym)
                return (sym, ticker.get("last"))
            except Exception:
                return (sym, None)

        tasks = [(n, s) for s in to_fetch for n in ["coinbase", "kraken", "bitstamp"]]
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            fetch_results = list(pool.map(_fetch, tasks))

        # Average per symbol
        from collections import defaultdict
        by_sym = defaultdict(list)
        for sym, price in fetch_results:
            if price is not None:
                by_sym[sym].append(price)

        with self._lock:
            for norm_sym, prices in by_sym.items():
                if prices:
                    avg = sum(prices) / len(prices)
                    self._cache[norm_sym] = (avg, time.time())
                    usdt = norm_sym.replace("/USD", "/USDT")
                    self._cache[usdt] = (avg, time.time())

        # Map back to original symbols
        for orig_sym in symbols:
            if orig_sym not in results:
                norm = sym_map[orig_sym]
                with self._lock:
                    cached = self._cache.get(norm)
                    if cached:
                        results[orig_sym] = cached[0]

        return results
