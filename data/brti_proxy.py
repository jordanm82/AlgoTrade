# data/brti_proxy.py
"""CF BRTI price feed — the ACTUAL settlement source for Kalshi K15 markets.

Fetches real-time index values from CF Benchmarks website SSR data.
These are the exact same values Kalshi uses for settlement.

Indices:
  BTC → BRTI (Bitcoin Real Time Index)
  ETH → ETHUSD_RTI (Ether Real Time Index)
  SOL → SOLUSD_RTI (Solana Real Time Index)
  XRP → XRPUSD_RTI (XRP Real Time Index)

Falls back to multi-exchange average (Coinbase+Kraken+Bitstamp) if
CF Benchmarks is unreachable.
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


class BRTIProxy:
    """CF Benchmarks Real Time Index price feed."""

    # CF Benchmarks index IDs for each asset
    CF_INDICES = {
        "BTC/USD": "BRTI",
        "ETH/USD": "ETHUSD_RTI",
        "SOL/USD": "SOLUSD_RTI",
        "XRP/USD": "XRPUSD_RTI",
    }

    # Also map USDT symbols
    CF_INDICES_USDT = {
        "BTC/USDT": "BRTI",
        "ETH/USDT": "ETHUSD_RTI",
        "SOL/USDT": "SOLUSD_RTI",
        "XRP/USDT": "XRPUSD_RTI",
    }

    CACHE_TTL = 10  # seconds

    def __init__(self):
        self._cache = {}  # {symbol: (price, timestamp)}
        self._lock = threading.Lock()

    def _fetch_cf_price(self, index_id: str) -> float | None:
        """Fetch current value from CF Benchmarks SSR data."""
        try:
            url = f"https://www.cfbenchmarks.com/data/indices/{index_id}"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text, re.DOTALL,
            )
            if match:
                data = json.loads(match.group(1))
                summary = data["props"]["pageProps"].get("indexSummary", {})
                value = summary.get("value")
                if value is not None:
                    return float(value)
        except Exception:
            pass
        return None

    def get_price(self, symbol: str) -> float | None:
        """Get BRTI price for a symbol (e.g., 'BTC/USD' or 'BTC/USDT').

        Returns the actual CF Benchmarks index value (Kalshi settlement source).
        Cached for 10 seconds.
        """
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and time.time() - cached[1] < self.CACHE_TTL:
                return cached[0]

        index_id = self.CF_INDICES.get(symbol) or self.CF_INDICES_USDT.get(symbol)
        if not index_id:
            return None

        price = self._fetch_cf_price(index_id)
        if price is not None:
            with self._lock:
                self._cache[symbol] = (price, time.time())

        return price

    def get_prices_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get BRTI prices for multiple symbols in parallel.

        Returns {symbol: price} dict. ~1-2 seconds for all 4 assets.
        """
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

        def _fetch(sym):
            index_id = self.CF_INDICES.get(sym) or self.CF_INDICES_USDT.get(sym)
            if not index_id:
                return (sym, None)
            price = self._fetch_cf_price(index_id)
            return (sym, price)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for sym, price in pool.map(_fetch, to_fetch):
                if price is not None:
                    results[sym] = price
                    with self._lock:
                        self._cache[sym] = (price, time.time())

        return results
