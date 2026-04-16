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
from datetime import datetime, timedelta, timezone

import ccxt


class BRTIProxy:
    """Multi-exchange real-time price averaging (BRTI constituent exchanges)."""

    CACHE_TTL = 5  # seconds — refresh every 5s for dashboard + model

    def __init__(self):
        self._exchanges = {}
        self._cache = {}  # {symbol: (price, timestamp)}
        self._lock = threading.Lock()

    def _get_exchange(self, name: str):
        if name not in self._exchanges:
            # Only Coinbase + Bitstamp — matches training data exactly
            # DO NOT add Kraken — model was trained on 2-exchange average
            constructors = {
                "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
                "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
            }
            if name in constructors:
                self._exchanges[name] = constructors[name]()
        return self._exchanges.get(name)

    def get_price(self, symbol: str) -> float | None:
        """Get real-time averaged price for a symbol (e.g., 'BTC/USD').

        Averages Coinbase + Bitstamp last trade prices (matches training).
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

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_fetch, ["coinbase", "bitstamp"]))

        prices = [p for p in results if p is not None]
        if len(prices) != 2:
            return None

        avg = sum(prices) / len(prices)

        with self._lock:
            self._cache[sym] = (avg, time.time())
            # Also cache the USDT variant
            usdt_sym = sym.replace("/USD", "/USDT")
            self._cache[usdt_sym] = (avg, time.time())

        return avg

    def get_5m_candle_open(self, symbol: str) -> float | None:
        """Get the OPEN price of the current 5m candle (Coinbase+Bitstamp averaged).

        Training computes M10 distance as: get_avg_price(cb_5m, bs_5m, min10, "open")
        which is the 5m candle open. Live MUST use the same price, not real-time spot.
        """
        sym = symbol.replace("/USDT", "/USD")

        def _fetch_ohlcv(name):
            try:
                ex = self._get_exchange(name)
                if ex is None:
                    return None
                # Fetch last 1 candle of 5m timeframe
                candles = ex.fetch_ohlcv(sym, "5m", limit=1)
                if candles and len(candles) > 0:
                    return candles[-1][1]  # [1] = open price
                return None
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(_fetch_ohlcv, ["coinbase", "bitstamp"]))

        opens = [p for p in results if p is not None]
        if len(opens) != 2:
            return None
        return sum(opens) / len(opens)

    def get_5m_open_at(self, symbol: str, candle_start: datetime) -> float | None:
        """Get the OPEN price for a specific 5m candle timestamp.

        Strict parity path:
        - Requires BOTH Coinbase and Bitstamp candle opens at `candle_start`
        - Returns None if either exchange is missing that exact 5m candle
        """
        sym = symbol.replace("/USDT", "/USD")
        anchor = candle_start if candle_start.tzinfo else candle_start.replace(tzinfo=timezone.utc)
        anchor_ms = int(anchor.timestamp() * 1000)

        def _fetch_open(name: str):
            try:
                ex = self._get_exchange(name)
                if ex is None:
                    return None
                candles = ex.fetch_ohlcv(sym, "5m", limit=12)
                if not candles:
                    return None
                by_ms = {int(c[0]): c for c in candles}
                c = by_ms.get(anchor_ms)
                if c is None:
                    return None
                return float(c[1])  # open
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            cb_open, bs_open = list(pool.map(_fetch_open, ["coinbase", "bitstamp"]))

        if cb_open is None or bs_open is None:
            return None
        return (cb_open + bs_open) / 2.0

    def get_5m_window_candles(self, symbol: str, window_start: datetime) -> dict[str, dict[str, float]] | None:
        """Return averaged 5m candles for minute-0/minute-5/minute-10 in a 15m window.

        Strict parity:
        - Requires BOTH Coinbase and Bitstamp candles for each required timestamp.
        - Returns None if any required candle is missing from either exchange.
        """
        sym = symbol.replace("/USDT", "/USD")
        ws = window_start if window_start.tzinfo else window_start.replace(tzinfo=timezone.utc)
        required = [ws, ws + timedelta(minutes=5), ws + timedelta(minutes=10)]
        required_ms = {int(dt.timestamp() * 1000): dt for dt in required}

        def _fetch_ohlcv(name: str):
            try:
                ex = self._get_exchange(name)
                if ex is None:
                    return None
                # Cover recent history around the current 15m window.
                candles = ex.fetch_ohlcv(sym, "5m", limit=12)
                if not candles:
                    return None
                by_ms = {int(c[0]): c for c in candles}
                picked = {}
                for ts_ms, dt in required_ms.items():
                    c = by_ms.get(ts_ms)
                    if c is None:
                        return None
                    picked[dt] = {
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    }
                return picked
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            cb, bs = list(pool.map(_fetch_ohlcv, ["coinbase", "bitstamp"]))

        if cb is None or bs is None:
            return None

        averaged = {}
        labels = ["minute_0", "minute_5", "minute_10"]
        for label, dt in zip(labels, required):
            c1 = cb.get(dt)
            c2 = bs.get(dt)
            if c1 is None or c2 is None:
                return None
            averaged[label] = {
                "open": (c1["open"] + c2["open"]) / 2.0,
                "high": (c1["high"] + c2["high"]) / 2.0,
                "low": (c1["low"] + c2["low"]) / 2.0,
                "close": (c1["close"] + c2["close"]) / 2.0,
                "volume": (c1["volume"] + c2["volume"]) / 2.0,
            }
        return averaged

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

        tasks = [(n, s) for s in to_fetch for n in ["coinbase", "bitstamp"]]
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
                if len(prices) == 2:
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
