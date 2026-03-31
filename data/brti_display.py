# data/brti_display.py
"""CF BRTI price for DASHBOARD DISPLAY ONLY.

Scrapes CF Benchmarks website for the real-time index values.
NOT used for model predictions — those use Coinbase (training parity).

This is for showing the user the actual settlement price on the dashboard
so they can see if they're winning or losing relative to Kalshi's source.

May be 30-60 seconds stale — fine for display, NOT for trading decisions.
"""
import json
import re
import threading
import time

import requests

_CF_INDICES = {
    "BTC": "BRTI",
    "ETH": "ETHUSD_RTI",
    "SOL": "SOLUSD_RTI",
    "XRP": "XRPUSD_RTI",
}

_cache = {}
_lock = threading.Lock()
_CACHE_TTL = 30  # seconds — display only, can be slightly stale


def get_brti_price(asset: str) -> float | None:
    """Get CF BRTI price for display. Cached 30 seconds.

    Args:
        asset: "BTC", "ETH", "SOL", or "XRP"

    Returns:
        Price as float, or None if unavailable.
    """
    with _lock:
        cached = _cache.get(asset)
        if cached and time.time() - cached[1] < _CACHE_TTL:
            return cached[0]

    index_id = _CF_INDICES.get(asset)
    if not index_id:
        return None

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
                price = float(value)
                with _lock:
                    _cache[asset] = (price, time.time())
                return price
    except Exception:
        pass

    return None
