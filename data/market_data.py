# data/market_data.py
"""Real-time market microstructure data -- order book, trade flow, cross-asset signals."""
import time
import ccxt

exchange = ccxt.binanceus({"enableRateLimit": True})

TRACKED_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]


def get_order_book_imbalance(symbol: str, depth: int = 20) -> dict:
    """Fetch order book and compute bid/ask imbalance.

    Returns:
        imbalance: -1 to +1 (-1 = all sell pressure, +1 = all buy pressure)
        bid_volume: total bid volume
        ask_volume: total ask volume
        spread_pct: spread as % of mid price
        wall_ratio: largest single level volume / average volume (detects walls)
    """
    book = exchange.fetch_order_book(symbol, limit=depth)
    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return {"imbalance": 0, "bid_volume": 0, "ask_volume": 0, "spread_pct": 0, "wall_ratio": 1}

    bid_vol = sum(b[1] for b in bids)
    ask_vol = sum(a[1] for a in asks)
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0

    mid = (bids[0][0] + asks[0][0]) / 2
    spread_pct = (asks[0][0] - bids[0][0]) / mid * 100 if mid > 0 else 0

    # Detect walls: largest single level vs average
    all_volumes = [b[1] for b in bids] + [a[1] for a in asks]
    avg_vol = sum(all_volumes) / len(all_volumes) if all_volumes else 1
    max_vol = max(all_volumes) if all_volumes else 1
    wall_ratio = max_vol / avg_vol if avg_vol > 0 else 1

    return {
        "imbalance": round(imbalance, 4),
        "bid_volume": round(bid_vol, 6),
        "ask_volume": round(ask_vol, 6),
        "spread_pct": round(spread_pct, 4),
        "wall_ratio": round(wall_ratio, 2),
    }


def get_trade_flow(symbol: str, limit: int = 200) -> dict:
    """Fetch recent trades and compute net buying/selling pressure.

    Returns:
        net_flow: positive = net buying, negative = net selling
        buy_ratio: fraction of trades that are buys (0-1)
        buy_volume: total buy volume
        sell_volume: total sell volume
        large_trade_bias: direction of trades > 2x average size
    """
    trades = exchange.fetch_trades(symbol, limit=limit)
    if not trades:
        return {"net_flow": 0, "buy_ratio": 0.5, "buy_volume": 0, "sell_volume": 0,
                "large_trade_bias": 0}

    buys = [t for t in trades if t.get("side") == "buy"]
    sells = [t for t in trades if t.get("side") == "sell"]

    buy_vol = sum(t["amount"] for t in buys)
    sell_vol = sum(t["amount"] for t in sells)
    total_vol = buy_vol + sell_vol

    net_flow = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
    buy_ratio = len(buys) / len(trades) if trades else 0.5

    # Large trade bias: trades > 2x average
    avg_size = total_vol / len(trades) if trades else 0
    large_buys = sum(t["amount"] for t in buys if t["amount"] > avg_size * 2)
    large_sells = sum(t["amount"] for t in sells if t["amount"] > avg_size * 2)
    large_total = large_buys + large_sells
    large_trade_bias = (large_buys - large_sells) / large_total if large_total > 0 else 0

    return {
        "net_flow": round(net_flow, 4),
        "buy_ratio": round(buy_ratio, 4),
        "buy_volume": round(buy_vol, 6),
        "sell_volume": round(sell_vol, 6),
        "large_trade_bias": round(large_trade_bias, 4),
    }


def get_cross_asset_momentum() -> dict:
    """Get short-term momentum across BTC/ETH/SOL/XRP.

    BTC leads alts. If BTC is moving, alts will follow.
    Returns per-asset % change and the "leader" signal.
    """
    results = {}
    for sym in TRACKED_PAIRS:
        try:
            ticker = exchange.fetch_ticker(sym)
            results[sym] = {
                "price": ticker.get("last", 0),
                "change_pct": ticker.get("percentage", 0) or 0,
                "volume_24h": ticker.get("quoteVolume", 0) or 0,
            }
        except Exception:
            results[sym] = {"price": 0, "change_pct": 0, "volume_24h": 0}
        time.sleep(0.15)

    # BTC as leader: is BTC diverging from alts?
    btc_change = results.get("BTC/USDT", {}).get("change_pct", 0)
    alt_changes = [results[s]["change_pct"] for s in TRACKED_PAIRS if s != "BTC/USDT" and results[s]["change_pct"] != 0]
    avg_alt_change = sum(alt_changes) / len(alt_changes) if alt_changes else 0

    results["btc_leads"] = btc_change - avg_alt_change  # positive = BTC outperforming alts
    results["market_direction"] = btc_change  # overall market direction

    return results


def get_all_signals(symbol: str) -> dict:
    """Fetch all leading indicators for a single pair."""
    ob = get_order_book_imbalance(symbol)
    tf = get_trade_flow(symbol)
    return {
        "order_book": ob,
        "trade_flow": tf,
    }
