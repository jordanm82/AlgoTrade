# kalshi_mm/mm_strategy.py
"""Market maker strategy: orderbook parsing, mid price, spread sizing, quote generation."""
from kalshi_mm.mm_config import SPREAD_MIN_CENTS, SPREAD_MAX_CENTS, VPIN_SAFE, VPIN_CAUTION


def compute_mid_cents(orderbook: dict) -> int | None:
    """Compute mid price in cents from Kalshi orderbook.
    Kalshi returns yes_dollars and no_dollars as [[price_str, count_str], ...].
    Only bids are shown. YES ask = 100 - best NO bid.
    """
    ob = orderbook.get("orderbook_fp", orderbook)
    yes_bids = ob.get("yes_dollars", [])
    no_bids = ob.get("no_dollars", [])

    if not yes_bids or not no_bids:
        return None

    best_yes_bid = round(float(yes_bids[0][0]) * 100)
    best_no_bid = round(float(no_bids[0][0]) * 100)
    implied_yes_ask = 100 - best_no_bid

    mid = (best_yes_bid + implied_yes_ask) // 2
    return mid


def compute_spread_cents(vpin: float) -> int | None:
    """Dynamic spread based on VPIN. Returns None if should go dark."""
    if vpin < VPIN_SAFE:
        return SPREAD_MIN_CENTS
    elif vpin < VPIN_CAUTION:
        # Linear scale from 3c to 4c across the caution range
        ratio = (vpin - VPIN_SAFE) / (VPIN_CAUTION - VPIN_SAFE)
        return 3 + round(ratio)
    return None


def compute_bid_cents(mid_cents: int, spread_cents: int) -> int | None:
    """Compute bid price. Returns None if out of bounds."""
    bid = mid_cents - spread_cents // 2
    if not (1 <= bid <= 99):
        return None
    return bid


def compute_ask_cents(entry_price_cents: int, spread_cents: int) -> int | None:
    """Compute ask price from entry + spread. Returns None if out of bounds."""
    ask = entry_price_cents + spread_cents
    if not (1 <= ask <= 99):
        return None
    return ask


def parse_ob_total_volume(levels: list) -> int:
    """Sum contract quantities from orderbook levels.
    Count strings may be float-formatted (e.g., '10001.00'), so parse via float first.
    """
    total = 0
    for level in levels:
        total += int(float(level[1]))
    return total
