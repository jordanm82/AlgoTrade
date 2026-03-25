# kalshi_mm/mm_inventory.py
"""Market maker inventory tracking, fee calculation, and position sizing."""
import math
import time
from dataclasses import dataclass

from kalshi_mm.mm_config import (
    RISK_BUDGET_PCT, MIN_CONTRACTS_PER_QUOTE, MAX_CONTRACTS_PER_ASSET,
    MAX_DAILY_LOSS_CENTS, MAX_WINDOW_LOSS_CENTS, MAX_EXIT_LOSS_CENTS,
    IDLE,
)


def calc_maker_fee_cents(contracts: int, price_cents: int) -> int:
    """Kalshi maker fee in cents, rounded up."""
    p = price_cents / 100
    fee_dollars = 0.0175 * contracts * p * (1 - p)
    return math.ceil(fee_dollars * 100)


def calc_round_trip_pnl(buy_cents: int, sell_cents: int, contracts: int) -> int:
    """Net P&L in cents for a completed round trip."""
    gross = (sell_cents - buy_cents) * contracts
    fee_buy = calc_maker_fee_cents(contracts, buy_cents)
    fee_sell = calc_maker_fee_cents(contracts, sell_cents)
    return gross - fee_buy - fee_sell


def compute_contracts(balance_cents: int, entry_price_cents: int) -> int | None:
    """Compute contracts to quote. Returns None if insufficient budget."""
    risk_budget = int(balance_cents * RISK_BUDGET_PCT)
    contracts = risk_budget // entry_price_cents
    contracts = min(contracts, MAX_CONTRACTS_PER_ASSET)
    if contracts < MIN_CONTRACTS_PER_QUOTE:
        return None
    return contracts


@dataclass
class MMInventory:
    """Per-asset market making state."""
    asset: str
    yes_held: int = 0
    entry_price_cents: int = 0
    pending_bid_id: str | None = None
    pending_ask_id: str | None = None
    bid_price_cents: int = 0
    ask_price_cents: int = 0
    window_pnl_cents: int = 0
    window_round_trips: int = 0
    day_pnl_cents: int = 0
    inventory_since: float = 0.0
    state: str = IDLE
    market_ticker: str = ""
    expiry_ts: float = 0.0
    exit_attempt_count: int = 0
    last_exit_attempt_ts: float = 0.0

    def has_inventory(self) -> bool:
        return self.yes_held > 0

    def inventory_age_seconds(self) -> float:
        if self.inventory_since == 0:
            return 0.0
        return time.time() - self.inventory_since

    def minutes_to_expiry(self) -> float:
        if self.expiry_ts == 0:
            return 999.0
        return (self.expiry_ts - time.time()) / 60

    def record_buy_fill(self, contracts: int, price_cents: int, order_id: str | None = None):
        self.yes_held = contracts
        self.entry_price_cents = price_cents
        self.inventory_since = time.time()
        self.pending_bid_id = None

    def record_sell_fill(self, sell_price_cents: int):
        pnl = calc_round_trip_pnl(self.entry_price_cents, sell_price_cents, self.yes_held)
        self.window_pnl_cents += pnl
        self.day_pnl_cents += pnl
        self.window_round_trips += 1
        rt_info = {
            "buy_cents": self.entry_price_cents,
            "sell_cents": sell_price_cents,
            "contracts": self.yes_held,
            "pnl_cents": pnl,
        }
        self.yes_held = 0
        self.entry_price_cents = 0
        self.inventory_since = 0.0
        self.pending_ask_id = None
        self.exit_attempt_count = 0
        self.last_exit_attempt_ts = 0.0
        return rt_info

    def reset_window(self):
        self.window_pnl_cents = 0
        self.window_round_trips = 0
        self.market_ticker = ""
        self.expiry_ts = 0.0
        self.exit_attempt_count = 0
        self.last_exit_attempt_ts = 0.0

    def is_daily_loss_hit(self) -> bool:
        return self.day_pnl_cents <= -MAX_DAILY_LOSS_CENTS

    def is_window_loss_hit(self) -> bool:
        return self.window_pnl_cents <= -MAX_WINDOW_LOSS_CENTS
