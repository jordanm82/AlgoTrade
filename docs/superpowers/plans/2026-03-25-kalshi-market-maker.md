# Kalshi 15m Crypto Market Maker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone market making bot for Kalshi 15m crypto UP/DOWN markets that captures bid/ask spread on YES contracts with a VPIN-based kill switch.

**Architecture:** Sequential bid-then-ask market maker with per-asset state machines (BTC, ETH). Posts resting BUY YES limit, waits for fill, posts resting SELL YES limit at entry + spread. VPIN computed from BinanceUS spot trade flow detects toxic order flow and triggers kill switch. Live ASCII dashboard renders state every 3 seconds.

**Tech Stack:** Python, existing `exchange/kalshi.py` client (extended), `data/market_data.py` for VPIN, MCP server integration.

**Spec:** `docs/superpowers/specs/2026-03-25-kalshi-market-maker-design.md`

---

## ERRATA — Required Fixes (from plan review)

**Read these BEFORE implementing each task. Apply these fixes during implementation.**

### Critical Fixes (Task 6 — mm_daemon.py)

**Fix C1: Handle ask-cancel-as-fill race condition.** In `_tick_quoting_ask()`, after every `_cancel_pending_ask()` call (hard cutoff, VPIN kill, inventory timeout), check the return value. If it returns `{"status": "filled"}`, the ask was already filled — call `self.inv.record_sell_fill(self.inv.ask_price_cents)` and transition to QUOTING_BID or IDLE (not EXITING). Pattern:
```python
result = self._cancel_pending_ask()
if result and result.get("status") == "filled":
    self.inv.record_sell_fill(self.inv.ask_price_cents)
    self.inv.state = IDLE
    return
self.inv.state = EXITING
```

**Fix C2: Capture entry_price before record_sell_fill() in _exit_at_market().** `record_sell_fill()` zeroes `entry_price_cents`. Save it first:
```python
entry = self.inv.entry_price_cents
exit_price = max(1, entry - MAX_EXIT_LOSS_CENTS)
# ... sell logic ...
self.inv.record_sell_fill(exit_price)
self._log_trade("FORCED_EXIT", exit_price, contracts)
```

**Fix C3: In _handle_bid_fill_from_cancel(), query actual fill count in live mode.** Don't recompute from balance — the balance was already debited. In live mode, call `get_order_status()` on the original bid order ID to get actual fill count and price. Only use `compute_contracts()` in dry-run mode.

### Important Fixes

**Fix I1: Add volatility spike detection to state machine.** In `_compute_vpin()`, also fetch spot price via `DataFetcher.ticker()` and call `self.kill_switch.record_price(price)`. In each quoting state, check `self.kill_switch.volatility_spike()` alongside VPIN checks — if True, cancel and go DARK.

**Fix I2: Check window loss limit.** At the top of `_tick_quoting_bid()` and `_tick_quoting_ask()`, add:
```python
if self.inv.is_window_loss_hit():
    self._cancel_pending_bid()  # or ask
    self.inv.state = IDLE
    return
```

**Fix I3: Track consecutive API errors.** Add `_consecutive_api_errors: int = 0` to `MMAssetRunner.__init__`. On successful API call, reset to 0. On exception, increment. If >= 2, cancel all and go DARK.

**Fix I4: Track exit origin (kill switch vs timeout vs cutoff).** Add `_exit_from_kill_switch: bool = False` to `MMAssetRunner`. Set to True when transitioning to EXITING from VPIN kill switch. In `_tick_exiting()`, after exit completes: if `_exit_from_kill_switch`, transition to DARK instead of QUOTING_BID.

**Fix I5: Progressive exit pricing.** In `_tick_exiting()`, compute exit price as `entry - (exit_attempt_count + 1)` cents (progressively lower), clamped to the `MAX_EXIT_LOSS_CENTS` floor. Don't re-read orderbook for exit pricing — that can give higher prices which defeats the purpose.

**Fix I6: Write persistence files.** In the main loop, after rendering dashboard:
- Append completed round trips to `data/store/mm_trades.csv` (columns: time, asset, action, buy_cents, sell_cents, contracts, pnl_cents, fees_cents)
- Write `data/store/mm_session.json` with day_pnl, total_rts, start_time, forced_exits
- Append VPIN readings to `data/store/mm_vpin.csv` (columns: time, asset, spot_vpin, kalshi_heuristic, blended)

**Fix I7: Add VPIN_WINDOW_SECONDS to mm_config.py** (Task 1):
```python
VPIN_WINDOW_SECONDS = 120  # rolling 2-min window
```

**Fix I8: Verify orderbook depth in DISCOVERING.** In `_tick_discovering()`, after reading the orderbook, check that both yes_bids and no_bids have entries. If empty, stay in IDLE.

---

## File Map

| File | Responsibility | New/Modify |
|------|---------------|------------|
| `kalshi_mm/mm_config.py` | All constants and thresholds | Create |
| `kalshi_mm/mm_inventory.py` | Fee calc, P&L, position sizing, MMInventory dataclass | Create |
| `kalshi_mm/mm_vpin.py` | VPIN computation, kill switch logic, volatility detection | Create |
| `kalshi_mm/mm_strategy.py` | Mid price calc, spread sizing, quote generation, orderbook parsing | Create |
| `kalshi_mm/mm_daemon.py` | Main loop, state machine, dashboard, market discovery, dry-run | Create |
| `kalshi_mm/__init__.py` | Package init | Create |
| `exchange/kalshi.py:153-186` | Add `action` param, `get_order_status()`, `cancel_order_safe()` | Modify |
| `mcp_server.py` | Add 3 MM tools (mm_start, mm_stop, mm_status) | Modify |
| `tests/test_mm_inventory.py` | Fee math, P&L, position sizing tests | Create |
| `tests/test_mm_vpin.py` | VPIN computation, kill switch trigger tests | Create |
| `tests/test_mm_strategy.py` | Orderbook parsing, mid calc, spread sizing tests | Create |
| `tests/test_mm_daemon.py` | State machine integration tests | Create |
| `tests/test_kalshi_extended.py` | Tests for new kalshi.py methods | Create |

---

### Task 1: Configuration Module

**Files:**
- Create: `kalshi_mm/__init__.py`
- Create: `kalshi_mm/mm_config.py`

- [ ] **Step 1: Create package init**

```python
# kalshi_mm/__init__.py
```

- [ ] **Step 2: Write mm_config.py with all constants from spec**

```python
# kalshi_mm/mm_config.py
"""Market maker configuration — all thresholds and parameters."""

# Assets: binance symbol → kalshi series ticker
MM_ASSETS = {
    "BTC/USDT": "KXBTC15M",
    "ETH/USDT": "KXETH15M",
}

# Spread (in cents)
SPREAD_MIN_CENTS = 2
SPREAD_DEFAULT_CENTS = 3
SPREAD_MAX_CENTS = 4

# VPIN thresholds
VPIN_SAFE = 0.3
VPIN_CAUTION = 0.5
VPIN_SPOT_WEIGHT = 0.7
VPIN_KALSHI_WEIGHT = 0.3

# Timing (seconds unless noted)
POLL_INTERVAL = 3
QUOTE_WINDOW_MIN_MINUTES = 5
QUOTE_WINDOW_MAX_MINUTES = 12
HARD_CUTOFF_MINUTES = 2
DISCOVERY_RETRY_SECONDS = 10
FORCED_EXIT_RETRY_SECONDS = 10

# Inventory
MIN_CONTRACTS_PER_QUOTE = 10
MAX_CONTRACTS_PER_ASSET = 50
INVENTORY_TIMEOUT_SECONDS = 300
RISK_BUDGET_PCT = 0.05

# Risk
MAX_DAILY_LOSS_CENTS = 2000
MAX_WINDOW_LOSS_CENTS = 1000
VOLATILITY_SPIKE_PCT = 0.005
MAX_EXIT_LOSS_CENTS = 5

# Re-quote
REQUOTE_DRIFT_CENTS = 1

# States
IDLE = "IDLE"
DISCOVERING = "DISCOVERING"
QUOTING_BID = "QUOTING_BID"
QUOTING_ASK = "QUOTING_ASK"
EXITING = "EXITING"
DARK = "DARK"
```

- [ ] **Step 3: Commit**

```bash
git add kalshi_mm/__init__.py kalshi_mm/mm_config.py
git commit -m "feat(mm): add kalshi_mm package with configuration constants"
```

---

### Task 2: Extend exchange/kalshi.py

**Files:**
- Modify: `exchange/kalshi.py:153-186`
- Create: `tests/test_kalshi_extended.py`

- [ ] **Step 1: Write failing tests for the 3 new methods**

```python
# tests/test_kalshi_extended.py
"""Tests for new KalshiClient methods: action param, get_order_status, cancel_order_safe."""
import pytest
from unittest.mock import patch, MagicMock
from exchange.kalshi import KalshiClient
import requests


def _make_client():
    """Create a KalshiClient with mocked key."""
    client = KalshiClient.__new__(KalshiClient)
    client.api_key_id = "test"
    client.base_url = "https://demo"
    client._private_key = MagicMock()
    client._private_key.sign.return_value = b"sig"
    return client


class TestPlaceOrderAction:
    def test_default_action_is_buy(self):
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "yes", 10, price_cents=48, order_type="limit")
            data = mock.call_args[0][1]
            assert data["action"] == "buy"

    def test_sell_action(self):
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "yes", 10, price_cents=52,
                               order_type="limit", action="sell")
            data = mock.call_args[0][1]
            assert data["action"] == "sell"
            assert data["side"] == "yes"
            assert data["yes_price"] == 52

    def test_existing_callers_unaffected(self):
        """Callers that don't pass action= still get 'buy'."""
        client = _make_client()
        with patch.object(client, "_post", return_value={"order": {}}) as mock:
            client.place_order("TICK", "no", 5, price_cents=30, order_type="limit")
            data = mock.call_args[0][1]
            assert data["action"] == "buy"
            assert data["no_price"] == 30


class TestGetOrderStatus:
    def test_returns_order_data(self):
        client = _make_client()
        with patch.object(client, "_get", return_value={"order": {"status": "resting"}}) as mock:
            result = client.get_order_status("order-123")
            mock.assert_called_once_with("/trade-api/v2/portfolio/orders/order-123")
            assert result["order"]["status"] == "resting"


class TestCancelOrderSafe:
    def test_successful_cancel(self):
        client = _make_client()
        with patch.object(client, "_delete", return_value={"order": {"status": "cancelled"}}):
            result = client.cancel_order_safe("order-123")
            assert result["order"]["status"] == "cancelled"

    def test_404_returns_filled(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 404
        http_err = requests.HTTPError(response=resp)
        with patch.object(client, "_delete", side_effect=http_err):
            result = client.cancel_order_safe("order-123")
            assert result["status"] == "filled"

    def test_other_http_error_reraises(self):
        client = _make_client()
        resp = MagicMock()
        resp.status_code = 500
        http_err = requests.HTTPError(response=resp)
        with patch.object(client, "_delete", side_effect=http_err):
            with pytest.raises(requests.HTTPError):
                client.cancel_order_safe("order-123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kalshi_extended.py -v`
Expected: FAIL — `place_order` has no `action` param, `get_order_status` and `cancel_order_safe` don't exist.

- [ ] **Step 3: Implement the changes in exchange/kalshi.py**

In `exchange/kalshi.py`, modify `place_order` signature at line 153 to add `action="buy"` parameter. Change line 163 from hardcoded `"action": "buy"` to `"action": action`.

Then add two new methods after `cancel_order` (after line 186):

```python
    def get_order_status(self, order_id: str) -> dict:
        """Get status of a specific order."""
        return self._get(f"/trade-api/v2/portfolio/orders/{order_id}")

    def cancel_order_safe(self, order_id: str) -> dict:
        """Cancel an order. Returns filled status on 404 (order already filled)."""
        try:
            return self._delete(f"/trade-api/v2/portfolio/orders/{order_id}")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return {"status": "filled"}
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kalshi_extended.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Run existing kalshi tests to verify no regression**

Run: `pytest tests/test_kalshi.py -v`
Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add exchange/kalshi.py tests/test_kalshi_extended.py
git commit -m "feat(kalshi): add sell action, get_order_status, cancel_order_safe"
```

---

### Task 3: Inventory Module — Fee Math & P&L

**Files:**
- Create: `kalshi_mm/mm_inventory.py`
- Create: `tests/test_mm_inventory.py`

- [ ] **Step 1: Write failing tests for fee calculation**

```python
# tests/test_mm_inventory.py
"""Tests for MM inventory: fee math, P&L, position sizing."""
import math
import pytest


class TestCalcMakerFeeCents:
    def test_1_contract_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(1, 50) == 1  # ceil(0.0175*1*0.5*0.5*100)=ceil(0.4375)=1

    def test_5_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(5, 50) == 3  # ceil(0.0175*5*0.5*0.5*100)=ceil(2.1875)=3

    def test_10_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 50) == 5  # ceil(4.375)=5

    def test_20_contracts_at_50c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(20, 50) == 9  # ceil(8.75)=9

    def test_10_contracts_at_30c(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 30) == 4  # ceil(0.0175*10*0.3*0.7*100)=ceil(3.675)=4

    def test_at_extreme_low_price(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 5) == 1  # ceil(0.0175*10*0.05*0.95*100)=ceil(0.83125)=1

    def test_at_extreme_high_price(self):
        from kalshi_mm.mm_inventory import calc_maker_fee_cents
        assert calc_maker_fee_cents(10, 95) == 1  # same as 5c (symmetric)


class TestCalcRoundTripPnl:
    def test_profit_at_3c_spread_10_contracts_50c(self):
        from kalshi_mm.mm_inventory import calc_round_trip_pnl
        pnl = calc_round_trip_pnl(buy_cents=49, sell_cents=52, contracts=10)
        # gross = 3*10 = 30c
        # fee_buy = ceil(0.0175*10*0.49*0.51*100) = ceil(4.37..) = 5c
        # fee_sell = ceil(0.0175*10*0.52*0.48*100) = ceil(4.368) = 5c
        # net = 30 - 5 - 5 = 20c
        assert pnl == 20

    def test_loss_when_forced_exit(self):
        from kalshi_mm.mm_inventory import calc_round_trip_pnl
        pnl = calc_round_trip_pnl(buy_cents=50, sell_cents=47, contracts=10)
        assert pnl < 0


class TestComputeContracts:
    def test_normal_balance(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $100 balance = 10000c, 5% = 500c, at 50c = 10 contracts
        assert compute_contracts(10000, 50) == 10

    def test_large_balance_capped_at_max(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $1000 = 100000c, 5% = 5000c, at 50c = 100, capped at 50
        assert compute_contracts(100000, 50) == 50

    def test_small_balance_returns_none(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # $5 = 500c, 5% = 25c, at 50c = 0 contracts < 10 min
        assert compute_contracts(500, 50) is None

    def test_borderline_budget(self):
        from kalshi_mm.mm_inventory import compute_contracts
        # Need exactly 10*50=500c budget. 500c/0.05=10000c balance
        assert compute_contracts(10000, 50) == 10
        assert compute_contracts(9999, 50) is None  # 499c budget, only 9 contracts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mm_inventory.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write mm_inventory.py**

```python
# kalshi_mm/mm_inventory.py
"""Market maker inventory tracking, fee calculation, and position sizing."""
import math
import time
from dataclasses import dataclass, field

from kalshi_mm.mm_config import (
    RISK_BUDGET_PCT, MIN_CONTRACTS_PER_QUOTE, MAX_CONTRACTS_PER_ASSET,
    MAX_DAILY_LOSS_CENTS, MAX_WINDOW_LOSS_CENTS, MAX_EXIT_LOSS_CENTS,
    IDLE,
)


def calc_maker_fee_cents(contracts: int, price_cents: int) -> int:
    """Kalshi maker fee in cents, rounded up.

    Formula: ceil(0.0175 * C * P * (1-P) * 100) where P = price_cents/100.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mm_inventory.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kalshi_mm/mm_inventory.py tests/test_mm_inventory.py
git commit -m "feat(mm): add inventory module with fee math, P&L, position sizing"
```

---

### Task 4: VPIN & Kill Switch Module

**Files:**
- Create: `kalshi_mm/mm_vpin.py`
- Create: `tests/test_mm_vpin.py`

- [ ] **Step 1: Write failing tests for VPIN computation and kill switch**

```python
# tests/test_mm_vpin.py
"""Tests for VPIN computation and kill switch logic."""
import pytest
import time


class TestComputeSpotVpin:
    def test_balanced_flow_low_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": 0.05, "buy_ratio": 0.51}
        assert compute_spot_vpin(flow) == pytest.approx(0.05)

    def test_heavy_buying_high_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": 0.7, "buy_ratio": 0.8}
        assert compute_spot_vpin(flow) == pytest.approx(0.7)

    def test_heavy_selling_high_vpin(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        flow = {"net_flow": -0.6, "buy_ratio": 0.2}
        assert compute_spot_vpin(flow) == pytest.approx(0.6)

    def test_empty_flow_returns_zero(self):
        from kalshi_mm.mm_vpin import compute_spot_vpin
        assert compute_spot_vpin({}) == 0.0


class TestComputeKalshiObHeuristic:
    def test_no_change_returns_zero(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        assert compute_kalshi_ob_heuristic(100, 100, 100, 100) == 0.0

    def test_one_sided_yes_increase(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        result = compute_kalshi_ob_heuristic(100, 200, 50, 50)
        assert result > 0.5  # large one-sided shift

    def test_symmetric_change(self):
        from kalshi_mm.mm_vpin import compute_kalshi_ob_heuristic
        result = compute_kalshi_ob_heuristic(100, 150, 100, 150)
        assert result == pytest.approx(0.0)  # both sides moved equally


class TestBlendedVpin:
    def test_blending_weights(self):
        from kalshi_mm.mm_vpin import compute_blended_vpin
        # spot=0.4 * 0.7 + kalshi=0.2 * 0.3 = 0.28 + 0.06 = 0.34
        assert compute_blended_vpin(0.4, 0.2) == pytest.approx(0.34)


class TestKillSwitch:
    def test_safe_vpin(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        assert ks.should_go_dark(0.2) is False

    def test_toxic_vpin(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        assert ks.should_go_dark(0.6) is True

    def test_volatility_spike(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_price(100.0)
        ks.record_price(100.6)  # 0.6% move
        assert ks.volatility_spike() is True

    def test_no_volatility_spike(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_price(100.0)
        ks.record_price(100.1)  # 0.1% move
        assert ks.volatility_spike() is False

    def test_rising_vpin_detection(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_vpin(0.2)
        ks.record_vpin(0.3)
        ks.record_vpin(0.4)
        assert ks.vpin_rising() is True

    def test_stable_vpin_not_rising(self):
        from kalshi_mm.mm_vpin import KillSwitch
        ks = KillSwitch()
        ks.record_vpin(0.3)
        ks.record_vpin(0.3)
        ks.record_vpin(0.3)
        assert ks.vpin_rising() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mm_vpin.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write mm_vpin.py**

```python
# kalshi_mm/mm_vpin.py
"""VPIN computation and kill switch logic for market maker."""
from collections import deque

from kalshi_mm.mm_config import (
    VPIN_SAFE, VPIN_CAUTION, VPIN_SPOT_WEIGHT, VPIN_KALSHI_WEIGHT,
    VOLATILITY_SPIKE_PCT,
)


def compute_spot_vpin(trade_flow: dict) -> float:
    """Compute VPIN proxy from BinanceUS spot trade flow.

    Uses abs(net_flow) as the VPIN proxy — already computed as
    |buy_vol - sell_vol| / total_vol by data/market_data.get_trade_flow().
    """
    net_flow = trade_flow.get("net_flow", 0)
    return abs(net_flow)


def compute_kalshi_ob_heuristic(
    prev_yes_vol: float, curr_yes_vol: float,
    prev_no_vol: float, curr_no_vol: float,
) -> float:
    """Compute orderbook-change heuristic for Kalshi.

    Tracks total yes_bid_volume and no_bid_volume between polls.
    Large one-sided shifts suggest informed positioning.
    Returns 0-1 range.
    """
    delta_yes = curr_yes_vol - prev_yes_vol
    delta_no = curr_no_vol - prev_no_vol
    denom = abs(delta_yes) + abs(delta_no) + 1
    return abs(delta_yes - delta_no) / denom


def compute_blended_vpin(spot_vpin: float, kalshi_heuristic: float) -> float:
    """Blend spot VPIN and Kalshi OB heuristic."""
    return VPIN_SPOT_WEIGHT * spot_vpin + VPIN_KALSHI_WEIGHT * kalshi_heuristic


class KillSwitch:
    """Monitors flow toxicity and volatility for kill switch decisions."""

    def __init__(self):
        self._vpin_history: deque[float] = deque(maxlen=10)
        self._prices: deque[float] = deque(maxlen=2)

    def should_go_dark(self, vpin: float) -> bool:
        """Returns True if VPIN exceeds toxic threshold."""
        return vpin >= VPIN_CAUTION

    def get_spread_state(self, vpin: float) -> str:
        """Returns 'SAFE', 'CAUTION', or 'TOXIC'."""
        if vpin < VPIN_SAFE:
            return "SAFE"
        elif vpin < VPIN_CAUTION:
            return "CAUTION"
        return "TOXIC"

    def record_vpin(self, vpin: float):
        self._vpin_history.append(vpin)

    def vpin_rising(self) -> bool:
        """Returns True if VPIN has increased for 2+ consecutive readings."""
        if len(self._vpin_history) < 3:
            return False
        vals = list(self._vpin_history)
        return vals[-1] > vals[-2] > vals[-3]

    def record_price(self, price: float):
        self._prices.append(price)

    def volatility_spike(self) -> bool:
        """Returns True if price moved > VOLATILITY_SPIKE_PCT between last 2 readings."""
        if len(self._prices) < 2:
            return False
        old, new = self._prices[0], self._prices[1]
        if old == 0:
            return False
        return abs(new - old) / old > VOLATILITY_SPIKE_PCT
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mm_vpin.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kalshi_mm/mm_vpin.py tests/test_mm_vpin.py
git commit -m "feat(mm): add VPIN computation and kill switch module"
```

---

### Task 5: Strategy Module — Orderbook Parsing, Mid Price, Spread

**Files:**
- Create: `kalshi_mm/mm_strategy.py`
- Create: `tests/test_mm_strategy.py`

- [ ] **Step 1: Write failing tests for orderbook parsing and mid price**

```python
# tests/test_mm_strategy.py
"""Tests for MM strategy: orderbook parsing, mid calc, spread sizing, quotes."""
import pytest


class TestComputeMidCents:
    def test_basic_mid(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.4800", "10"]],
                "no_dollars": [["0.5000", "5"]],
            }
        }
        # best_yes_bid=48, best_no_bid=50, implied_yes_ask=50, mid=(48+50)//2=49
        assert compute_mid_cents(ob) == 49

    def test_asymmetric_book(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.4000", "20"]],
                "no_dollars": [["0.5500", "15"]],
            }
        }
        # best_yes_bid=40, best_no_bid=55, yes_ask=45, mid=(40+45)//2=42
        assert compute_mid_cents(ob) == 42

    def test_empty_yes_returns_none(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.50", "5"]]}}
        assert compute_mid_cents(ob) is None

    def test_empty_no_returns_none(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {"orderbook_fp": {"yes_dollars": [["0.48", "10"]], "no_dollars": []}}
        assert compute_mid_cents(ob) is None

    def test_round_not_truncate_29c(self):
        """Verify 0.29 converts to 29, not 28 (float precision)."""
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.2900", "10"]],
                "no_dollars": [["0.7100", "10"]],
            }
        }
        # yes_bid=29, no_bid=71, yes_ask=29, mid=(29+29)//2=29
        assert compute_mid_cents(ob) == 29

    def test_round_not_truncate_57c(self):
        from kalshi_mm.mm_strategy import compute_mid_cents
        ob = {
            "orderbook_fp": {
                "yes_dollars": [["0.5700", "10"]],
                "no_dollars": [["0.4300", "10"]],
            }
        }
        assert compute_mid_cents(ob) == 57


class TestComputeSpreadCents:
    def test_safe_vpin(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        assert compute_spread_cents(0.1) == 2

    def test_caution_low(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        assert compute_spread_cents(0.3) == 3

    def test_caution_high(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        assert compute_spread_cents(0.49) == 4

    def test_toxic_returns_none(self):
        from kalshi_mm.mm_strategy import compute_spread_cents
        assert compute_spread_cents(0.6) is None


class TestComputeBidCents:
    def test_normal_bid(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(50, 2) == 49  # 50 - 2//2 = 49

    def test_wide_spread(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(50, 4) == 48  # 50 - 4//2 = 48

    def test_boundary_low_returns_none(self):
        from kalshi_mm.mm_strategy import compute_bid_cents
        assert compute_bid_cents(1, 4) is None  # 1 - 2 = -1, invalid


class TestComputeAskCents:
    def test_normal_ask(self):
        from kalshi_mm.mm_strategy import compute_ask_cents
        assert compute_ask_cents(48, 3) == 51  # 48 + 3

    def test_boundary_high_returns_none(self):
        from kalshi_mm.mm_strategy import compute_ask_cents
        assert compute_ask_cents(97, 4) is None  # 97 + 4 = 101, invalid


class TestParseOrderbookVolume:
    def test_sums_volumes(self):
        from kalshi_mm.mm_strategy import parse_ob_total_volume
        levels = [["0.48", "10"], ["0.47", "20"], ["0.46", "5"]]
        assert parse_ob_total_volume(levels) == 35
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mm_strategy.py -v`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Write mm_strategy.py**

```python
# kalshi_mm/mm_strategy.py
"""Market maker strategy: orderbook parsing, mid price, spread sizing, quote generation."""
from kalshi_mm.mm_config import SPREAD_MIN_CENTS, SPREAD_MAX_CENTS


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
    if vpin < 0.3:
        return SPREAD_MIN_CENTS
    elif vpin < 0.5:
        return 3 + round((vpin - 0.3) / 0.2)
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
    """Sum contract quantities from orderbook levels [[price_str, count_str], ...]."""
    total = 0
    for level in levels:
        total += int(level[1])
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mm_strategy.py -v`
Expected: All 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add kalshi_mm/mm_strategy.py tests/test_mm_strategy.py
git commit -m "feat(mm): add strategy module with orderbook parsing, mid calc, spread sizing"
```

---

### Task 6: Market Maker Daemon — State Machine, Dashboard, Main Loop

**Files:**
- Create: `kalshi_mm/mm_daemon.py`
- Create: `tests/test_mm_daemon.py`

This is the largest task. The daemon contains:
- Per-asset state machine (IDLE → DISCOVERING → QUOTING_BID → QUOTING_ASK → EXITING → DARK)
- Market discovery (find 15m markets with right time-to-expiry)
- Order placement and fill checking
- Dashboard rendering
- Dry-run simulation
- CLI entry point with argparse

- [ ] **Step 1: Write integration tests for the state machine**

```python
# tests/test_mm_daemon.py
"""Integration tests for MM daemon state machine."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import time

from kalshi_mm.mm_config import (
    IDLE, DISCOVERING, QUOTING_BID, QUOTING_ASK, EXITING, DARK,
)


def _mock_kalshi():
    """Create a mocked KalshiClient."""
    client = MagicMock()
    client.get_balance.return_value = {"balance": 10000}  # $100 in cents
    client.get_orderbook.return_value = {
        "orderbook_fp": {
            "yes_dollars": [["0.4800", "50"]],
            "no_dollars": [["0.5000", "30"]],
        }
    }
    client.place_order.return_value = {"order": {"order_id": "ord-123"}}
    client.get_order_status.return_value = {"order": {"status": "resting"}}
    client.cancel_order_safe.return_value = {"order": {"status": "cancelled"}}
    return client


def _mock_market(mins_to_exp=8):
    return {
        "ticker": "KXBTC15M-TEST",
        "expiration_time": "2026-03-25T15:00:00Z",
        "_mins_to_expiry": mins_to_exp,
    }


class TestStateTransitions:
    def test_idle_to_discovering_on_start(self):
        from kalshi_mm.mm_daemon import MMAssetRunner
        runner = MMAssetRunner("BTC/USDT", "KXBTC15M", _mock_kalshi(), dry_run=True)
        assert runner.inv.state == IDLE

    def test_discovering_to_quoting_bid(self):
        from kalshi_mm.mm_daemon import MMAssetRunner
        client = _mock_kalshi()
        runner = MMAssetRunner("BTC/USDT", "KXBTC15M", client, dry_run=True)

        market = _mock_market(8)
        with patch.object(runner, "_find_market", return_value=market):
            with patch("kalshi_mm.mm_daemon.get_trade_flow", return_value={"net_flow": 0.05}):
                runner.tick()  # IDLE → DISCOVERING → QUOTING_BID
        assert runner.inv.state == QUOTING_BID

    def test_quoting_bid_to_ask_on_fill(self):
        from kalshi_mm.mm_daemon import MMAssetRunner
        client = _mock_kalshi()
        client.get_order_status.return_value = {"order": {"status": "executed"}}
        runner = MMAssetRunner("BTC/USDT", "KXBTC15M", client, dry_run=False)
        # Set up in QUOTING_BID state
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.pending_bid_id = "ord-123"
        runner.inv.bid_price_cents = 48

        with patch("kalshi_mm.mm_daemon.get_trade_flow", return_value={"net_flow": 0.05}):
            runner.tick()
        assert runner.inv.state == QUOTING_ASK
        assert runner.inv.yes_held > 0

    def test_dark_on_toxic_vpin(self):
        from kalshi_mm.mm_daemon import MMAssetRunner
        client = _mock_kalshi()
        runner = MMAssetRunner("BTC/USDT", "KXBTC15M", client, dry_run=True)
        runner.inv.state = QUOTING_BID
        runner.inv.market_ticker = "KXBTC15M-TEST"
        runner.inv.expiry_ts = time.time() + 600
        runner.inv.pending_bid_id = "ord-123"
        runner.inv.bid_price_cents = 48

        # Simulate toxic flow
        with patch("kalshi_mm.mm_daemon.get_trade_flow", return_value={"net_flow": 0.7}):
            runner.tick()
        assert runner.inv.state == DARK
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mm_daemon.py -v`
Expected: FAIL — `MMAssetRunner` doesn't exist.

- [ ] **Step 3: Write mm_daemon.py — MMAssetRunner class (state machine core)**

The `MMAssetRunner` class manages one asset's state machine. Key methods:
- `tick()` — called every POLL_INTERVAL, dispatches to state handler
- `_tick_idle()` — transition to DISCOVERING
- `_tick_discovering()` — find market, check VPIN, transition to QUOTING_BID
- `_tick_quoting_bid()` — check fills, re-quote drift, check kill switch
- `_tick_quoting_ask()` — check fills, check timeout, check kill switch
- `_tick_exiting()` — aggressive limit exit sequence
- `_tick_dark()` — monitor VPIN recovery

Write the full `kalshi_mm/mm_daemon.py` with:

```python
# kalshi_mm/mm_daemon.py
"""Kalshi market maker daemon — state machine, dashboard, main loop."""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exchange.kalshi import KalshiClient
from data.market_data import get_trade_flow
from data.fetcher import DataFetcher
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID

from kalshi_mm.mm_config import *
from kalshi_mm.mm_inventory import MMInventory, compute_contracts, calc_round_trip_pnl
from kalshi_mm.mm_strategy import (
    compute_mid_cents, compute_spread_cents, compute_bid_cents,
    compute_ask_cents, parse_ob_total_volume,
)
from kalshi_mm.mm_vpin import (
    compute_spot_vpin, compute_kalshi_ob_heuristic,
    compute_blended_vpin, KillSwitch,
)


class MMAssetRunner:
    """State machine for one asset (BTC or ETH)."""

    def __init__(self, symbol: str, series_ticker: str,
                 kalshi: KalshiClient, dry_run: bool = True):
        self.symbol = symbol
        self.series_ticker = series_ticker
        self.kalshi = kalshi
        self.dry_run = dry_run
        self.inv = MMInventory(asset=symbol.split("/")[0])
        self.kill_switch = KillSwitch()
        self._prev_yes_vol = 0
        self._prev_no_vol = 0
        self._last_discovery_ts = 0.0
        self._current_spread = SPREAD_DEFAULT_CENTS
        self._current_vpin = 0.0
        self._trade_log: list[dict] = []

    def tick(self):
        """Main tick — dispatch to current state handler."""
        state = self.inv.state
        if state == IDLE:
            self._tick_idle()
        elif state == DISCOVERING:
            self._tick_discovering()
        elif state == QUOTING_BID:
            self._tick_quoting_bid()
        elif state == QUOTING_ASK:
            self._tick_quoting_ask()
        elif state == EXITING:
            self._tick_exiting()
        elif state == DARK:
            self._tick_dark()

    # --- State handlers ---

    def _tick_idle(self):
        now = time.time()
        if now - self._last_discovery_ts < DISCOVERY_RETRY_SECONDS:
            return
        self._last_discovery_ts = now
        self.inv.state = DISCOVERING
        self._tick_discovering()

    def _tick_discovering(self):
        market = self._find_market()
        if market is None:
            self.inv.state = IDLE
            return

        self.inv.market_ticker = market["ticker"]
        exp_str = market.get("expiration_time") or market.get("close_time", "")
        if exp_str and "T" in exp_str:
            self.inv.expiry_ts = datetime.fromisoformat(
                exp_str.replace("Z", "+00:00")
            ).timestamp()

        # Compute VPIN
        vpin = self._compute_vpin()
        spread = compute_spread_cents(vpin)
        if spread is None:
            self.inv.state = DARK
            return

        self._current_spread = spread
        self.inv.reset_window()
        self.inv.state = QUOTING_BID
        self._place_bid()

    def _tick_quoting_bid(self):
        if self.inv.minutes_to_expiry() < HARD_CUTOFF_MINUTES:
            self._cancel_pending_bid()
            self.inv.state = IDLE
            return

        vpin = self._compute_vpin()
        spread = compute_spread_cents(vpin)
        if spread is None:
            self._cancel_pending_bid()
            self.inv.state = DARK
            return
        self._current_spread = spread

        # Check if bid filled
        if self._check_bid_filled():
            self.inv.state = QUOTING_ASK
            self._place_ask()
            return

        # Re-quote if mid drifted
        ob = self._get_orderbook()
        if ob:
            new_mid = compute_mid_cents(ob)
            if new_mid and abs(new_mid - self.inv.bid_price_cents) > REQUOTE_DRIFT_CENTS:
                result = self._cancel_pending_bid()
                if result and result.get("status") == "filled":
                    # Bid filled during cancel — treat as fill
                    self._handle_bid_fill_from_cancel()
                    return
                self._place_bid(mid_override=new_mid)

    def _tick_quoting_ask(self):
        if self.inv.minutes_to_expiry() < HARD_CUTOFF_MINUTES:
            self._cancel_pending_ask()
            self.inv.state = EXITING
            return

        vpin = self._compute_vpin()
        if self.kill_switch.should_go_dark(vpin) and self.kill_switch.vpin_rising():
            self._cancel_pending_ask()
            self.inv.state = EXITING
            return

        if self.inv.inventory_age_seconds() > INVENTORY_TIMEOUT_SECONDS:
            self._cancel_pending_ask()
            self.inv.state = EXITING
            return

        if self._check_ask_filled():
            if self.inv.minutes_to_expiry() >= QUOTE_WINDOW_MIN_MINUTES:
                self.inv.state = QUOTING_BID
                self._place_bid()
            else:
                self.inv.state = IDLE
            return

    def _tick_exiting(self):
        if not self.inv.has_inventory():
            self.inv.state = IDLE if self.inv.minutes_to_expiry() < QUOTE_WINDOW_MIN_MINUTES else QUOTING_BID
            return

        now = time.time()
        if self.inv.exit_attempt_count >= 3:
            # Last resort: market order
            self._exit_at_market()
            return

        if self.inv.last_exit_attempt_ts > 0 and now - self.inv.last_exit_attempt_ts < FORCED_EXIT_RETRY_SECONDS:
            # Check if previous exit order filled
            if self.inv.pending_ask_id and self._check_ask_filled():
                return
            return  # Wait for retry interval

        # Place aggressive limit exit
        ob = self._get_orderbook()
        if ob:
            ob_data = ob.get("orderbook_fp", ob)
            yes_bids = ob_data.get("yes_dollars", [])
            if yes_bids:
                best_bid = round(float(yes_bids[0][0]) * 100)
                floor = self.inv.entry_price_cents - MAX_EXIT_LOSS_CENTS
                exit_price = max(best_bid - 1, floor)
                exit_price = max(1, min(99, exit_price))
                self._place_exit_sell(exit_price)
                self.inv.exit_attempt_count += 1
                self.inv.last_exit_attempt_ts = now
            else:
                self._exit_at_market()
        else:
            self._exit_at_market()

    def _tick_dark(self):
        vpin = self._compute_vpin()
        if vpin < VPIN_SAFE and self.inv.minutes_to_expiry() >= QUOTE_WINDOW_MIN_MINUTES:
            self.inv.state = QUOTING_BID
            self._place_bid()
        elif self.inv.minutes_to_expiry() < HARD_CUTOFF_MINUTES:
            self.inv.state = IDLE

    # --- Helpers ---

    def _find_market(self) -> dict | None:
        """Find an active 15m market in the right time window."""
        try:
            markets = self.kalshi.get_markets(series_ticker=self.series_ticker, status="open")
        except Exception:
            return None
        now = time.time()
        for m in markets:
            exp = m.get("expiration_time") or m.get("close_time", "")
            if exp and "T" in exp:
                try:
                    exp_ts = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
                    mins = (exp_ts - now) / 60
                    if QUOTE_WINDOW_MIN_MINUTES <= mins <= QUOTE_WINDOW_MAX_MINUTES:
                        m["_mins_to_expiry"] = round(mins, 1)
                        return m
                except Exception:
                    continue
        return None

    def _get_orderbook(self) -> dict | None:
        if not self.inv.market_ticker:
            return None
        try:
            return self.kalshi.get_orderbook(self.inv.market_ticker)
        except Exception:
            return None

    def _compute_vpin(self) -> float:
        try:
            flow = get_trade_flow(self.symbol, limit=200)
            spot_vpin = compute_spot_vpin(flow)
        except Exception:
            spot_vpin = 0.0

        ob = self._get_orderbook()
        kalshi_h = 0.0
        if ob:
            ob_data = ob.get("orderbook_fp", ob)
            yes_vol = parse_ob_total_volume(ob_data.get("yes_dollars", []))
            no_vol = parse_ob_total_volume(ob_data.get("no_dollars", []))
            kalshi_h = compute_kalshi_ob_heuristic(self._prev_yes_vol, yes_vol, self._prev_no_vol, no_vol)
            self._prev_yes_vol = yes_vol
            self._prev_no_vol = no_vol

        vpin = compute_blended_vpin(spot_vpin, kalshi_h)
        self._current_vpin = vpin
        self.kill_switch.record_vpin(vpin)
        return vpin

    def _place_bid(self, mid_override: int | None = None):
        ob = self._get_orderbook()
        if not ob:
            return
        mid = mid_override or compute_mid_cents(ob)
        if mid is None:
            return

        bid = compute_bid_cents(mid, self._current_spread)
        if bid is None:
            return

        balance = self._get_balance_cents()
        contracts = compute_contracts(balance, bid)
        if contracts is None:
            return

        if self.dry_run:
            self.inv.pending_bid_id = f"dry-bid-{time.time()}"
            self.inv.bid_price_cents = bid
            self.inv.yes_held = 0  # not filled yet
        else:
            try:
                resp = self.kalshi.place_order(
                    self.inv.market_ticker, "yes", contracts,
                    price_cents=bid, order_type="limit", action="buy",
                )
                order = resp.get("order", {})
                self.inv.pending_bid_id = order.get("order_id")
                self.inv.bid_price_cents = bid
            except Exception as e:
                self.inv.state = DARK
                return

        self._log_trade("BID", bid, contracts)

    def _place_ask(self):
        ask = compute_ask_cents(self.inv.entry_price_cents, self._current_spread)
        if ask is None:
            self.inv.state = EXITING
            return

        if self.dry_run:
            self.inv.pending_ask_id = f"dry-ask-{time.time()}"
            self.inv.ask_price_cents = ask
        else:
            try:
                resp = self.kalshi.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    price_cents=ask, order_type="limit", action="sell",
                )
                order = resp.get("order", {})
                self.inv.pending_ask_id = order.get("order_id")
                self.inv.ask_price_cents = ask
            except Exception:
                self.inv.state = EXITING
                return

        self._log_trade("ASK", ask, self.inv.yes_held)

    def _place_exit_sell(self, price_cents: int):
        if self.dry_run:
            self.inv.pending_ask_id = f"dry-exit-{time.time()}"
            self.inv.ask_price_cents = price_cents
        else:
            try:
                resp = self.kalshi.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    price_cents=price_cents, order_type="limit", action="sell",
                )
                order = resp.get("order", {})
                self.inv.pending_ask_id = order.get("order_id")
                self.inv.ask_price_cents = price_cents
            except Exception:
                pass

    def _exit_at_market(self):
        if self.dry_run:
            # Simulate worst-case exit
            exit_price = max(1, self.inv.entry_price_cents - MAX_EXIT_LOSS_CENTS)
            self.inv.record_sell_fill(exit_price)
        else:
            try:
                self.kalshi.place_order(
                    self.inv.market_ticker, "yes", self.inv.yes_held,
                    order_type="market", action="sell",
                )
            except Exception:
                pass
            self.inv.record_sell_fill(max(1, self.inv.entry_price_cents - MAX_EXIT_LOSS_CENTS))
        self.inv.state = IDLE
        self._log_trade("FORCED_EXIT", self.inv.entry_price_cents - MAX_EXIT_LOSS_CENTS, 0)

    def _check_bid_filled(self) -> bool:
        if not self.inv.pending_bid_id:
            return False
        if self.dry_run:
            ob = self._get_orderbook()
            if ob:
                ob_data = ob.get("orderbook_fp", ob)
                no_bids = ob_data.get("no_dollars", [])
                if no_bids:
                    yes_ask = 100 - round(float(no_bids[0][0]) * 100)
                    if yes_ask <= self.inv.bid_price_cents:
                        balance = self._get_balance_cents()
                        contracts = compute_contracts(balance, self.inv.bid_price_cents) or MIN_CONTRACTS_PER_QUOTE
                        self.inv.record_buy_fill(contracts, self.inv.bid_price_cents)
                        self._log_trade("BUY_FILL", self.inv.entry_price_cents, self.inv.yes_held)
                        return True
            return False
        try:
            resp = self.kalshi.get_order_status(self.inv.pending_bid_id)
            status = resp.get("order", {}).get("status", "")
            if status in ("executed", "filled"):
                order = resp["order"]
                contracts = order.get("count", MIN_CONTRACTS_PER_QUOTE)
                price = order.get("yes_price", self.inv.bid_price_cents)
                self.inv.record_buy_fill(contracts, price)
                self._log_trade("BUY_FILL", price, contracts)
                return True
        except Exception:
            pass
        return False

    def _check_ask_filled(self) -> bool:
        if not self.inv.pending_ask_id:
            return False
        if self.dry_run:
            ob = self._get_orderbook()
            if ob:
                ob_data = ob.get("orderbook_fp", ob)
                yes_bids = ob_data.get("yes_dollars", [])
                if yes_bids:
                    best_bid = round(float(yes_bids[0][0]) * 100)
                    if best_bid >= self.inv.ask_price_cents:
                        rt = self.inv.record_sell_fill(self.inv.ask_price_cents)
                        self._log_trade("SELL_FILL", self.inv.ask_price_cents, rt.get("contracts", 0), rt)
                        return True
            return False
        try:
            resp = self.kalshi.get_order_status(self.inv.pending_ask_id)
            status = resp.get("order", {}).get("status", "")
            if status in ("executed", "filled"):
                order = resp["order"]
                sell_price = order.get("yes_price", self.inv.ask_price_cents)
                rt = self.inv.record_sell_fill(sell_price)
                self._log_trade("SELL_FILL", sell_price, rt.get("contracts", 0), rt)
                return True
        except Exception:
            pass
        return False

    def _handle_bid_fill_from_cancel(self):
        balance = self._get_balance_cents()
        contracts = compute_contracts(balance, self.inv.bid_price_cents) or MIN_CONTRACTS_PER_QUOTE
        self.inv.record_buy_fill(contracts, self.inv.bid_price_cents)
        self.inv.state = QUOTING_ASK
        self._place_ask()

    def _cancel_pending_bid(self) -> dict | None:
        if not self.inv.pending_bid_id:
            return None
        if self.dry_run:
            self.inv.pending_bid_id = None
            return {"status": "cancelled"}
        try:
            result = self.kalshi.cancel_order_safe(self.inv.pending_bid_id)
            self.inv.pending_bid_id = None
            return result
        except Exception:
            self.inv.pending_bid_id = None
            return None

    def _cancel_pending_ask(self) -> dict | None:
        if not self.inv.pending_ask_id:
            return None
        if self.dry_run:
            self.inv.pending_ask_id = None
            return {"status": "cancelled"}
        try:
            result = self.kalshi.cancel_order_safe(self.inv.pending_ask_id)
            self.inv.pending_ask_id = None
            return result
        except Exception:
            self.inv.pending_ask_id = None
            return None

    def _get_balance_cents(self) -> int:
        try:
            resp = self.kalshi.get_balance()
            return resp.get("balance", 0)
        except Exception:
            return 0

    def _log_trade(self, action: str, price: int, contracts: int, rt_info: dict | None = None):
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "asset": self.inv.asset,
            "action": action,
            "price_cents": price,
            "contracts": contracts,
            "pnl_cents": rt_info.get("pnl_cents") if rt_info else None,
        }
        self._trade_log.append(entry)
        if len(self._trade_log) > 50:
            self._trade_log = self._trade_log[-50:]


# ====================================================================
# Dashboard
# ====================================================================

def render_dashboard(runners: list[MMAssetRunner], mode: str,
                     start_time: float, balance_cents: int) -> str:
    """Render the live ASCII dashboard."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total_day_pnl = sum(r.inv.day_pnl_cents for r in runners)
    total_rts = sum(r.inv.window_round_trips for r in runners)
    uptime = time.time() - start_time
    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m"

    lines = []
    w = 71
    lines.append("=" * w)
    tag = "DRY-RUN" if mode == "dry-run" else "LIVE"
    lines.append(f"  KALSHI MARKET MAKER  |  {now}  |  {tag}")
    lines.append("-" * w)
    lines.append(f"  BALANCE: ${balance_cents / 100:.2f}  |  Day P&L: {_fmt_pnl(total_day_pnl)}  |  Round Trips: {total_rts}")
    lines.append("-" * w)

    for r in runners:
        inv = r.inv
        vpin_label = r.kill_switch.get_spread_state(r._current_vpin)
        exp_str = f"{inv.minutes_to_expiry():.0f}m" if inv.expiry_ts > 0 else "--"
        ticker = inv.market_ticker or "searching..."
        lines.append(f"  {inv.asset} ({ticker})  exp {exp_str}  |  VPIN: {r._current_vpin:.2f} [{vpin_label}]")
        lines.append(f"  | Mid: {r.inv.bid_price_cents + r._current_spread // 2 if inv.bid_price_cents else '--'}c  Spread: {r._current_spread}c  |  State: {inv.state}")

        if inv.state == QUOTING_BID and inv.pending_bid_id:
            lines.append(f"  | BID: YES @ {inv.bid_price_cents}c [RESTING]")
        elif inv.state == QUOTING_ASK and inv.pending_ask_id:
            age = f"{inv.inventory_age_seconds():.0f}s" if inv.has_inventory() else ""
            lines.append(f"  | ASK: {inv.yes_held}x YES @ {inv.ask_price_cents}c [RESTING]  |  Held @ {inv.entry_price_cents}c ({age})")
        elif inv.state == EXITING:
            lines.append(f"  | EXITING inventory: {inv.yes_held}x @ {inv.entry_price_cents}c  attempt {inv.exit_attempt_count}/3")

        lines.append(f"  | Inventory: {inv.yes_held}  |  Window RTs: {inv.window_round_trips}  |  Window P&L: {_fmt_pnl(inv.window_pnl_cents)}")

        if r._trade_log:
            last = r._trade_log[-1]
            pnl_str = f" -> {_fmt_pnl(last['pnl_cents'])}" if last.get("pnl_cents") is not None else ""
            lines.append(f"  | Last: {last['action']} @ {last['price_cents']}c{pnl_str}")
        lines.append("-" * w)

    # Kill switch status
    any_dark = any(r.inv.state == DARK for r in runners)
    ks_label = "TRIGGERED" if any_dark else "OK"
    daily_remaining = MAX_DAILY_LOSS_CENTS + total_day_pnl
    lines.append(f"  KILL SWITCH: {ks_label}  |  Daily loss remaining: ${daily_remaining / 100:.2f}")
    lines.append("-" * w)

    # Recent trades (last 10 across all runners)
    all_trades = []
    for r in runners:
        all_trades.extend(r._trade_log)
    all_trades.sort(key=lambda t: t["time"], reverse=True)
    lines.append("  RECENT TRADES")
    for t in all_trades[:10]:
        ts = t["time"][11:19]
        pnl = f" -> {_fmt_pnl(t['pnl_cents'])}" if t.get("pnl_cents") is not None else ""
        lines.append(f"  {ts}  {t['asset']}  {t['action']}  {t['contracts']}x @ {t['price_cents']}c{pnl}")
    lines.append("-" * w)

    # Session stats
    total_session_rts = sum(len([t for t in r._trade_log if t.get("pnl_cents") is not None]) for r in runners)
    forced = sum(1 for r in runners for t in r._trade_log if t["action"] == "FORCED_EXIT")
    lines.append(f"  SESSION: {total_session_rts} RTs  |  Forced exits: {forced}  |  Uptime: {uptime_str}")
    lines.append("=" * w)

    return "\n".join(lines)


def _fmt_pnl(cents: int | None) -> str:
    if cents is None:
        return "--"
    sign = "+" if cents >= 0 else ""
    return f"{sign}${cents / 100:.2f}"


# ====================================================================
# Main loop
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Kalshi Market Maker")
    parser.add_argument("--mode", choices=["live", "dry-run"], default="dry-run")
    parser.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    args = parser.parse_args()

    kalshi = KalshiClient(KALSHI_API_KEY_ID, str(KALSHI_KEY_FILE))
    dry_run = args.mode == "dry-run"

    runners = []
    for symbol, series in MM_ASSETS.items():
        runners.append(MMAssetRunner(symbol, series, kalshi, dry_run=dry_run))

    start_time = time.time()
    cycle = 0
    data_dir = Path("data/store")
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MM] Starting market maker ({args.mode}) for {list(MM_ASSETS.keys())}")

    try:
        while True:
            # Get balance once per cycle
            try:
                bal = kalshi.get_balance().get("balance", 0)
            except Exception:
                bal = 0

            # Tick each asset runner
            for runner in runners:
                try:
                    runner.tick()
                except Exception as e:
                    print(f"[MM] Error in {runner.inv.asset}: {e}")

            # Render dashboard
            dashboard = render_dashboard(runners, args.mode, start_time, bal)
            os.system("clear" if os.name != "nt" else "cls")
            print(dashboard)

            # Persist dashboard to file for MCP
            (data_dir / "mm_dashboard.log").write_text(dashboard)

            # Check daily loss halt
            total_day_pnl = sum(r.inv.day_pnl_cents for r in runners)
            if total_day_pnl <= -MAX_DAILY_LOSS_CENTS:
                print("[MM] Daily loss limit hit. Halting.")
                break

            cycle += 1
            if args.cycles > 0 and cycle >= args.cycles:
                break

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[MM] Shutting down...")
    finally:
        # Cancel all pending orders
        for runner in runners:
            runner._cancel_pending_bid()
            runner._cancel_pending_ask()
        print("[MM] All orders cancelled. Goodbye.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mm_daemon.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/test_mm_inventory.py tests/test_mm_vpin.py tests/test_mm_strategy.py tests/test_mm_daemon.py tests/test_kalshi_extended.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add kalshi_mm/mm_daemon.py tests/test_mm_daemon.py
git commit -m "feat(mm): add market maker daemon with state machine, dashboard, main loop"
```

---

### Task 7: MCP Server Integration

**Files:**
- Modify: `mcp_server.py`

- [ ] **Step 1: Add 3 new MCP tool definitions to `list_tools()` in mcp_server.py**

Add after the existing tool definitions (around line 200). Add `algotrade_mm_start`, `algotrade_mm_stop`, and `algotrade_mm_status` tools.

- [ ] **Step 2: Add a global `_mm_process` handle** alongside the existing `_daemon_process` at line 20.

```python
_mm_process = None
```

- [ ] **Step 3: Add handler functions for the 3 new tools**

```python
def handle_mm_start(mode="dry-run", assets=None):
    """Start market maker daemon."""
    global _mm_process
    if _daemon_process and _daemon_process.poll() is None:
        return {"error": "Predictor daemon is running. Stop it first (algotrade_stop)."}
    if _mm_process and _mm_process.poll() is None:
        return {"error": "Market maker already running.", "pid": _mm_process.pid}

    cmd = [sys.executable, "-m", "kalshi_mm.mm_daemon", "--mode", mode]
    log_path = "/tmp/mm_stdout.log"
    with open(log_path, "w") as log_f:
        _mm_process = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent))
    return {"status": "started", "mode": mode, "pid": _mm_process.pid}

def handle_mm_stop():
    global _mm_process
    if not _mm_process or _mm_process.poll() is not None:
        return {"status": "not running"}
    _mm_process.terminate()
    _mm_process.wait(timeout=10)
    _mm_process = None
    return {"status": "stopped"}

def handle_mm_status():
    dash_path = Path("data/store/mm_dashboard.log")
    if dash_path.exists():
        return {"dashboard": dash_path.read_text()}
    return {"status": "no dashboard data"}
```

- [ ] **Step 4: Wire the handlers into the `call_tool()` dispatcher** (around line 1388).

Add before the `else: result = {"error": ...}` block:

```python
elif name == "algotrade_mm_start":
    result = handle_mm_start(
        mode=arguments.get("mode", "dry-run"),
        assets=arguments.get("assets"),
    )
elif name == "algotrade_mm_stop":
    result = handle_mm_stop()
elif name == "algotrade_mm_status":
    result = handle_mm_status()
```

- [ ] **Step 5: Test MCP tools manually**

Run: `python -c "from mcp_server import handle_mm_status; print(handle_mm_status())"`
Expected: `{"status": "no dashboard data"}` (no daemon running).

- [ ] **Step 6: Commit**

```bash
git add mcp_server.py
git commit -m "feat(mm): add MCP tools for market maker start/stop/status"
```

---

### Task 8: End-to-End Dry Run Verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass, no regressions.

- [ ] **Step 2: Start market maker in dry-run mode for 2 cycles**

Run: `python -m kalshi_mm.mm_daemon --mode dry-run --cycles 2`
Expected: Dashboard renders, state machine ticks through IDLE → DISCOVERING, attempts to find markets. If Kalshi API is reachable, should discover markets and begin quoting. Dashboard should refresh every 3 seconds.

- [ ] **Step 3: Verify dashboard log is written**

Run: `cat data/store/mm_dashboard.log`
Expected: Latest dashboard frame visible.

- [ ] **Step 4: Commit any fixes found during verification**

```bash
git add -u
git commit -m "fix(mm): dry-run verification fixes"
```

- [ ] **Step 5: Final commit — tag the feature as complete**

```bash
git add -A
git commit -m "feat(mm): complete Kalshi market maker bot v1

Separate market making bot for Kalshi 15m crypto (BTC, ETH).
Spread capture strategy with VPIN-based kill switch, inventory
management, and live ASCII dashboard.

Components:
- kalshi_mm/mm_config.py — configuration constants
- kalshi_mm/mm_inventory.py — fee math, P&L, position sizing
- kalshi_mm/mm_vpin.py — VPIN computation, kill switch
- kalshi_mm/mm_strategy.py — orderbook parsing, mid calc, spread sizing
- kalshi_mm/mm_daemon.py — state machine, dashboard, main loop
- exchange/kalshi.py — extended with sell action, cancel_order_safe
- mcp_server.py — 3 new MM tools"
```
