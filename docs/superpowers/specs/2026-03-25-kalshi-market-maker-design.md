# Kalshi 15m Crypto Market Maker — Design Spec

**Date:** 2026-03-25
**Status:** Draft (v3 — second review fixes)
**Assets:** BTC, ETH (KXBTC15M, KXETH15M)

## Overview

A separate, standalone market making bot for Kalshi 15-minute crypto UP/DOWN markets. The bot captures bid/ask spread by posting resting limit orders on YES contracts — buying low, then selling high — profiting from the spread regardless of outcome. A VPIN-based kill switch detects toxic order flow and stops quoting before adverse selection destroys profits.

This is NOT a directional predictor. The bot does not predict price direction. It earns the spread by providing liquidity.

## Strategy: Spread Capture Market Making

### Core Mechanic

The bot operates as a **sequential bid-then-ask** market maker. You cannot sell YES contracts you don't hold, so the flow is:

```
Phase 1: Post resting BUY YES at (mid - spread/2)     ← bid
Phase 2: Bid fills → now holding YES → post resting SELL YES at (entry + spread)  ← ask
Phase 3: Ask fills → round trip complete → profit = spread - fees
Phase 4: Immediately re-quote new bid at current mid → back to Phase 1
```

Multiple round trips per 15m window. Each completed round trip frees capital for the next.

### Why It Works

- Retail traders cross the spread for immediate fills (pay taker fees)
- We post resting limit orders (pay lower maker fees)
- The spread between our buy and sell is profit
- Outcome doesn't matter when both sides fill within the window

### Price Units — CENTS Throughout

**All prices in this spec and implementation are in cents (integers 1-99).** This matches the existing `exchange/kalshi.py` codebase where `yes_price` and `no_price` are cent integers.

The Kalshi orderbook API (`/v2/markets/{ticker}/orderbook`) returns `yes_dollars` and `no_dollars` as arrays of `[price_string, count_string]` where price is a dollar decimal string (e.g., `"0.4800"`). We convert to cents immediately on read: `int(float(price_str) * 100)`.

### Fee Math

Kalshi fee formula (P in dollars, output in dollars, rounded up to nearest cent):
- **Maker:** `ceil_cents(0.0175 * C * P * (1 - P))`
- **Taker:** `ceil_cents(0.07 * C * P * (1 - P))`

Where `ceil_cents(x)` = `math.ceil(x * 100)` (result in cents).

**Fee per contract at 50c price point (worst case — parabolic peak):**

| Contracts | Maker/side (total) | Maker/side (per contract) | Round trip/contract |
|-----------|-------------------|--------------------------|---------------------|
| 1 | 1c | 1.00c | 2.00c |
| 5 | 3c | 0.60c | 1.20c |
| 10 | 5c | 0.50c | 1.00c |
| 20 | 9c | 0.45c | 0.90c |

**Key insight: batch size matters.** The `ceil()` amortizes better over more contracts. We should trade 10+ contracts per order to minimize per-unit fees.

**At 30c price point (more favorable):**

| Contracts | Maker/side (per contract) | Round trip/contract |
|-----------|--------------------------|---------------------|
| 10 | 0.40c | 0.80c |
| 20 | 0.40c | 0.80c |

**Profit per round trip at target spreads (10 contracts at 50c):**

| Spread | Revenue/contract | Fee round trip/contract | Net Profit/contract |
|--------|-----------------|------------------------|---------------------|
| 2c | 2.00c | 1.00c | 1.00c |
| 3c | 3.00c | 1.00c | 2.00c |
| 4c | 4.00c | 1.00c | 3.00c |

**Minimum viable spread at 50c: 2c** (with 10+ contracts). At fewer contracts, need 3c+ to profit.

All orders are **limit orders** (maker). We never cross the spread (taker).

## Architecture

### File Structure

```
kalshi_mm/
├── mm_daemon.py          — Main event loop, dashboard rendering, lifecycle
├── mm_strategy.py        — Mid price calc, spread sizing, quote generation
├── mm_vpin.py            — VPIN computation, kill switch logic
├── mm_inventory.py       — Position tracking, P&L, risk limits, fee calc
└── mm_config.py          — All configurable thresholds and parameters
```

### Reused Modules

| Module | Usage | Changes |
|--------|-------|---------|
| `exchange/kalshi.py` | API client | Add `action` param to `place_order()` (default `"buy"`, backward-compatible) |
| `data/market_data.py` | BinanceUS trade flow for spot VPIN | None — use existing `get_trade_flow()` |
| `data/fetcher.py` | Spot price for volatility monitoring | None |

### New MCP Tools

| Tool | Purpose |
|------|---------|
| `algotrade_mm_start` | Start market maker daemon (mode, assets) |
| `algotrade_mm_stop` | Graceful stop — cancel all orders, exit inventory |
| `algotrade_mm_status` | Dashboard state + VPIN levels + P&L |

### Shared Account Safety

The MM and the existing directional predictor share the same Kalshi account. To prevent collisions:

- **Do not run both simultaneously** — the MM is a separate bot, run one or the other
- `algotrade_mm_start` checks if the predictor daemon is running and refuses to start if so
- `algotrade_mm_stop` cancels ONLY orders placed by the MM (tracked by order ID in local state)
- The existing `algotrade_close_all` will also cancel MM orders — this is intentional (emergency stop)

## Market Making Loop

### Per-Asset State Machine

Each asset (BTC, ETH) runs independently with its own state:

```
IDLE → DISCOVERING → QUOTING_BID → QUOTING_ASK → EXITING → DARK → IDLE
```

**IDLE:** No active market. Waiting for next 15m window. Retry market discovery every 10 seconds with backoff (Kalshi may not list the next window immediately after the previous one settles).

**DISCOVERING:** Searching for an active 15m market with 5-12 minutes to expiry. Read orderbook, compute mid and VPIN. If VPIN is safe and orderbook has depth, transition to QUOTING_BID.

**QUOTING_BID:** Resting BUY YES limit posted. No inventory held. Poll every 3 seconds:
- Check if bid filled (order status)
- Update mid price from orderbook
- Recompute VPIN
- If mid drifts > 1c from bid price → cancel and re-post bid at new mid
- If bid fills → transition to QUOTING_ASK
- If VPIN > TOXIC threshold → cancel bid, transition to DARK
- If < 2 min to expiry → cancel bid, transition to IDLE

**QUOTING_ASK:** Holding YES inventory from filled bid. Resting SELL YES limit posted. Poll every 3 seconds:
- Check if ask filled (order status)
- Recompute VPIN
- If ask fills → log round trip profit → check time: if >= 5 min to expiry, transition to QUOTING_BID (new round trip); otherwise transition to IDLE
- If VPIN > TOXIC AND trending up → cancel ask, transition to EXITING
- If inventory held > 5 min timeout → cancel ask, transition to EXITING
- If < 2 min to expiry → cancel ask, transition to EXITING
- Handle cancel returning 404 (order already filled between check and cancel) → treat as fill

**EXITING:** Actively unwinding inventory via aggressive limit orders. Tracks `exit_attempt_count` and `last_exit_attempt_time`. Poll every 3 seconds:
- Post SELL YES at `max(best_yes_bid - 1c, entry_price - 5c)` (undercut best bid)
- If filled → log forced exit with loss, transition to IDLE (or QUOTING_BID if > 5 min to expiry and not from kill switch)
- If not filled after 10 seconds → lower price by 1c and retry (`exit_attempt_count++`)
- After 3 failed attempts → use market order as last resort
- Log every forced exit with loss amount
- Does NOT block the poll loop — integrated into the normal 3-second cadence

**DARK:** Kill switch active. All orders cancelled. No inventory (already exited via EXITING). No quoting.
- Continue monitoring VPIN every 3 seconds
- If VPIN recovers below SAFE AND > 5 min to expiry → transition to QUOTING_BID
- Otherwise wait for next window → IDLE

### Inventory Exit Strategy

"Exit at market" on thin Kalshi books is dangerous. Instead, use **aggressive limit orders**:

- Exit price floor: `entry_price - 5c` (max acceptable loss per contract on forced exit)
- Post SELL YES at `max(best_yes_bid - 1c, entry_price - 5c)` — undercut the best bid slightly
- If not filled in 10 seconds, lower by 1c and retry (up to 3 attempts)
- After 3 failed attempts, use market order as last resort
- Log every forced exit with loss amount

### Timing Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quote window | 5-12 min to expiry | Enough time for round trip to complete |
| Hard cutoff | 2 min to expiry | Cancel all, avoid settlement exposure |
| Poll interval | 3 seconds | Balance responsiveness vs rate limits |
| Re-quote threshold | Mid drifts > 1c | Keep bid competitive |
| Inventory timeout | 5 minutes | Give ask time to fill on thin books |
| Dashboard refresh | Tied to poll cycle (3s) | Only redraw on new data |
| Market discovery retry | 10s with backoff | Handle gap between windows |

### API Call Budget Per Poll Cycle (per asset)

| Call | Type | Count |
|------|------|-------|
| Get orderbook | Read | 1 |
| Get order status | Read | 1 |
| Cancel stale order | Write | 0-1 |
| Place new order | Write | 0-1 |
| **Total** | | **2 reads + 0-2 writes** |

With 2 assets at 3-second polling: ~1.3 reads/sec, ~0.7 writes/sec max. Well within Basic tier (20 read/sec, 10 write/sec).

## VPIN Kill Switch

### VPIN Computation

```
VPIN = |V_buy - V_sell| / (V_buy + V_sell)
```

We compute VPIN primarily from **BinanceUS spot trade flow**, which leads Kalshi contract pricing by seconds. Crypto spot is where informed traders act first (larger, more liquid market).

**Spot VPIN (primary signal):**
- Source: `data/market_data.get_trade_flow(symbol, limit=200)`
- Returns `net_flow` in range [-1, +1] and `buy_ratio` in [0, 1]
- VPIN proxy: `abs(net_flow)` — already computed as `|buy_vol - sell_vol| / total_vol`
- Note: `get_trade_flow()` returns the last 200 trades without time filtering. Since BTC/ETH trade frequently on BinanceUS (~50-100 trades/minute), 200 trades ≈ 2-4 minutes — close enough to our 2-minute VPIN window.

**Kalshi orderbook heuristic (secondary signal):**
- Track total `yes_bid_volume` and `no_bid_volume` between consecutive polls
- Delta: `abs(delta_yes - delta_no) / (abs(delta_yes) + abs(delta_no) + 1)`
- This is a proxy — not true trade flow, but large one-sided orderbook shifts suggest informed positioning

**Blended VPIN:**
```
vpin = 0.7 * spot_vpin + 0.3 * kalshi_ob_heuristic
```

Spot gets high weight because it is actual trade flow. Kalshi OB heuristic is noisy (order additions/cancels are not trades).

### VPIN Thresholds

| VPIN | State | Spread Action | Quote Action |
|------|-------|---------------|--------------|
| < 0.3 | SAFE | 2c target | Quote normally |
| 0.3 - 0.5 | CAUTION | 3-4c target | Widen spread |
| > 0.5 | TOXIC | N/A | **Cancel all, go dark** |

### Additional Kill Switch Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| Volatility spike | BTC spot moves > 0.5% in 60s | Cancel all, go dark |
| Daily loss limit | Day P&L < -$20 | Halt bot for the day |
| Window loss limit | Window P&L < -$10 | Skip to next window |
| API error | 2+ consecutive failures | Cancel all, go dark |
| One-sided fill + rising VPIN | Inventory held AND VPIN increased 2 consecutive polls | Exit via aggressive limit |

## Inventory Management

### Position Tracking

```python
@dataclass
class MMInventory:
    asset: str                  # "BTC" or "ETH"
    yes_held: int               # contracts held (from filled bids)
    entry_price_cents: int      # buy price in cents
    pending_bid_id: str | None  # resting buy order ID
    pending_ask_id: str | None  # resting sell order ID
    bid_price_cents: int        # price of resting bid
    ask_price_cents: int        # price of resting ask
    window_pnl_cents: int       # P&L for current 15m window (cents)
    window_round_trips: int     # completed round trips this window
    inventory_since: float      # timestamp when inventory acquired (0 if none)
    state: str                  # IDLE, DISCOVERING, QUOTING_BID, QUOTING_ASK, DARK
```

### Position Sizing

```python
def compute_contracts(balance_cents: int, entry_price_cents: int) -> int | None:
    """Compute contracts to quote. Returns None if insufficient budget."""
    risk_budget = int(balance_cents * RISK_BUDGET_PCT)
    contracts = risk_budget // entry_price_cents
    contracts = min(contracts, MAX_CONTRACTS_PER_ASSET)  # enforce cap
    if contracts < MIN_CONTRACTS_PER_QUOTE:
        return None  # budget too small for fee-efficient quoting, skip
    return contracts
```

Risk budget (5% of balance) always takes precedence. If the budget can't support 10 contracts (minimum for fee amortization), skip the quote entirely rather than overallocating.

| Parameter | Value |
|-----------|-------|
| Risk budget per quote | 5% of Kalshi balance |
| Minimum contracts | 10 (fee amortization — skip if budget can't support) |
| Max inventory per asset | 50 contracts |
| Inventory timeout | 5 minutes — transition to EXITING state |

### P&L Calculation (all in cents)

```python
def calc_maker_fee_cents(contracts: int, price_cents: int) -> int:
    """Kalshi maker fee in cents, rounded up."""
    p = price_cents / 100  # convert to dollar decimal
    fee_dollars = 0.0175 * contracts * p * (1 - p)
    return math.ceil(fee_dollars * 100)  # round up to nearest cent

def calc_round_trip_pnl(buy_cents: int, sell_cents: int, contracts: int) -> int:
    """Net P&L in cents for a completed round trip."""
    gross = (sell_cents - buy_cents) * contracts
    fee_buy = calc_maker_fee_cents(contracts, buy_cents)
    fee_sell = calc_maker_fee_cents(contracts, sell_cents)
    return gross - fee_buy - fee_sell
```

## Quote Strategy

### Mid Price Calculation

```python
def compute_mid_cents(orderbook: dict) -> int | None:
    """Compute mid price in cents from Kalshi orderbook.

    Kalshi orderbook format:
      orderbook_fp.yes_dollars: [[price_str, count_str], ...]
      orderbook_fp.no_dollars:  [[price_str, count_str], ...]

    Only bids are shown. A YES ask = 100 - best NO bid.
    """
    ob = orderbook.get("orderbook_fp", orderbook)
    yes_bids = ob.get("yes_dollars", [])
    no_bids = ob.get("no_dollars", [])

    if not yes_bids or not no_bids:
        return None

    best_yes_bid = round(float(yes_bids[0][0]) * 100)  # e.g., "0.4800" → 48
    best_no_bid = round(float(no_bids[0][0]) * 100)    # e.g., "0.5000" → 50
    implied_yes_ask = 100 - best_no_bid                  # 100 - 50 = 50

    mid = (best_yes_bid + implied_yes_ask) // 2
    return mid
```

### Spread Sizing

```python
def compute_spread_cents(vpin: float) -> int | None:
    """Dynamic spread based on VPIN. Returns None if should go dark."""
    if vpin < 0.3:
        return 2          # tight — safe flow
    elif vpin < 0.5:
        # Linear scale from 3c to 4c
        return 3 + round((vpin - 0.3) / 0.2)
    else:
        return None        # go dark — toxic flow
```

### Quote Placement

```python
# Phase 1: Bid
bid_cents = mid_cents - spread_cents // 2

# Boundary check — skip if mid too close to edge
if not (1 <= bid_cents <= 99):
    return None  # skip this cycle, mid too extreme

# Phase 2: Ask (after bid fills)
ask_cents = entry_price_cents + spread_cents

if not (1 <= ask_cents <= 99) or ask_cents <= entry_price_cents:
    # Can't place valid ask — exit inventory via EXITING state
    transition_to_exiting()
```

### Re-Quote Logic

On each poll cycle in QUOTING_BID state:
1. Fetch orderbook → compute new mid
2. If `|new_mid - bid_cents| > 1` → `cancel_order_safe()` bid, re-post at new mid
   - If cancel returns `"filled"` → treat as fill, transition to QUOTING_ASK
3. If bid filled → record entry, transition to QUOTING_ASK, post ask

On each poll cycle in QUOTING_ASK state:
1. Check if ask filled
2. If filled → log round trip profit → if >= 5 min to expiry, transition to QUOTING_BID; else IDLE
3. If inventory timeout reached → transition to EXITING
4. Do NOT re-quote asks based on mid drift (we want to capture our target spread from entry)

All cancel operations use `cancel_order_safe()` which returns `{"status": "filled"}` on 404 instead of throwing. The caller treats this as a fill event.

## Live ASCII Dashboard

Refreshes every **3 seconds** (tied to poll cycle — only redraws on new data).

```
╔═══════════════════════════════════════════════════════════════════════╗
║  KALSHI MARKET MAKER  │  2026-03-25 14:32:17  │  LIVE              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  BALANCE: $142.50  │  Day P&L: +$3.82  │  Round Trips: 14          ║
╠═══════════════════════════════════════════════════════════════════════╣
║  BTC (KXBTC15M-26mar1430)  exp 11:28  │  VPIN: 0.21 [SAFE]        ║
║  ├─ Mid: 52c  Spread: 2c  │  State: QUOTING_BID                   ║
║  ├─ BID: 10x YES @ 51c [RESTING]                                   ║
║  ├─ Inventory: 0  │  Window RTs: 3  │  Window P&L: +$0.62         ║
║  └─ Last RT: BUY 10x@51c → SELL 10x@53c → +$0.20 net (14:31:44)  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  ETH (KXETH15M-26mar1430)  exp 11:28  │  VPIN: 0.38 [CAUTION]     ║
║  ├─ Mid: 47c  Spread: 4c  │  State: QUOTING_ASK                   ║
║  ├─ ASK: 8x YES @ 49c [RESTING]  │  Held: 8x @ 45c (2m ago)      ║
║  ├─ Inventory: 8  │  Window RTs: 1  │  Window P&L: +$0.18         ║
║  └─ Last RT: BUY 8x@44c → SELL 8x@47c → +$0.18 net (14:28:10)    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  KILL SWITCH: OK  │  Daily loss remaining: $16.18                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  RECENT TRADES                                                       ║
║  14:31:44  BTC  SELL 10x YES@53c  RT#3  → +$0.20 net               ║
║  14:31:02  BTC  BUY  10x YES@51c                                    ║
║  14:30:22  ETH  SELL  8x YES@47c  RT#1  → +$0.18 net               ║
║  14:29:50  ETH  BUY   8x YES@44c                                    ║
║  14:29:15  BTC  SELL 10x YES@54c  RT#2  → +$0.20 net               ║
╠═══════════════════════════════════════════════════════════════════════╣
║  SESSION: 14 RTs │ Avg spread: 2.8c │ Avg net: 1.9c/RT             ║
║  Forced exits: 1 │ VPIN events: 2 caution, 0 kills │ Up: 2h 14m   ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Dashboard Data Sources

| Section | Source |
|---------|--------|
| Balance | `KalshiClient.get_balance()` (polled every 30s) |
| Day P&L | Internal accumulator, persisted to JSON |
| Per-asset state | `MMInventory` dataclass per asset |
| VPIN | `mm_vpin.compute()` every poll cycle |
| Kill switch | `mm_vpin.kill_switch_status()` |
| Recent trades | Ring buffer of last 20 fills |
| Session stats | Running counters (round trips, forced exits, VPIN events) |

### Dashboard Output

- **Stdout** — live terminal display (clear + reprint every 3s)
- **Log file** — `data/store/mm_dashboard.log` (latest frame, for MCP `algotrade_mm_status`)
- **Trade log** — `data/store/mm_trades.csv` (append-only, all fills with timestamps and P&L)
- **Session stats** — `data/store/mm_session.json` (persisted for restarts)

## Configuration (mm_config.py)

```python
# Assets
MM_ASSETS = {
    "BTC/USDT": "KXBTC15M",
    "ETH/USDT": "KXETH15M",
}

# Spread (in cents)
SPREAD_MIN_CENTS = 2          # 2c minimum spread
SPREAD_DEFAULT_CENTS = 3      # 3c default
SPREAD_MAX_CENTS = 4          # 4c max (caution mode)

# VPIN thresholds
VPIN_SAFE = 0.3
VPIN_CAUTION = 0.5            # above this → go dark
VPIN_WINDOW_SECONDS = 120     # rolling 2-min window
VPIN_SPOT_WEIGHT = 0.7        # spot flow weight in blended VPIN
VPIN_KALSHI_WEIGHT = 0.3      # kalshi OB heuristic weight

# Timing (seconds)
POLL_INTERVAL = 3
QUOTE_WINDOW_MIN_MINUTES = 5  # min minutes to expiry to start quoting
QUOTE_WINDOW_MAX_MINUTES = 12 # max minutes to expiry to start quoting
HARD_CUTOFF_MINUTES = 2       # cancel all at this many minutes to expiry
DISCOVERY_RETRY_SECONDS = 10  # retry market discovery interval
FORCED_EXIT_RETRY_SECONDS = 10  # time between aggressive limit attempts

# Inventory
MIN_CONTRACTS_PER_QUOTE = 10  # minimum for fee amortization
MAX_CONTRACTS_PER_ASSET = 50
INVENTORY_TIMEOUT_SECONDS = 300  # 5 minutes
RISK_BUDGET_PCT = 0.05        # 5% of balance per quote

# Risk
MAX_DAILY_LOSS_CENTS = 2000   # $20 — halt bot for the day
MAX_WINDOW_LOSS_CENTS = 1000  # $10 — skip to next window
VOLATILITY_SPIKE_PCT = 0.005  # 0.5% spot move in 60s → go dark
MAX_EXIT_LOSS_CENTS = 5       # 5c max loss per contract on forced exit

# Re-quote
REQUOTE_DRIFT_CENTS = 1       # 1c mid drift → re-quote bid
```

## Changes to Existing Code

### exchange/kalshi.py

Add `action` parameter to `place_order()` with default `"buy"` (backward-compatible — no existing callers pass `action`, so they continue getting `"buy"`):

```python
def place_order(self, ticker, side, count, price_cents=None,
                order_type="market", action="buy"):
    data = {
        "ticker": ticker,
        "action": action,     # "buy" or "sell"
        "side": side,
        "type": order_type,
        "count": count,
    }
    if order_type == "limit" and price_cents is not None:
        if side == "yes":
            data["yes_price"] = price_cents
        else:
            data["no_price"] = price_cents
    return self._post("/trade-api/v2/portfolio/orders", data)
```

Add `get_order_status()` for checking individual order fills:

```python
def get_order_status(self, order_id):
    return self._get(f"/trade-api/v2/portfolio/orders/{order_id}")
```

Add `cancel_order_safe()` that handles the race condition where an order fills between check and cancel:

```python
def cancel_order_safe(self, order_id):
    """Cancel an order, returning status dict.
    Returns {"status": "cancelled"} on success.
    Returns {"status": "filled"} if order was already filled (404/error).
    """
    try:
        return self._delete(f"/trade-api/v2/portfolio/orders/{order_id}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"status": "filled"}  # order already filled
        raise
```

### mcp_server.py

Add 3 new MCP tools:
- `algotrade_mm_start(mode="dry-run"|"live", assets=["BTC","ETH"])` — spawn mm_daemon subprocess, refuse if predictor is running
- `algotrade_mm_stop()` — graceful shutdown (cancel MM orders, exit inventory)
- `algotrade_mm_status()` — read `data/store/mm_dashboard.log`

## Dry-Run Mode

Dry-run uses real orderbook data but simulates fills locally:
- Reads real Kalshi orderbook for mid price and spread
- Simulates bid fill when orderbook YES ask crosses our bid price
- Simulates ask fill when orderbook YES bid crosses our ask price
- Tracks P&L identically to live mode
- Dashboard shows `[DRY-RUN]` in header
- Does NOT call any write endpoints (place_order, cancel_order)

## Testing Strategy

Unit tests for:
- `mm_vpin.py` — VPIN computation, threshold transitions, kill switch triggers
- `mm_strategy.py` — Mid price calculation from real orderbook format, spread sizing, quote generation
- `mm_inventory.py` — P&L calculation at various price points and contract counts, fee math verification, inventory limits, timeout logic
- Fee edge cases: 1 contract vs 10 vs 50 at different price points

Integration tests:
- Full state machine cycle with mocked Kalshi API — DISCOVERING → QUOTING_BID → QUOTING_ASK → profit → re-quote
- Kill switch activation mid-window — verify all orders cancelled
- One-sided fill → timeout → aggressive limit exit sequence
- Window expiry hard cutoff cleanup
- Cancel race condition — cancel returns 404, verify treated as fill
- Market discovery retry with backoff between windows

## Decisions Made (resolved from v1 open questions)

1. **Only quote YES side** — sequential BUY YES → SELL YES. No NO-side quoting. Simpler inventory model, one contract type to track.
2. **Run when started** — no built-in schedule. User starts/stops via MCP tools. Can run 24/7 if desired.
3. **Persist VPIN history** — yes, append to `data/store/mm_vpin.csv` for post-session analysis of flow patterns.
