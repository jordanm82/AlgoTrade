# Kalshi 15m Crypto Market Maker — Design Spec

**Date:** 2026-03-25
**Status:** Draft
**Assets:** BTC, ETH (KXBTC15M, KXETH15M)

## Overview

A separate, standalone market making bot for Kalshi 15-minute crypto UP/DOWN markets. The bot captures bid/ask spread by posting resting limit orders on both sides of YES contracts, profiting from the spread regardless of outcome. A VPIN-based kill switch detects toxic order flow and stops quoting before adverse selection destroys profits.

This is NOT a directional predictor. The bot does not predict price direction. It earns the spread by providing liquidity.

## Strategy: Spread Capture Market Making

### Core Mechanic

```
1. Post BUY YES at (mid - spread/2)   ← resting bid
2. Post SELL YES at (mid + spread/2)  ← resting ask
3. Both fill → profit = spread - fees
4. Re-quote immediately at new mid
5. Repeat until window expires or kill switch fires
```

### Why It Works

In Kalshi 15m crypto markets:
- Retail traders cross the spread to get immediate fills (they pay taker fees)
- Market makers post resting orders (pay lower maker fees)
- The spread between what buyers pay and sellers receive is the market maker's profit
- Outcome doesn't matter when both sides fill — one side wins $1, the other loses, net = spread captured

### Fee Math

Kalshi fee formula:
- **Maker fee:** `ceil(0.0175 * contracts * price * (1 - price))` per side
- **Taker fee:** `ceil(0.07 * contracts * price * (1 - price))` per side

At 50c price point (worst case — parabolic fee peaks here):
- Maker: ~0.44c/contract/side → ~0.88c round trip
- Taker: ~1.75c/contract/side → ~3.50c round trip

**Profit per round trip at target spreads:**

| Spread | Revenue | Fees (maker both sides) | Net Profit/Contract |
|--------|---------|------------------------|---------------------|
| 2c | 2.00c | ~0.88c | ~1.12c |
| 3c | 3.00c | ~0.88c | ~2.12c |
| 4c | 4.00c | ~0.88c | ~3.12c |

All orders are **limit orders** (maker). We never cross the spread (taker).

## Architecture

### File Structure

```
kalshi_mm/
├── mm_daemon.py          — Main event loop, dashboard rendering, lifecycle
├── mm_strategy.py        — Mid price calc, spread sizing, quote generation
├── mm_vpin.py            — VPIN computation, kill switch logic
├── mm_inventory.py       — Position tracking, P&L, risk limits
└── mm_config.py          — All configurable thresholds and parameters
```

### Reused Modules

| Module | Usage |
|--------|-------|
| `exchange/kalshi.py` | API client — add `action: "sell"` support to `place_order()` |
| `data/market_data.py` | BinanceUS trade flow for VPIN proxy signal |
| `data/fetcher.py` | Spot price for volatility monitoring |

### New MCP Tools

| Tool | Purpose |
|------|---------|
| `algotrade_mm_start` | Start market maker daemon (mode, assets) |
| `algotrade_mm_stop` | Graceful stop — cancel all orders, exit inventory |
| `algotrade_mm_status` | Dashboard state + VPIN levels + P&L |

## Market Making Loop

### Per-Asset State Machine

Each asset (BTC, ETH) runs independently with its own state:

```
IDLE → DISCOVERING → QUOTING → DARK → IDLE
```

**IDLE:** No active market. Waiting for next 15m window.

**DISCOVERING:** Searching for an active 15m market with 5-12 minutes to expiry. Read orderbook, compute mid and VPIN. If VPIN is safe, transition to QUOTING.

**QUOTING:** Actively posting bid/ask. Poll every 3 seconds:
- Check for fills (orders API)
- Update mid price from orderbook
- Recompute VPIN
- If mid drifts > 1c from quote mid → cancel and re-quote
- If both sides fill → log profit, immediately re-quote (new round trip)
- If VPIN > threshold → transition to DARK
- If < 2 min to expiry → cancel all, transition to IDLE

**DARK:** Kill switch active. All orders cancelled. No quoting.
- Continue monitoring VPIN
- If VPIN recovers AND > 4 min to expiry → transition back to QUOTING
- If holding inventory → exit at market
- Otherwise wait for next window → IDLE

### Timing Rules

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Quote window | 5-12 min to expiry | Enough time for both sides to fill |
| Hard cutoff | 2 min to expiry | Cancel all, avoid settlement exposure |
| Poll interval | 3 seconds | Balance responsiveness vs rate limits |
| Re-quote threshold | Mid drifts > 1c | Keep quotes competitive |
| Inventory timeout | 3 minutes | Exit one-sided fills that aren't completing |

### API Call Budget Per Poll Cycle (per asset)

| Call | Type | Count |
|------|------|-------|
| Get orderbook | Read | 1 |
| Get orders (check fills) | Read | 1 |
| Cancel stale order | Write | 0-2 |
| Place new order | Write | 0-2 |
| **Total** | | **2 reads + 0-4 writes** |

With 2 assets at 3-second polling: ~1.3 reads/sec, ~1.3 writes/sec max. Well within Basic tier (20 read/sec, 10 write/sec).

## VPIN Kill Switch

### VPIN Computation

```
VPIN = |V_buy - V_sell| / (V_buy + V_sell)
```

We compute VPIN from two sources, blended:

1. **Kalshi orderbook delta** — Track YES bid/ask volume changes between polls. Large one-sided volume shifts indicate informed flow.
2. **BinanceUS spot trade flow** — From `data/market_data.get_trade_flow()`. Crypto spot flow leads Kalshi contract pricing by seconds.

Blended VPIN = `0.4 * kalshi_vpin + 0.6 * spot_vpin`

Spot flow gets higher weight because it's where informed traders act first (larger, more liquid market). Kalshi contract prices follow spot.

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
| One-sided fill + rising VPIN | Inventory held AND VPIN trending up | Exit inventory at market |

### VPIN Decay

VPIN is computed over a rolling 2-minute window of trade flow data. This ensures it responds quickly to flow regime changes without being noisy from individual trades.

## Inventory Management

### Position Tracking

```python
@dataclass
class MMInventory:
    asset: str                  # "BTC" or "ETH"
    yes_held: int               # contracts held (from filled bids)
    avg_entry: float            # weighted average entry price
    pending_bid: Order | None   # resting buy order
    pending_ask: Order | None   # resting sell order
    window_pnl: float           # P&L for current 15m window
    window_trades: int          # completed round trips this window
    inventory_since: float      # timestamp when inventory acquired
```

### Inventory Limits

| Parameter | Value |
|-----------|-------|
| Max contracts per quote | `risk_budget / entry_price` |
| Risk budget per quote | 5% of Kalshi balance |
| Max inventory per asset | 50 contracts |
| Inventory timeout | 3 minutes — exit at market if ask hasn't filled |

### One-Sided Fill Handling

When a bid fills but the ask hasn't:

1. **Immediately post ask** at (entry + spread) if not already posted
2. Monitor VPIN — if rising, widen the ask or exit at market
3. If **3 minutes** pass with no ask fill → exit at market (cut losses)
4. If VPIN goes TOXIC → immediate market exit

When an ask fills but we have no inventory (shouldn't happen — we only post asks when holding):
- This is a logic error, log and investigate

### P&L Calculation

```
gross_profit = (sell_price - buy_price) * contracts
maker_fee_buy = ceil(0.0175 * contracts * buy_price * (1 - buy_price))
maker_fee_sell = ceil(0.0175 * contracts * sell_price * (1 - sell_price))
net_profit = gross_profit - maker_fee_buy - maker_fee_sell
```

All P&L tracked in cents internally, displayed in dollars.

## Quote Strategy

### Mid Price Calculation

```python
def compute_mid(orderbook):
    best_yes_bid = orderbook["yes"][0]["price"]   # highest yes bid
    best_no_bid = orderbook["no"][0]["price"]      # highest no bid
    yes_ask = 1.0 - best_no_bid                    # implied yes ask
    mid = (best_yes_bid + yes_ask) / 2
    return mid
```

In Kalshi's orderbook, only bids are shown. A YES ask = `1.0 - best NO bid`.

### Spread Sizing

```python
def compute_spread(vpin, base_spread=0.03):
    if vpin < 0.3:
        return 0.02          # tight — safe flow
    elif vpin < 0.5:
        return base_spread + (vpin - 0.3) * 0.05  # linear widen 3-4c
    else:
        return None           # go dark
```

### Quote Placement

```python
bid_price = round_to_cent(mid - spread / 2)
ask_price = round_to_cent(mid + spread / 2)

# Sanity checks
assert bid_price >= 0.01 and bid_price <= 0.99
assert ask_price >= 0.01 and ask_price <= 0.99
assert ask_price > bid_price
assert (ask_price - bid_price) >= 0.02  # minimum 2c spread
```

All prices in dollars (Kalshi API uses dollar strings like "0.4800").

### Re-Quote Logic

On each poll cycle:
1. Fetch current orderbook → compute new mid
2. If `|new_mid - quote_mid| > 0.01` → cancel and re-quote at new mid
3. If pending bid filled → post ask if not already posted
4. If pending ask filled → log round trip profit, post new bid+ask (new round trip)
5. If both filled between polls → log profit, re-quote

## Live ASCII Dashboard

Refreshes every **2 seconds** via terminal clear + reprint.

```
╔═══════════════════════════════════════════════════════════════════════╗
║  KALSHI MARKET MAKER  │  2026-03-25 14:32:17  │  RUNNING            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  BALANCE: $142.50  │  Day P&L: +$3.82  │  Round Trips: 14          ║
╠═══════════════════════════════════════════════════════════════════════╣
║  BTC (KXBTC15M-26mar1430)  exp 11:28  │  VPIN: 0.21 [SAFE]        ║
║  ├─ Mid: 52c  Spread: 2c  │  State: QUOTING                       ║
║  ├─ BID: 10x YES @ 51c [RESTING]                                   ║
║  ├─ ASK: 10x YES @ 53c [RESTING]                                   ║
║  ├─ Inventory: 0  │  Window RTs: 3  │  Window P&L: +$0.62         ║
║  └─ Last fill: SELL 10x @ 53c (14:31:44) → +$0.21 net             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  ETH (KXETH15M-26mar1430)  exp 11:28  │  VPIN: 0.38 [CAUTION]     ║
║  ├─ Mid: 47c  Spread: 4c  │  State: QUOTING (wide)                ║
║  ├─ BID: 8x YES @ 45c [RESTING]                                    ║
║  ├─ ASK: 8x YES @ 49c [FILLED]                                     ║
║  ├─ Inventory: 0  │  Window RTs: 1  │  Window P&L: +$0.18         ║
║  └─ Last fill: SELL 8x @ 49c (14:30:22) → +$0.18 net              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  KILL SWITCH: OK  │  Daily loss remaining: $16.18                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  RECENT TRADES                                                       ║
║  14:31:44  BTC  SELL 10x YES @ 53c  (maker)  RT#3  → +$0.21 net   ║
║  14:31:02  BTC  BUY  10x YES @ 51c  (maker)                        ║
║  14:30:22  ETH  SELL  8x YES @ 49c  (maker)  RT#1  → +$0.18 net   ║
║  14:29:50  ETH  BUY   8x YES @ 45c  (maker)                        ║
║  14:29:15  BTC  SELL 10x YES @ 54c  (maker)  RT#2  → +$0.21 net   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  SESSION: 14 RTs │ Avg spread: 2.8c │ Avg net: 1.9c/RT            ║
║  VPIN events: 2 caution │ 0 kills │ Uptime: 2h 14m                ║
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
| Session stats | Running counters |

### Dashboard Output

- **Stdout** — live terminal display (clear + reprint every 2s)
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

# Spread
SPREAD_MIN = 0.02          # 2c minimum spread
SPREAD_DEFAULT = 0.03      # 3c default
SPREAD_MAX = 0.04          # 4c max (caution mode)

# VPIN thresholds
VPIN_SAFE = 0.3
VPIN_CAUTION = 0.5          # above this → go dark
VPIN_WINDOW_SECONDS = 120   # rolling 2-min window
VPIN_SPOT_WEIGHT = 0.6      # spot flow weight in blended VPIN
VPIN_KALSHI_WEIGHT = 0.4    # kalshi orderbook delta weight

# Timing
POLL_INTERVAL = 3           # seconds between poll cycles
QUOTE_WINDOW_MIN = 5        # min minutes to expiry to start quoting
QUOTE_WINDOW_MAX = 12       # max minutes to expiry to start quoting
HARD_CUTOFF_MIN = 2         # cancel all at this many minutes to expiry
DASHBOARD_REFRESH = 2       # seconds between dashboard redraws

# Inventory
MAX_CONTRACTS_PER_QUOTE = 50
INVENTORY_TIMEOUT = 180     # 3 minutes — exit one-sided fills
RISK_BUDGET_PCT = 0.05      # 5% of balance per quote

# Risk
MAX_DAILY_LOSS = 20.00      # dollars — halt bot
MAX_WINDOW_LOSS = 10.00     # dollars — skip to next window
VOLATILITY_SPIKE_PCT = 0.005  # 0.5% spot move in 60s → go dark

# Re-quote
REQUOTE_DRIFT_THRESHOLD = 0.01  # 1c mid drift → re-quote
```

## Changes to Existing Code

### exchange/kalshi.py

Add sell support to `place_order()`:

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
    ...
```

Add `get_order_status()` method for checking individual order fills:

```python
def get_order_status(self, order_id):
    return self._get(f"/trade-api/v2/portfolio/orders/{order_id}")
```

### mcp_server.py

Add 3 new MCP tools:
- `algotrade_mm_start(mode="dry-run"|"live", assets=["BTC","ETH"])` — spawn mm_daemon subprocess
- `algotrade_mm_stop()` — graceful shutdown
- `algotrade_mm_status()` — read `data/store/mm_dashboard.log`

## Dry-Run Mode

Dry-run simulates fills locally without hitting Kalshi order API:
- Posts quotes to a local orderbook simulation
- Simulates fills when price crosses bid/ask (based on real orderbook data)
- Tracks P&L identically to live mode
- Dashboard shows `[DRY-RUN]` in header

## Testing Strategy

Unit tests for:
- `mm_vpin.py` — VPIN computation, threshold transitions, kill switch triggers
- `mm_strategy.py` — Mid price calculation, spread sizing, quote generation
- `mm_inventory.py` — P&L calculation, inventory limits, timeout logic
- Fee calculations at various price points

Integration tests:
- Full loop with mocked Kalshi API — quote, fill, re-quote cycle
- Kill switch activation mid-window
- One-sided fill → timeout → exit sequence
- Window expiry cleanup

## Open Questions

1. Should we also quote on the NO side (in addition to YES) to capture opportunities on both sides of the book?
2. Should the bot run 24/7 or only during high-volume hours?
3. Should we persist VPIN history for analysis of flow patterns over time?
