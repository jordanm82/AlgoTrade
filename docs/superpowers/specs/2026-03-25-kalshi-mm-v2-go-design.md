# Kalshi Market Maker V2 — Go + WebSocket + Avellaneda-Stoikov

**Date:** 2026-03-25
**Status:** Draft
**Language:** Go
**Project:** New standalone repo (separate from algotrade)

---

## Historical Context

### V1 Market Maker (Python, this repo)

We built a Python market maker in `kalshi_mm/` that attempted to capture bid/ask spread on Kalshi 15-minute crypto UP/DOWN contracts. The approach was sequential: BUY YES at (mid - spread/2), wait for fill, then SELL YES at (entry + spread).

**What worked:**
- State machine architecture (IDLE → DISCOVERING → QUOTING_BID → QUOTING_ASK → EXITING → DARK)
- VPIN-based kill switch detected toxic flow and prevented entries during adverse conditions
- Contract price volatility check (MID_MOVE_DARK) caught rapid repricing events
- Dry-run completed profitable round trips: +$1.57 in one window (ETH: 3 RTs for +$1.08)
- Volume-based fill simulation using orderbook delta tracking

**What failed in live trading:**
1. **Sell mechanic is broken by design.** On Kalshi, `action: "sell"` is internally buying the opposite side, requiring ADDITIONAL balance. With 4 assets deploying capital simultaneously, no balance remained for sell orders. Positions went to settlement as unhedged directional bets. **Lost ~$91 on first live run.**
2. **Position limit race conditions.** Runners cycled through DISCOVERING → DARK → IDLE between polls, slipping through the position gate. Multiple assets deployed capital simultaneously despite MAX_CONCURRENT=1. Required 5 iterations to get a bulletproof single-runner-per-cycle fix.
3. **Re-quote chased price down.** When mid dropped, the bot cancelled and re-placed bids at lower prices. Cancel was slow, old bid filled, new bid also filled — 7 contracts instead of 1 in safe mode. **Lost ~$4 on second live run.**
4. **Fills didn't happen in dry-run.** Our bid sat 1c behind the best bid (mid - spread//2 = mid - 1). Volume-based simulation required trades at our exact price level, but all trades happened at the best bid above us.
5. **REST API polling at 3s too slow.** Missed fills between polls, couldn't react to orderbook changes in real-time, cancel-check-repost cycle introduced race conditions.

**Key lessons:**
- Never "sell" on Kalshi — use YES bid + NO bid (both are buys, automatically paired for $1)
- Position limits must be enforced at the loop level, not per-runner state
- Never re-quote downward (chasing = directional betting)
- Need WebSocket for real-time fills and orderbook, not REST polling
- BTC 15m spread is 1c (too tight), SOL/XRP have 3c spreads (viable)
- VPIN from BinanceUS spot flow is useful but doesn't catch Kalshi-specific repricing

### Research Findings

Comprehensive research of working Kalshi/Polymarket MM bots revealed:

**Working implementations:**
- [rodlaf/KalshiMarketMaker](https://github.com/rodlaf/KalshiMarketMaker) — Avellaneda-Stoikov on Kalshi, Python, Fly.io deployment. Key params: gamma=0.2, k=1.5, sigma=0.001, min_spread=2c, max_position=3/market, max_global=20. Selects top 6 markets by volume/spread score.
- [nikhilnd/kalshi-market-making](https://github.com/nikhilnd/kalshi-market-making) — Cauchy distribution model, 51 trades, $6.80 profit on $33.40 (20.3% in one day).
- [Polymarket MM ($700-800/day)](https://news.polymarket.com/p/automated-market-making-on-polymarket) — $10k capital, found low-risk markets with same reward rates as high-risk ones.

**The correct Kalshi MM mechanic:**
```
Place YES bid at 47c  (cost: 47c)
Place NO  bid at 47c  (cost: 47c, equivalent to YES ask at 53c)
Both fill → hold YES + NO → Kalshi pairs for $1
Cost: 94c → Payout: 100c → Profit: 6c gross
```
No selling. Both sides are buys. Kalshi automatically pairs YES+NO positions.

**Academic findings (Whelan/CEPR):**
- Takers lose ~32% on average; Makers lose ~10%
- Being a maker is structurally advantaged
- The edge comes from having a BETTER probability model, not just spread capture
- Favorite-longshot bias: cheap contracts win less than stated, expensive win more

**Fee structure:**
- Maker: `ceil(0.0175 * C * P * (1-P))` — max ~0.44c/contract at 50c
- Taker: `ceil(0.07 * C * P * (1-P))` — max ~1.75c/contract at 50c
- Break-even at 50c with both sides as maker: ~0.88c → need 1c+ gross spread
- At price extremes (10c/90c): fees drop to ~0.16c/side → much easier to profit

**What profitable strategies have in common:**
1. Two-sided quoting (YES + NO) is mandatory
2. Inventory management is the #1 risk (binary settlement = total loss on wrong side)
3. Market selection matters more than quote optimization
4. Work in logit space, not probability space (catastrophic failure otherwise)
5. Time-to-settlement risk: reduce inventory aggressively near expiry
6. Need markets with 2c+ native spread and adequate volume

---

## V2 Design: Go + WebSocket + Avellaneda-Stoikov

### Overview

A standalone Go market maker for Kalshi 15-minute crypto contracts. Uses WebSocket for real-time orderbook deltas and fill notifications. Implements Avellaneda-Stoikov for optimal quote placement with inventory skewing. Quotes both YES and NO sides simultaneously — no selling required.

### Why Go

- **WebSocket native**: gorilla/websocket, sub-ms event processing
- **Concurrency**: goroutines for per-market state machines + shared inventory manager
- **Single binary**: no Python venv, no dependency hell, easy deployment
- **Speed**: order placement in <1ms vs Python's 10-50ms
- **The KalshiMarketMaker (Python) works but is slow**: 5s refresh cycle. Go can react to every WebSocket delta in real-time

### Why WebSocket

Kalshi WebSocket (`wss://api.elections.kalshi.com/trade-api/ws/v2`):
- `orderbook_delta` channel: snapshot on subscribe + incremental updates
- `fill` channel: instant fill notifications (no polling!)
- `ticker` channel: real-time bid/ask
- `market_lifecycle_v2`: market open/close events
- Same RSA-PSS auth as REST (headers on handshake)

This eliminates every race condition from V1:
- Fills detected instantly (no 3s poll gap)
- Orderbook changes seen in real-time (no stale mid prices)
- Market open/close detected immediately (no discovery polling)

### Core Architecture

```
┌─────────────────────────────────────────────┐
│                 Main Process                 │
├─────────────────────────────────────────────┤
│  WebSocket Client (goroutine)               │
│  ├─ orderbook_delta → OrderbookManager      │
│  ├─ fill            → FillHandler           │
│  ├─ ticker          → PriceTracker          │
│  └─ market_lifecycle → MarketDiscovery      │
├─────────────────────────────────────────────┤
│  Avellaneda-Stoikov Engine                  │
│  ├─ Reservation price (per market)          │
│  ├─ Optimal spread (per market)             │
│  ├─ Inventory skew (global + per market)    │
│  └─ Logit-space probability model           │
├─────────────────────────────────────────────┤
│  Order Manager                              │
│  ├─ Place YES bid + NO bid simultaneously   │
│  ├─ Cancel/replace on quote update          │
│  ├─ Track pending orders by ID              │
│  └─ REST API for order placement            │
├─────────────────────────────────────────────┤
│  Risk Manager                               │
│  ├─ Per-market position limits              │
│  ├─ Global contract limit                   │
│  ├─ VPIN / flow toxicity monitor            │
│  ├─ Contract price volatility (mid-move)    │
│  ├─ Time-to-settlement inventory reduction  │
│  └─ Daily P&L halt                          │
├─────────────────────────────────────────────┤
│  Dashboard (terminal UI)                    │
│  └─ Real-time: positions, P&L, VPIN, fills  │
└─────────────────────────────────────────────┘
```

### Avellaneda-Stoikov Model

**Work in logit space** (critical — probability space fails at extremes):
```
logit(p) = ln(p / (1-p))
```

**Reservation price** (shifts with inventory to discourage accumulation):
```
r = s - q * gamma * sigma_b^2 * (T - t)

s     = current mid price (in logit space)
q     = net inventory (positive = long YES, negative = long NO)
gamma = risk aversion (0.1-0.5, higher = more conservative)
sigma_b = belief volatility (estimated via Kalman filter on mid changes)
T - t = time remaining to settlement (in fraction of window)
```

**Optimal spread:**
```
delta = gamma * sigma_b^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

k = order book liquidity parameter (estimated from book depth)
```

**Quote placement:**
```
yes_bid = logit_inv(r - delta/2)   →  convert back to price cents
no_bid  = 100 - logit_inv(r + delta/2)   →  the "ask" side as a NO bid

If both fill: profit = (yes_bid + no_bid cost) vs $1 payout
```

**Inventory skewing in action:**
```
Holding 3 YES, 0 NO (q=+3):
  r shifts DOWN → yes_bid drops (harder to buy more YES)
                → no_bid rises (easier to fill NO, rebalance)

Holding 0 YES, 2 NO (q=-2):
  r shifts UP   → yes_bid rises (easier to fill YES, rebalance)
                → no_bid drops (harder to buy more NO)
```

### Dual-Sided Quoting Flow

```
1. DISCOVER: WebSocket market_lifecycle → new 15m window opens
2. SUBSCRIBE: Subscribe to orderbook_delta for the market ticker
3. SNAPSHOT: Receive full orderbook → compute mid, spread, sigma
4. QUOTE: Place YES bid + NO bid simultaneously via REST
5. MONITOR: WebSocket deltas update local orderbook in real-time
   - On orderbook change: recompute A-S quotes, cancel/replace if shifted
   - On YES fill: update inventory (q++), widen YES bid, tighten NO bid
   - On NO fill: update inventory (q--), widen NO bid, tighten YES bid
   - On BOTH filled: pair settles for $1, log profit, re-quote
6. WIND DOWN: At T-3min, stop new quotes, aggressively close inventory
7. SETTLE: At T-0, any remaining inventory settles at $0 or $1
```

### Market Selection

Not all 15m crypto markets are good for MM. Filter by:

| Criteria | Threshold | Rationale |
|----------|-----------|-----------|
| Native spread | >= 2c | Need room for our quotes + fees |
| 24h volume | >= 500 contracts | Need counterparties to fill |
| Mid price | 20c-80c | Fees peak at 50c, extremes have lower fees but less flow |
| Assets | ETH, SOL, XRP | BTC too tight (1c spread). BNB/DOGE if volume sufficient |

Score = `0.35 * volume_score + 0.65 * spread_score` (from KalshiMarketMaker repo).

Run A-S on top N markets simultaneously (start with 2-3, scale up).

### Risk Controls

| Control | Value | Mechanism |
|---------|-------|-----------|
| Max position per market | 5 contracts | A-S blocks new quotes beyond this |
| Max global contracts | 15 | Sum across all markets |
| Max inventory imbalance | 3:0 per market | If YES=3, NO=0 → block YES bids entirely |
| Daily loss halt | $20 | Stop all quoting for the day |
| Time-to-settlement | Last 3 min | Cancel all quotes, exit inventory aggressively |
| VPIN threshold | 0.8 (from V1 calibration) | Cancel quotes, go dark |
| Mid-move threshold | 5c/3s | Cancel quotes, go dark |
| Consecutive API errors | 2 | Cancel all, go dark |

### Kill Switch Layers

Same three layers from V1, plus WebSocket-aware:
1. **VPIN** (spot flow imbalance from BinanceUS) — >= 0.8 → dark
2. **Mid-move** (contract repricing) — >= 5c between updates → dark
3. **Fill imbalance** (NEW) — if one side fills 3x more than the other in a window → widen/pause
4. **WebSocket disconnect** — immediately cancel all resting orders via REST fallback

### Fee Optimization

- Always use limit orders (maker fee 0.0175 vs taker 0.07)
- At 50c: maker ~0.44c/side, need 1c+ spread to profit
- At extremes (20c/80c): maker ~0.28c/side, easier to profit
- Batch orders when possible (ceil() amortizes better over more contracts)
- Target: 2-4c gross spread → 1-3c net after fees

### Project Structure

```
kalshi-mm/
├── cmd/
│   └── kalshi-mm/
│       └── main.go              — Entry point, config loading, signal handling
├── internal/
│   ├── auth/
│   │   └── signer.go            — RSA-PSS request signing
│   ├── ws/
│   │   ├── client.go            — WebSocket connection, reconnect, subscribe
│   │   ├── messages.go          — Message types (snapshot, delta, fill, ticker)
│   │   └── handler.go           — Route messages to handlers
│   ├── rest/
│   │   ├── client.go            — REST API client (orders, balance, markets)
│   │   └── orders.go            — Place, cancel, amend orders
│   ├── orderbook/
│   │   ├── book.go              — Local orderbook maintained from WS deltas
│   │   └── book_test.go
│   ├── engine/
│   │   ├── avellaneda_stoikov.go — A-S reservation price, spread, inventory skew
│   │   ├── avellaneda_stoikov_test.go
│   │   ├── quoter.go            — Translates A-S output to YES bid + NO bid orders
│   │   └── quoter_test.go
│   ├── risk/
│   │   ├── manager.go           — Position limits, daily P&L, kill switch
│   │   ├── vpin.go              — VPIN from BinanceUS spot flow
│   │   └── vpin_test.go
│   ├── market/
│   │   ├── selector.go          — Market discovery + scoring (volume/spread)
│   │   ├── lifecycle.go         — 15m window tracking, settlement countdown
│   │   └── selector_test.go
│   ├── inventory/
│   │   ├── tracker.go           — Per-market and global inventory tracking
│   │   ├── fees.go              — Kalshi fee calculation
│   │   ├── pnl.go               — P&L tracking, round trip detection
│   │   └── tracker_test.go
│   └── dashboard/
│       └── terminal.go          — Live ASCII dashboard
├── config/
│   └── config.go                — YAML config loading, defaults
├── config.yaml                  — User config (API keys, thresholds, assets)
├── go.mod
├── go.sum
├── Makefile
└── README.md
```

### Configuration (config.yaml)

```yaml
kalshi:
  api_key_id: "your-key-id"
  private_key_path: "/path/to/KalshiPrimaryKey.txt"
  ws_url: "wss://api.elections.kalshi.com/trade-api/ws/v2"
  rest_url: "https://api.elections.kalshi.com"

assets:
  - symbol: "ETH/USDT"
    series: "KXETH15M"
  - symbol: "SOL/USDT"
    series: "KXSOL15M"
  - symbol: "XRP/USDT"
    series: "KXXRP15M"
  # BTC excluded — 1c native spread too tight

avellaneda_stoikov:
  gamma: 0.2              # risk aversion (0.1=aggressive, 0.5=conservative)
  k: 1.5                  # order book liquidity parameter
  sigma_window: 30        # seconds of mid-price history for volatility estimation
  min_spread_cents: 2     # minimum gross spread to quote
  max_spread_cents: 10    # maximum spread (wider = safer but fewer fills)

risk:
  max_position_per_market: 5
  max_global_contracts: 15
  max_daily_loss_cents: 2000    # $20
  vpin_dark_threshold: 0.8
  mid_move_dark_cents: 5
  settlement_cutoff_minutes: 3

inventory:
  risk_budget_pct: 0.10         # 10% of balance per market
  max_imbalance: 3              # max contracts on one side without the other

safe_mode:
  enabled: false
  max_contracts: 1
  stop_after_first_profit: true

dashboard:
  refresh_ms: 500               # dashboard refresh rate
```

### Dashboard

```
══════════════════════════════════════════════════════════════
 KALSHI MM v2 [LIVE]  │  22:15:33 UTC  │  Up: 1h 23m
 Balance: $78.30  │  Day P&L: +$2.14  │  Round Trips: 8
══════════════════════════════════════════════════════════════
 ETH (KXETH15M-26MAR251815-15)  exp 8.2m  │  VPIN: 0.41
 ├─ Mid: 62c  │  A-S spread: 4c  │  State: QUOTING
 ├─ YES bid: 3x @ 60c [RESTING]
 ├─ NO  bid: 3x @ 36c [RESTING]  (= YES ask @ 64c)
 ├─ Inventory: YES=1 NO=0 (q=+1)  │  Skew: -0.8c
 └─ Window: 2 RTs +38c  │  Last: YES+NO paired +12c (22:14:51)
──────────────────────────────────────────────────────────────
 SOL (KXSOL15M-26MAR251815-15)  exp 8.2m  │  VPIN: 0.33
 ├─ Mid: 45c  │  A-S spread: 3c  │  State: QUOTING
 ├─ YES bid: 4x @ 43c [RESTING]
 ├─ NO  bid: 4x @ 53c [RESTING]  (= YES ask @ 47c)
 ├─ Inventory: balanced  │  Skew: 0c
 └─ Window: 1 RT +18c  │  Last: YES+NO paired +18c (22:13:22)
──────────────────────────────────────────────────────────────
 XRP (KXXRP15M-26MAR251815-15)  exp 8.2m  │  VPIN: 0.28
 ├─ Mid: 38c  │  A-S spread: 3c  │  State: QUOTING
 ├─ YES bid: 5x @ 36c [RESTING]
 ├─ NO  bid: 5x @ 59c [RESTING]  (= YES ask @ 41c)
 ├─ Inventory: NO=2 (q=-2)  │  Skew: +1.6c
 └─ Window: 0 RTs  │  Awaiting fills
══════════════════════════════════════════════════════════════
 KILL SWITCH: OK  │  Global: 4 YES + 2 NO = 6/15
 RECENT FILLS:
   22:14:51  ETH  NO  fill 1x@36c → paired +12c net
   22:14:38  ETH  YES fill 1x@60c
   22:13:22  SOL  YES+NO paired 4x → +18c net
   22:12:55  XRP  NO  fill 2x@59c (inventory: q=-2)
══════════════════════════════════════════════════════════════
```

### Implementation Phases

**Phase 1: Foundation (Week 1)**
- Go project scaffolding, config loading
- RSA-PSS auth (same algorithm as Python, ported to Go)
- REST client: markets, orderbook, orders, balance
- Basic tests against Kalshi demo API

**Phase 2: WebSocket (Week 1-2)**
- WebSocket client with auto-reconnect
- Orderbook snapshot + delta processing → local book
- Fill notification handler
- Market lifecycle subscription (window open/close)

**Phase 3: Avellaneda-Stoikov Engine (Week 2)**
- Logit-space mid price computation
- Belief volatility estimation (rolling window on mid changes)
- Reservation price with inventory skew
- Optimal spread calculation
- Quote generation: A-S output → YES bid cents + NO bid cents
- Comprehensive unit tests for all A-S math

**Phase 4: Order Management + Inventory (Week 2-3)**
- Place YES bid + NO bid simultaneously
- Cancel/replace when A-S quotes shift
- Fill tracking → inventory update → A-S re-quote
- Round trip detection (YES+NO pair → log profit)
- Per-market and global position limits

**Phase 5: Risk + Kill Switch (Week 3)**
- VPIN from BinanceUS spot flow (HTTP polling, separate goroutine)
- Contract mid-move detection (from WS orderbook deltas)
- Fill imbalance detection
- Daily P&L halt
- Time-to-settlement inventory reduction
- WebSocket disconnect → REST cancel-all fallback

**Phase 6: Dashboard + Safe Mode (Week 3)**
- Terminal UI with real-time updates
- Safe mode flag (1 contract, 1 market, stop after first profit)
- Logging to file for post-session analysis

**Phase 7: Live Testing (Week 4)**
- Safe mode on demo API
- Safe mode on production (1 contract)
- Gradual scale-up: 2 contracts, 3 contracts, multiple markets

### What We Carry Forward From V1

| V1 Component | V2 Usage |
|---|---|
| VPIN thresholds (SAFE=0.6, CAUTION=0.8) | Calibrated from live crypto flow — use as starting point |
| Mid-move dark (5c) | Validated — catches contract repricing VPIN misses |
| Market selection (ETH/SOL/XRP, not BTC) | BTC 1c spread too tight, confirmed in live |
| Fee math (0.0175 maker formula) | Same formula, port to Go |
| Position sizing (10% of balance) | Same approach, with A-S inventory limits on top |
| Safe mode concept | Keep for initial live testing |
| Volume-based fill detection | Replaced by WebSocket fill notifications |
| REST polling loop | Replaced by WebSocket event-driven architecture |
| Sequential buy-then-sell | Replaced by simultaneous YES bid + NO bid |

### Open Questions

1. Should we use Kalshi's demo API for initial development, or go straight to production with safe mode?
2. Do we want to integrate our existing V1/V3 directional predictor as an "informed" skew on top of A-S? (The research suggests this is the highest-edge approach — "informed market making")
3. Should we support 5-minute contracts too? (Research shows 5m markets are overtaking 15m in volume)
4. Do we want to deploy on a VPS near Kalshi's servers (NYC area) for lower latency?

### References

- [Avellaneda & Stoikov (2008): High-frequency trading in a limit order book](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)
- [rodlaf/KalshiMarketMaker](https://github.com/rodlaf/KalshiMarketMaker) — Working A-S implementation on Kalshi
- [nikhilnd/kalshi-market-making](https://github.com/nikhilnd/kalshi-market-making) — Cauchy model, 20.3% single-day return
- [Polymarket Market-Making Bible](https://www.weex.com/news/detail/polymarket-market-making-bible-pricing-spread-formula-379890) — Logit space, Kalman filtering
- [Hummingbot A-S Guide](https://medium.com/hummingbot/a-comprehensive-guide-to-avellaneda-stoikovs-market-making-strategy-102d64bf5df6)
- [Kalshi WebSocket Docs](https://docs.kalshi.com/getting_started/quick_start_websockets)
- [Kalshi REST API](https://docs.kalshi.com/api-reference/orders/create-order)
- [kalshi-rs (Rust HFT-grade client)](https://github.com/rmadev01/kalshi-rs) — Reference for Go implementation
- [Economics of Kalshi (Whelan/CEPR)](https://cepr.org/voxeu/columns/economics-kalshi-prediction-market) — Makers outperform takers structurally
