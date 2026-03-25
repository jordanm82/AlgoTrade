# K15 Rich Interactive Dashboard

**Date:** 2026-03-25
**Goal:** Build a `rich`-powered interactive terminal dashboard (`k15.py`) for the Kalshi V3 predictor with live-updating panels and a command prompt for start/stop/control.

## Problem

The current `dashboard.py` outputs plain text that overwrites itself every tick. It has no interactivity (can't start/stop without killing the process), no color-coded status, and no activity log. Running the bot requires separate terminal commands and monitoring requires reading log files.

## Design

### 1. Layout

Two-panel layout using `rich.live.Live` with `rich.layout.Layout`:

```
┌─ K15 UPDOWN [DRY-RUN] V3 ──────────────────┬─ ACTIVITY LOG ──────────────────┐
│                                              │                                 │
│  KALSHI BALANCE: $94.85                      │ 21:31 BTC YES 75% BET $4.50    │
│  OPEN BETS: 2/2  |  W:3 L:1  |  WR: 75.0%  │ 21:31 SOL YES 67% BET $3.35    │
│  SESSION P&L: +$4.20                         │ 21:26 BNB NO  31% BET $3.45    │
│                                              │ 21:16 XRP YES 90% → settled +$ │
│  ASSET  SIDE  PROB  STATE         STRIKE     │ 21:16 window :15 opened        │
│  ─────────────────────────────────────────   │ 21:11 window :00 expired       │
│  BTC    YES   75%   BET_PLACED    $70,722    │                                 │
│  ETH    --    63%   SKIP                     │                                 │
│  SOL    YES   67%   BET_PLACED    $91,044    │                                 │
│  XRP    --    58%   SKIP                     │                                 │
│  BNB    --    52%   SKIP                     │                                 │
│                                              │                                 │
│  Next eval: ~3 min  |  Window: :15-:30       │                                 │
└──────────────────────────────────────────────┴─────────────────────────────────┘
k15> _
```

**Left panel — Status + Predictions:**
- Kalshi balance (queried from Kalshi API — source of truth)
- Open bets count / max, session W/L/WR
- Session P&L (from settled bets this session)
- Predictions table: asset, side (YES/NO/--), probability, lifecycle state, strike price
- Next eval countdown, current window time range

**Right panel — Activity Log:**
- Rolling list of last ~20 session events, most recent at top
- Events: bets placed (side, prob, amount), bets settled (win/loss, P&L), window transitions, skipped signals
- Session-only (in-memory, clears on restart)
- Not persisted — Kalshi is source of truth for actual balance and positions

**Bottom — Command prompt:**
- `k15>` prompt accepts `/` commands
- Dashboard refreshes above the prompt without interrupting input

### 2. Colors

| Element | Color |
|---------|-------|
| Wins, probability > 70%, YES side | Green |
| Losses, probability < 30%, NO side | Red |
| SETUP/waiting, SKIP, warnings | Yellow |
| Headers, borders, labels | Cyan |
| Neutral data, prices | White |
| BET_PLACED state | Magenta |

### 3. Interactive Commands

Entered at the `k15>` prompt. Typing `/` shows the command list.

| Command | Action |
|---------|--------|
| `/` | Show all available commands |
| `/dry-run` | Start V3 predictor in dry-run mode |
| `/live` | Start V3 predictor in live mode (requires "YES" confirmation) |
| `/stop` | Gracefully stop the running bot |
| `/status` | Show current state (running/stopped, predictor, bets) |
| `/balance` | Query and display Kalshi balance from API |
| `/closeall` | Emergency close all positions at market price (requires "YES" confirmation) |
| `/quit` | Stop bot if running and exit the program |

**Safety gates:**
- `/live` prompts: "Start LIVE trading? Type YES to confirm:"
- `/closeall` prompts: "EMERGENCY CLOSE ALL positions? Type YES to confirm:"
- Both require exact "YES" string, anything else cancels

### 4. Architecture

**Single file:** `k15.py` at project root.

**Threading model:**
- **Main thread:** Runs the `rich.live.Live` display loop + reads input from prompt
- **Daemon thread:** Runs the LiveDaemon eval loop (start/stop controlled by main thread)
- **Communication:** Simple shared state — `daemon_running` flag, `activity_log` list, `predictions` list

**`rich` components used:**
- `rich.live.Live` — flicker-free live display
- `rich.layout.Layout` — two-column split
- `rich.table.Table` — predictions table
- `rich.panel.Panel` — bordered panels
- `rich.text.Text` — colored text
- `rich.prompt.Confirm` — for /live and /closeall confirmation

**Refresh rate:** Every 2 seconds. The Live display rebuilds the layout from current state on each refresh.

**Existing `dashboard.py` changes:**
- Add `--simple` flag check: if `--simple` is passed OR `sys.stdout.isatty()` is False, use current plain text output
- Otherwise, current behavior unchanged — `k15.py` is the new interactive entry point
- MCP `algotrade_start` continues to use `dashboard.py` for background operation

### 5. Daemon Integration

`k15.py` imports and wraps `LiveDaemon` directly (same as `dashboard.py` does):

```python
from cli.live_daemon import LiveDaemon

daemon = LiveDaemon(dry_run=True, kalshi_only=True, predictor_version="v3")
```

On `/dry-run`:
- Create LiveDaemon with `dry_run=True, kalshi_only=True, predictor_version="v3"`
- Start daemon thread: calls `daemon.startup()`, then runs eval loop
- Dashboard reads `daemon.kalshi_predictions` for display

On `/live`:
- Same but `dry_run=False`
- Safety confirmation required

On `/stop`:
- Set `daemon._running = False`
- Daemon thread completes current cycle and exits
- Dashboard shows "STATUS: STOPPED"

On `/closeall`:
- Calls `daemon.kalshi_client.cancel_all_orders()` if available, or iterates open positions
- Safety confirmation required

### 6. Data Flow

```
LiveDaemon._kalshi_eval()
    → daemon.kalshi_predictions (list of dicts)
    → k15.py reads on each refresh
    → builds rich Table from predictions
    → renders in left panel

LiveDaemon._kalshi_execute_bet() (or dry-run equivalent)
    → k15.py intercepts via callback or polling
    → appends to activity_log (in-memory list)
    → renders in right panel

Kalshi API
    → daemon.kalshi_client.get_balance()
    → k15.py queries on each refresh (or every 30s to reduce API calls)
    → renders balance in header
```

## Files to Create

| File | Purpose |
|------|---------|
| `k15.py` | Interactive rich dashboard with command prompt |

## Files to Modify

| File | Change |
|------|--------|
| `dashboard.py` | Add `--simple` flag / TTY detection for plain text fallback |

## Files NOT Modified

- `cli/live_daemon.py` — k15.py wraps it, doesn't change it
- `strategy/strategies/` — no predictor changes
- `exchange/kalshi.py` — API client unchanged
- `mcp_server.py` — still uses dashboard.py for background operation

## Dependencies

- `rich` — install via `pip install rich` (add to requirements if not present)

## Success Criteria

- Dashboard renders without flicker at 2-second refresh
- Commands work while dashboard is live (no blocking)
- `/dry-run` starts the V3 predictor and predictions appear in the table
- `/stop` gracefully stops the daemon
- `/live` requires confirmation before starting
- `/closeall` requires confirmation before executing
- Activity log shows bets placed, settled, window transitions
- Balance comes from Kalshi API (source of truth)
- Falls back to plain text when piped or `--simple` is used
