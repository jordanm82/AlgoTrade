# Kalshi Multi-Timeframe Evaluation System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single 15-minute Kalshi scoring cycle with a progressive 5-minute evaluation system that accumulates confirmation within each 15-minute contract window, enabling more high-confidence bets.

**Architecture:** A wall-clock-aligned 5-minute evaluation cycle (`_kalshi_eval()`) replaces the old `_kalshi_cycle()`. Signals progress through a lifecycle (SETUP → CONFIRMED → DOUBLE_CONFIRMED → LAST_LOOK → EXPIRED) within each 15-minute window, requiring at least one closed 5m candle as confirmation before betting. Bet execution logic is extracted into `_kalshi_execute_bet()`.

**Tech Stack:** Python 3, pandas, pandas_ta, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-kalshi-multi-timeframe-eval-design.md`

---

### Task 1: Add `check_1m_momentum()` to predictor

**Files:**
- Modify: `strategy/strategies/kalshi_predictor.py`
- Test: `tests/test_kalshi.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_kalshi.py`:

```python
class TestKalshi1mMomentum:
    """Tests for 1-minute momentum check."""

    def test_up_momentum_confirmed(self):
        """2 of 3 green candles confirms UP."""
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0, 101.5, 103.0],
            "close": [101.0, 102.0, 101.0, 103.0, 104.0],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is True

    def test_down_momentum_confirmed(self):
        """2 of 3 red candles confirms DOWN."""
        df = pd.DataFrame({
            "open":  [104.0, 103.0, 102.0, 103.0, 101.0],
            "close": [103.0, 102.0, 103.0, 101.0, 100.0],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "DOWN", lookback=3) is True

    def test_momentum_not_confirmed(self):
        """Only 1 of 3 candles in direction fails."""
        df = pd.DataFrame({
            "open":  [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [101.0, 100.5, 101.5, 102.0, 103.5],
        }, index=pd.date_range("2026-01-01", periods=5, freq="1min"))
        predictor = KalshiPredictor()
        # Last 3: open=[102,103,104] close=[101.5,102,103.5] → 1 down, 1 down, 1 down
        # Actually: 102→101.5 DOWN, 103→102 DOWN, 104→103.5 DOWN → all 3 are DOWN
        # So for UP this should fail
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is False

    def test_insufficient_data_returns_false(self):
        """Less than lookback candles returns False."""
        df = pd.DataFrame({
            "open": [100.0],
            "close": [101.0],
        }, index=pd.date_range("2026-01-01", periods=1, freq="1min"))
        predictor = KalshiPredictor()
        assert predictor.check_1m_momentum(df, "UP", lookback=3) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi.py::TestKalshi1mMomentum -v`
Expected: FAIL — `check_1m_momentum` not defined

- [ ] **Step 3: Implement `check_1m_momentum()`**

Add to `KalshiPredictor` class in `strategy/strategies/kalshi_predictor.py`:

```python
def check_1m_momentum(self, df_1m: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
    """Check if recent 1m candles confirm the direction.

    Returns True if at least 2 of the last `lookback` candles moved in
    the expected direction (close > open for UP, close < open for DOWN).
    Simple majority rule avoids single noisy 1m candle killing the signal.
    """
    if df_1m is None or len(df_1m) < lookback:
        return False

    recent = df_1m.iloc[-lookback:]
    if direction == "UP":
        confirming = (recent["close"] > recent["open"]).sum()
    else:
        confirming = (recent["close"] < recent["open"]).sum()

    return confirming >= 2
```

- [ ] **Step 4: Run all kalshi tests**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add strategy/strategies/kalshi_predictor.py tests/test_kalshi.py && git commit -m "feat: add check_1m_momentum() for Kalshi last-look entries

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Extract `_kalshi_execute_bet()` from existing execution logic

**Files:**
- Modify: `cli/live_daemon.py`

- [ ] **Step 1: Extract the bet execution block into a standalone method**

The execution logic lives in `_kalshi_cycle()` at lines 670-843. Extract it into a new method:

```python
def _kalshi_execute_bet(self, symbol: str, series_ticker: str, signal, market_data: dict | None) -> dict:
    """Execute a single Kalshi bet. Returns a pred dict with outcome details.

    Handles: balance check, market discovery, orderbook pricing,
    order placement, fill verification, active bet tracking.
    """
    asset = symbol.split("/")[0]
    ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
    net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

    pred = {
        "symbol": symbol, "asset": asset,
        "direction": signal.direction, "confidence": signal.confidence,
        "ob": ob_imb, "flow": net_flow, "reason": "",
    }

    MAX_ENTRY_CENTS = 50
    RISK_PER_BET_PCT = 0.05

    side = "yes" if signal.direction == "UP" else "no"
    pred["reason"] = f"would bet {side.upper()} (conf={signal.confidence})"

    if self.dry_run:
        print(colored(
            f"  [KALSHI DRY] {asset} {signal.direction} "
            f"conf={signal.confidence} | "
            f"OB={ob_imb:+.2f} flow={net_flow:+.2f} | "
            f"bet {side.upper()} (hold to settlement)",
            "magenta",
        ))
        return pred

    # === Live execution (lines 690-843 of current _kalshi_cycle) ===
    # Move the entire live execution block here verbatim:
    # - _init_kalshi_client()
    # - get_balance()
    # - find markets via events API
    # - query orderbook for pricing
    # - calculate fill_price with spread logic
    # - enforce MAX_ENTRY_CENTS cap
    # - calculate count from risk budget
    # - place_order()
    # - verify fill
    # - track in _active_kalshi_bets
    # (Copy the full block from lines 690-843 without modification)

    return pred
```

The key change: copy the ENTIRE existing execution block (lines 690-843) into this method with these specific transformations:

1. **Line 694** (`continue` after client init failed) → `return pred`
2. **Line 714** (`continue` after no markets found) → `return pred`
3. **Line 780** (`continue` after price > MAX_ENTRY_CENTS) → `return pred`
4. **Line 796** (`continue` after insufficient balance) → `return pred`

Each `continue` was skipping to the next asset in the old loop. Now it returns the pred dict with the reason already set. No other logic changes needed.

- [ ] **Step 2: Update `_kalshi_cycle()` to call the new method**

Replace the inline execution block in `_kalshi_cycle()` with:

```python
# Execute top signals up to concurrency limit
for vs in valid_signals:
    if len(self._active_kalshi_bets) >= MAX_CONCURRENT_KALSHI_BETS:
        vs["pred"]["reason"] = "at max concurrent bets (lower confidence skipped)"
        predictions.append(vs["pred"])
        continue

    pred = self._kalshi_execute_bet(
        vs["symbol"], vs["series_ticker"], vs["signal"], vs["market_data"]
    )
    predictions.append(pred)
```

- [ ] **Step 3: Run daemon tests to verify no regressions**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_kalshi.py -v`
Expected: ALL PASS (test_daemon.py tests a different Daemon class, not LiveDaemon)

- [ ] **Step 4: Quick manual validation**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -c "from cli.live_daemon import LiveDaemon; d = LiveDaemon(dry_run=True, kalshi_only=True); print('LiveDaemon imported OK')"`
Expected: Prints "LiveDaemon imported OK" without errors

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py && git commit -m "refactor: extract _kalshi_execute_bet() from inline execution logic

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add `_kalshi_eval()` lifecycle state machine

**Files:**
- Modify: `cli/live_daemon.py`

- [ ] **Step 1: Add new constants and state**

At the top of `cli/live_daemon.py` (near line 53), add:

```python
KALSHI_CUTOFF_MINUTES = 13     # no new bets after this minute in the 15m window
KALSHI_LASTLOOK_MINUTE = 12    # elevated threshold window
KALSHI_THRESHOLD_BOOST = 10    # added to per-asset threshold for last-look
```

In `__init__()` (after `self._active_kalshi_bets`), add:

```python
self._kalshi_pending_signals = {}   # {asset: {direction, base_conf, last_5m_conf, ...}}
self._last_kalshi_eval = 0          # timestamp of last eval
self._kalshi_5m_dataframes = {}     # {symbol: DataFrame} — cached 5m data with indicators
```

- [ ] **Step 2: Implement `_kalshi_eval()` method**

Add to `LiveDaemon`:

```python
def _kalshi_eval(self):
    """5-minute Kalshi evaluation cycle with progressive confirmation.

    Lifecycle within each 15m window:
    - SETUP (min 0-4): Score direction, no betting
    - CONFIRMED (min 5-9): First 5m candle confirms, eligible to bet
    - DOUBLE_CONFIRMED (min 10-11): Second 5m candle, highest conviction
    - LAST_LOOK (min 12): 1m momentum check, elevated threshold
    - EXPIRED (min 13+): No new bets
    """
    from data.market_data import get_order_book_imbalance, get_trade_flow

    now_utc = datetime.now(timezone.utc)
    minute_in_window = now_utc.minute % 15

    # Compute current window start (round down to :00/:15/:30/:45)
    window_minute = now_utc.minute - minute_in_window
    current_window_start = now_utc.replace(minute=window_minute, second=0, microsecond=0)

    # Prune expired bets
    now_ts = time.time()
    self._active_kalshi_bets = {
        t: placed for t, placed in self._active_kalshi_bets.items()
        if now_ts - placed < 900
    }

    if minute_in_window >= KALSHI_CUTOFF_MINUTES:
        # Too close to settlement — clear predictions to show EXPIRED state
        self.kalshi_predictions = [{
            "symbol": sym, "asset": sym.split("/")[0],
            "direction": "--", "confidence": 0,
            "reason": "expired — too close to settlement",
            "ob": 0, "flow": 0, "state": "EXPIRED",
        } for sym in self.KALSHI_PAIRS]
        return

    predictions = []
    actionable_signals = []

    for symbol, series_ticker in self.KALSHI_PAIRS.items():
        asset = symbol.split("/")[0]

        # Check if pending signal belongs to current window
        pending = self._kalshi_pending_signals.get(asset)
        if pending and pending.get("window_start") != current_window_start:
            # New window — clear old signal
            self._kalshi_pending_signals.pop(asset, None)
            self._kalshi_5m_dataframes.pop(symbol, None)
            pending = None

        # Skip if already bet this window
        if pending and pending.get("bet_placed"):
            predictions.append({
                "symbol": symbol, "asset": asset,
                "direction": pending["direction"],
                "confidence": pending.get("last_5m_conf", pending["base_conf"]),
                "reason": "already bet this window",
                "ob": 0, "flow": 0, "state": "BET_PLACED",
            })
            continue

        # Determine lifecycle state
        if minute_in_window <= 4:
            state = "SETUP"
        elif minute_in_window <= 9:
            state = "CONFIRMED"
        elif minute_in_window <= 11:
            state = "DOUBLE_CONFIRMED"
        elif minute_in_window == 12:
            state = "LAST_LOOK"
        else:
            state = "EXPIRED"

        # --- LAST_LOOK: use cached 5m scores + 1m momentum ---
        if state == "LAST_LOOK":
            if not pending or not pending.get("confirmed"):
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending["direction"] if pending else "--",
                    "confidence": pending.get("last_5m_conf", 0) if pending else 0,
                    "reason": "last-look: no prior confirmation",
                    "ob": 0, "flow": 0, "state": state,
                })
                continue

            last_conf = pending.get("last_5m_conf", 0)
            asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, self.kalshi_threshold)
            elevated = asset_threshold + KALSHI_THRESHOLD_BOOST

            if last_conf < elevated:
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending["direction"],
                    "confidence": last_conf,
                    "reason": f"last-look: conf {last_conf} < elevated threshold {elevated}",
                    "ob": 0, "flow": 0, "state": state,
                })
                continue

            # Fetch 1m candles for momentum check
            try:
                df_1m = self.fetcher.ohlcv(symbol, "1m", limit=5)
            except Exception:
                df_1m = None

            if df_1m is not None and self.kalshi_predictor.check_1m_momentum(df_1m, pending["direction"]):
                # Momentum confirmed — actionable
                from strategy.strategies.kalshi_predictor import KalshiSignal
                fake_signal = KalshiSignal(
                    asset=asset, direction=pending["direction"],
                    confidence=last_conf, components={}, price=0, rsi=0,
                )
                actionable_signals.append({
                    "symbol": symbol, "series_ticker": series_ticker,
                    "signal": fake_signal, "market_data": None,
                    "state": state,
                })
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending["direction"],
                    "confidence": last_conf,
                    "reason": f"last-look: 1m momentum confirmed, taking bet",
                    "ob": 0, "flow": 0, "state": state,
                })
            else:
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending["direction"],
                    "confidence": last_conf,
                    "reason": "last-look: 1m momentum NOT confirmed",
                    "ob": 0, "flow": 0, "state": state,
                })
            continue

        # --- SETUP / CONFIRMED / DOUBLE_CONFIRMED: fetch 5m data + score ---

        # Fetch 5m candles
        try:
            df_5m = self.fetcher.ohlcv(symbol, "5m", limit=200)
            if df_5m is not None and not df_5m.empty:
                # Drop partial candle: if the last candle started less than 5 min ago, it's still forming
                last_candle_age = (now_utc - df_5m.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()
                if last_candle_age < 300:  # less than 5 minutes old = partial
                    df_5m = df_5m.iloc[:-1]
                df_5m = add_indicators(df_5m)
                self._kalshi_5m_dataframes[symbol] = df_5m
        except Exception:
            df_5m = self._kalshi_5m_dataframes.get(symbol)

        if df_5m is None or len(df_5m) < 50:
            predictions.append({
                "symbol": symbol, "asset": asset,
                "direction": "--", "confidence": 0,
                "reason": "no 5m data", "ob": 0, "flow": 0, "state": state,
            })
            continue

        # Fetch 1h data for MTF
        df_1h = None
        try:
            df_1h = self.fetcher.ohlcv(symbol, "1h", limit=50)
            if df_1h is not None and not df_1h.empty:
                df_1h = add_indicators(df_1h)
        except Exception:
            pass

        # Fetch leading indicators
        market_data = None
        try:
            ob = get_order_book_imbalance(symbol)
            tf = get_trade_flow(symbol, limit=100)
            market_data = {"order_book": ob, "trade_flow": tf}
        except Exception:
            pass

        # Score
        signal = self.kalshi_predictor.score(df_5m, market_data=market_data, df_1h=df_1h)

        ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
        net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

        if state == "SETUP":
            # Store direction + confidence, no betting
            if signal is not None and asset not in self._kalshi_pending_signals:
                self._kalshi_pending_signals[asset] = {
                    "direction": signal.direction,
                    "base_conf": signal.confidence,
                    "last_5m_conf": signal.confidence,
                    "setup_time": now_utc,
                    "confirmed": False,
                    "bet_placed": False,
                    "window_start": current_window_start,
                }
            predictions.append({
                "symbol": symbol, "asset": asset,
                "direction": signal.direction if signal else "--",
                "confidence": signal.confidence if signal else 0,
                "reason": "setup — waiting for confirmation",
                "ob": ob_imb, "flow": net_flow, "state": state,
            })

        elif state in ("CONFIRMED", "DOUBLE_CONFIRMED"):
            if signal is None:
                # None = skip, signal survives
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending["direction"] if pending else "--",
                    "confidence": 0,
                    "reason": f"{state.lower()}: score returned None, signal survives",
                    "ob": ob_imb, "flow": net_flow, "state": state,
                })
                continue

            # If no pending signal exists (SETUP returned None), treat this as a late SETUP
            if not pending:
                self._kalshi_pending_signals[asset] = {
                    "direction": signal.direction,
                    "base_conf": signal.confidence,
                    "last_5m_conf": signal.confidence,
                    "setup_time": now_utc,
                    "confirmed": False,
                    "bet_placed": False,
                    "window_start": current_window_start,
                }
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "reason": f"{state.lower()}: late setup (no prior SETUP), waiting for next eval",
                    "ob": ob_imb, "flow": net_flow, "state": "LATE_SETUP",
                })
                continue

            if signal.direction != pending["direction"]:
                # Direction flipped — kill signal
                self._kalshi_pending_signals.pop(asset, None)
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "reason": f"{state.lower()}: direction flipped ({pending['direction']}→{signal.direction}), signal killed",
                    "ob": ob_imb, "flow": net_flow, "state": state,
                })
                continue

            # Update last_5m_conf
            if pending:
                pending["last_5m_conf"] = signal.confidence
                pending["confirmed"] = True

            asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, self.kalshi_threshold)
            if signal.confidence >= asset_threshold:
                # Actionable!
                actionable_signals.append({
                    "symbol": symbol, "series_ticker": series_ticker,
                    "signal": signal, "market_data": market_data,
                    "state": state,
                })
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "reason": f"{state.lower()}: conf {signal.confidence} >= threshold {asset_threshold}",
                    "ob": ob_imb, "flow": net_flow, "state": state,
                })
            else:
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "reason": f"{state.lower()}: conf {signal.confidence} < threshold {asset_threshold}",
                    "ob": ob_imb, "flow": net_flow, "state": state,
                })

    # Execute top actionable signals by confidence
    actionable_signals.sort(key=lambda x: x["signal"].confidence, reverse=True)
    for vs in actionable_signals:
        if len(self._active_kalshi_bets) >= MAX_CONCURRENT_KALSHI_BETS:
            break
        pred = self._kalshi_execute_bet(
            vs["symbol"], vs["series_ticker"], vs["signal"], vs["market_data"]
        )
        # Mark bet as placed in pending signals
        asset = vs["symbol"].split("/")[0]
        if asset in self._kalshi_pending_signals:
            self._kalshi_pending_signals[asset]["bet_placed"] = True

    self.kalshi_predictions = predictions
```

- [ ] **Step 3: Run daemon tests**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_daemon.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py && git commit -m "feat: add _kalshi_eval() lifecycle state machine with progressive confirmation

SETUP (min 0-4) → CONFIRMED (min 5-9) → DOUBLE_CONFIRMED (min 10-11) → LAST_LOOK (min 12)
Requires at least one closed 5m candle before betting.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Wire `_kalshi_eval()` into the run loop and remove old cycle

**Files:**
- Modify: `cli/live_daemon.py`

- [ ] **Step 1: Update `run()` to add wall-clock aligned Kalshi eval**

In the `run()` method, after line 1503 (`last_signal_time = time.time()`), add:

```python
self._last_kalshi_eval = 0
```

In the main `while` loop, after the signal cycle block, add the Kalshi eval trigger:

```python
# Wall-clock aligned Kalshi eval at :01, :06, :11 + LAST_LOOK at :12/:27/:42/:57
current_minute = datetime.now(timezone.utc).minute
should_eval = (current_minute % 5 == 1 and now - self._last_kalshi_eval >= 240) \
           or (current_minute % 15 == 12 and now - self._last_kalshi_eval >= 50)
if should_eval:
    try:
        self._kalshi_eval()
    except Exception as e:
        print(colored(f"  [KALSHI EVAL ERR] {e}", "red"))
    self._last_kalshi_eval = now
```

- [ ] **Step 2: Remove `_kalshi_cycle()` call from `signal_cycle()`**

In `signal_cycle()`, remove the block:
```python
# Kalshi prediction cycle
try:
    self._kalshi_cycle()
except Exception as e:
    print(colored(f"  [KALSHI ERR] cycle failed: {e}", "yellow"))
```

Delete the `_kalshi_cycle()` method entirely — it is replaced by `_kalshi_eval()` and `_kalshi_execute_bet()`. No code references it after this change.

- [ ] **Step 3: Run an initial `_kalshi_eval()` on startup**

In `run()`, after `self.signal_cycle()` (line 1501), add:

```python
# Run initial Kalshi eval immediately
try:
    self._kalshi_eval()
    self._last_kalshi_eval = time.time()
except Exception as e:
    print(colored(f"  [KALSHI EVAL ERR] startup eval: {e}", "red"))
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python -m pytest tests/test_daemon.py tests/test_kalshi.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py && git commit -m "feat: wire _kalshi_eval() into run loop, remove old _kalshi_cycle() from signal_cycle

Wall-clock aligned at minutes :01, :06, :11 with LAST_LOOK at :12.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Update dashboard to show lifecycle state

**Files:**
- Modify: `dashboard.py`

- [ ] **Step 1: Update Kalshi predictions display**

Find the KALSHI PREDICTIONS section in the dashboard (around line 361). Update to include lifecycle state:

The `kalshi_predictions` list now includes a `"state"` key. Update the display to show it:

```python
# In the predictions display loop, add state info:
state = pred.get("state", "")
state_tag = f" [{state}]" if state else ""
# Include state_tag in the formatted output line
```

- [ ] **Step 2: Test manually (dry run)**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python dashboard.py --dry-run --kalshi-only --cycles 1`
Verify: Dashboard shows lifecycle state tags (SETUP, CONFIRMED, etc.) in Kalshi predictions section.

- [ ] **Step 3: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add dashboard.py && git commit -m "feat: show Kalshi lifecycle state in dashboard predictions display

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Rewrite backtest for 5m evaluation lifecycle

**Files:**
- Modify: `backtest_kalshi.py`

- [ ] **Step 1: Add 5m data fetching**

Update `fetch_candles()` — it already works for any timeframe. Add a helper to fetch 5m data and the ASSETS dict already has the right pairs.

- [ ] **Step 2: Implement lifecycle simulation**

Add a new function `simulate_lifecycle()` that processes the 5m candle data in 15-minute windows:

```python
def simulate_lifecycle(
    asset_name: str,
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_1m: pd.DataFrame | None,
    predictor: KalshiPredictor,
) -> list[dict]:
    """Simulate the 5m evaluation lifecycle on historical data.

    For each 15-minute window:
    1. Minute 0: Score SETUP (5m data up to this point)
    2. Minute 5: Score CONFIRMED (includes first closed 5m candle)
    3. Minute 10: Score DOUBLE_CONFIRMED (includes second closed 5m candle)
    4. Minute 12: LAST_LOOK (1m momentum check if available)
    5. Record the FIRST qualifying entry per window

    Returns list of bet results with actual direction for P&L calculation.
    """
```

The key logic:
- Add indicators to 5m data once
- Group 5m candles into 15-minute windows based on their timestamp
- Within each window, there are exactly 3 five-minute candles (at :00, :05, :10)
- SETUP scores using data up to the :00 candle (exclusive — it just opened)
- CONFIRMED scores using data through the :05 candle (it just closed)
- DOUBLE_CONFIRMED scores using data through the :10 candle
- The "actual direction" = did the 15m window close higher or lower than it opened? Compare close of the :10 candle (last data point) with close of the previous window's :10 candle. OR more precisely: compare the price at the window open (:00 open) with the price at window close (:14:59 close, which is the :10 candle's close since 5m candles end at :15).
- For LAST_LOOK: find 1m candles within minutes 9-12 of the window, check momentum

- [ ] **Step 3: Update `main()` to run both 15m baseline and 5m lifecycle**

The backtest should run:
1. **Baseline (existing):** 15m candle-close scoring (for comparison)
2. **5m lifecycle:** The new progressive evaluation

Report both side-by-side with threshold sweep on the 5m version.

- [ ] **Step 4: Run backtest on 30-day data**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python backtest_kalshi.py --days 30`

Verify output shows both baseline and 5m lifecycle results.

- [ ] **Step 5: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add backtest_kalshi.py && git commit -m "feat: rewrite backtest with 5m evaluation lifecycle simulation

Side-by-side comparison: 15m baseline vs 5m progressive confirmation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Run backtest, validate, and tune thresholds

**Files:**
- None created — analysis work

- [ ] **Step 1: Run 30-day backtest**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python backtest_kalshi.py --days 30`

Record: 5m lifecycle WR vs 15m baseline WR, bet count, per-asset breakdown.

- [ ] **Step 2: Run 90-day validation**

Run: `cd /Users/jordan/Documents/Dev/algotrade && ./venv/bin/python backtest_kalshi.py --days 90`

Success criteria: WR >= 60%, more bets than 15m baseline, PF >= 3.0.

- [ ] **Step 3: Adjust thresholds if needed**

If 5m scoring produces different confidence distributions, re-tune per-asset thresholds.

- [ ] **Step 4: Commit any threshold adjustments**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add cli/live_daemon.py && git commit -m "tune: adjust Kalshi thresholds for 5m evaluation system

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update Strategy 3 section**

Update the Kalshi section in CLAUDE.md:
- Change "Confidence scoring" to describe the 5m evaluation lifecycle
- Add lifecycle state table (SETUP → CONFIRMED → DOUBLE_CONFIRMED → LAST_LOOK → EXPIRED)
- Update evaluation frequency from "every 15m" to "every 5m (wall-clock aligned)"
- Add LAST_LOOK elevated threshold info
- Note the confirmation requirement (no betting before first 5m candle closes)

- [ ] **Step 2: Commit**

```bash
cd /Users/jordan/Documents/Dev/algotrade && git add CLAUDE.md && git commit -m "docs: update CLAUDE.md with 5m multi-timeframe evaluation lifecycle

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
