# Chop Detection Logging (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log BBW + ATR chop metrics (per-asset and market-wide, 15m and 1h) at every bet placement and carry them through to SETTLED / M10 exit events, so offline analysis can correlate chop intensity with outcomes.

**Architecture:** A new `_compute_chop_metrics(asset)` helper on `KalshiDaemon` returns an 8-field dict from cached dataframes (no new fetches). The dict is emitted on `MARKET_SELECT`, stashed on the `_pending_bets` entry, and re-emitted on `SETTLED`, `M10_EXIT`, and `M10_RESTING_SELL`. No gating, no model change.

**Tech Stack:** Python 3, pandas, pytest. All work is in `cli/kalshi_daemon.py` plus one new test file.

**Spec:** `docs/superpowers/specs/2026-04-13-chop-detection-logging-design.md`

---

## File Structure

- **Modify:** `cli/kalshi_daemon.py`
  - Add `_compute_chop_metrics` + two small static helpers (`_bbw_from_df`, `_atr_pct_from_df`) near line 280 (existing snapshot-feature helpers).
  - Call helper in `_kalshi_execute_bet` (~line 2605, before `MARKET_SELECT` log).
  - Stash dict on `_pending_bets` entry (~line 2838).
  - Emit dict in `SETTLED` log (~line 1606).
  - Emit dict in `M10_EXIT` log (~line 1080) and `M10_RESTING_SELL` log (~line 1028).
  - Add `chop_metrics: {all-None dict}` to both recovery paths (`_recover_positions` bet entry ~line 3409, resting-order entry ~line 3509).

- **Create:** `tests/test_chop_metrics.py` — unit tests for the helper.

---

### Task 1: Test scaffolding and full-data test

**Files:**
- Create: `tests/test_chop_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chop_metrics.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from cli.kalshi_daemon import KalshiDaemon


def _make_df(n: int = 30, atr: float = 100.0, bb_upper: float = 105.0,
             bb_lower: float = 95.0, sma_20: float = 100.0,
             atr_variable: bool = False) -> pd.DataFrame:
    """Build a synthetic indicator-ready dataframe with n rows."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    atr_col = (
        np.linspace(atr * 0.5, atr * 1.5, n) if atr_variable
        else np.full(n, atr)
    )
    return pd.DataFrame(
        {
            "atr": atr_col,
            "bb_upper": np.full(n, bb_upper),
            "bb_lower": np.full(n, bb_lower),
            "sma_20": np.full(n, sma_20),
            "close": np.full(n, sma_20),
        },
        index=idx,
    )


def _make_daemon(cache: dict) -> KalshiDaemon:
    """Bare daemon with only the cache populated. Avoids touching I/O."""
    d = KalshiDaemon.__new__(KalshiDaemon)
    d._kalshi_cached_dataframes = cache
    return d


def test_compute_chop_metrics_full_data_all_assets():
    """All four assets have both 15m and 1h cached — all 8 fields non-None."""
    cache = {}
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
        cache[pair] = _make_df(n=30, atr_variable=True)
        cache[f"{pair}_1h"] = _make_df(n=30, atr_variable=True)

    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert set(out.keys()) == {
        "bbw_15m", "atr_pct_15m", "bbw_1h", "atr_pct_1h",
        "bbw_15m_mkt", "atr_pct_15m_mkt", "bbw_1h_mkt", "atr_pct_1h_mkt",
    }
    # Per-asset BBW: (105 - 95) / 100 * 100 = 10.0
    assert out["bbw_15m"] == pytest.approx(10.0)
    assert out["bbw_1h"] == pytest.approx(10.0)
    # ATR percentile: with variable ATR, final value is near the max => ≈1.0
    assert 0.0 <= out["atr_pct_15m"] <= 1.0
    assert 0.0 <= out["atr_pct_1h"] <= 1.0
    # Market-wide matches per-asset since all four are identical
    assert out["bbw_15m_mkt"] == pytest.approx(10.0)
    assert out["bbw_1h_mkt"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run test to verify it fails**

```
./venv/bin/pytest tests/test_chop_metrics.py::test_compute_chop_metrics_full_data_all_assets -v
```

Expected: FAIL with `AttributeError: 'KalshiDaemon' object has no attribute '_compute_chop_metrics'`.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_chop_metrics.py
git commit -m "test: chop metrics full-data scaffold (failing)"
```

---

### Task 2: Implement `_compute_chop_metrics` helper

**Files:**
- Modify: `cli/kalshi_daemon.py` — add the helper and two small static sub-helpers. Insert right after the `_build_kalshi_extras` region around line 280 (the existing snapshot-feature helpers). Use a new grep target to place: add below the `_refresh_higher_timeframes` method (~line 275) before the next method begins.

- [ ] **Step 1: Add the helper**

Insert this block just before `def _build_kalshi_extras` (or wherever the next method begins after `_refresh_higher_timeframes`). If you cannot locate that anchor, add the block immediately after `_refresh_higher_timeframes`'s closing `return` / method body — the point is it lives in the `KalshiDaemon` class body with the other snapshot helpers, not inside any existing method.

```python
    # ------------------------------------------------------------------
    # Chop detection metrics (v1: logging only, no gating)
    # ------------------------------------------------------------------

    @staticmethod
    def _bbw_from_df(df) -> float | None:
        """Bollinger Band Width as percent of sma_20 on the last completed row.
        Returns None if df is missing, too short, or bands are degenerate.
        """
        if df is None or len(df) < 20:
            return None
        cols = df.columns
        if not ("bb_upper" in cols and "bb_lower" in cols and "sma_20" in cols):
            return None
        row = df.iloc[-1]
        try:
            upper = float(row["bb_upper"])
            lower = float(row["bb_lower"])
            mid = float(row["sma_20"])
        except (TypeError, ValueError):
            return None
        if not (mid > 0) or not np.isfinite(upper) or not np.isfinite(lower) or not np.isfinite(mid):
            return None
        return (upper - lower) / mid * 100.0

    @staticmethod
    def _atr_pct_from_df(df) -> float | None:
        """ATR percentile — rolling-20 min/max normalization of the last ATR value.
        Mirrors the existing atr_percentile formula in _build_kalshi_extras.
        Returns None if df is missing, too short, or range is degenerate.
        """
        if df is None or len(df) < 20 or "atr" not in df.columns:
            return None
        atr_s = df["atr"].dropna()
        if len(atr_s) < 20:
            return None
        r20 = atr_s.rolling(20)
        try:
            atr = float(atr_s.iloc[-1])
            mn = float(r20.min().iloc[-1])
            mx = float(r20.max().iloc[-1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(atr) or not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return None
        return (atr - mn) / (mx - mn)

    def _compute_chop_metrics(self, asset: str) -> dict:
        """Compute BBW + ATR chop metrics for `asset` (per-asset) plus market-wide
        averages over all four KALSHI_PAIRS assets. Returns a dict with 8 keys;
        missing fields are None (no stale substitution — see CLAUDE.md fallback rule).
        Never raises.
        """
        # 15m cache retains the in-progress window's opening candle; drop it so
        # the last row matches what the model sees at decision time.
        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15
        window_start = now_utc.replace(
            minute=now_utc.minute - minute_in_window,
            second=0, microsecond=0, tzinfo=None,
        )
        window_start_pd = pd.Timestamp(window_start)

        def _per_asset(symbol: str):
            """Returns (bbw_15m, atr_pct_15m, bbw_1h, atr_pct_1h), each possibly None."""
            try:
                df15 = self._kalshi_cached_dataframes.get(symbol)
                if df15 is not None:
                    df15 = df15[df15.index < window_start_pd]
                df1h = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
                return (
                    self._bbw_from_df(df15),
                    self._atr_pct_from_df(df15),
                    self._bbw_from_df(df1h),
                    self._atr_pct_from_df(df1h),
                )
            except Exception as e:
                print(colored(f"  [CHOP] {symbol} metric error: {e}", "yellow"))
                return (None, None, None, None)

        my_symbol = f"{asset}/USDT"
        my15_bbw, my15_atr, my1h_bbw, my1h_atr = _per_asset(my_symbol)

        all_15_bbw, all_15_atr, all_1h_bbw, all_1h_atr = [], [], [], []
        for pair in self.KALSHI_PAIRS:
            b15, a15, b1h, a1h = _per_asset(pair)
            if b15 is not None: all_15_bbw.append(b15)
            if a15 is not None: all_15_atr.append(a15)
            if b1h is not None: all_1h_bbw.append(b1h)
            if a1h is not None: all_1h_atr.append(a1h)

        def _mean(lst):
            # Need at least 2 assets to call it "market-wide"
            return sum(lst) / len(lst) if len(lst) >= 2 else None

        return {
            "bbw_15m": my15_bbw,
            "atr_pct_15m": my15_atr,
            "bbw_1h": my1h_bbw,
            "atr_pct_1h": my1h_atr,
            "bbw_15m_mkt": _mean(all_15_bbw),
            "atr_pct_15m_mkt": _mean(all_15_atr),
            "bbw_1h_mkt": _mean(all_1h_bbw),
            "atr_pct_1h_mkt": _mean(all_1h_atr),
        }
```

- [ ] **Step 2: Run the full-data test to verify it passes**

```
./venv/bin/pytest tests/test_chop_metrics.py::test_compute_chop_metrics_full_data_all_assets -v
```

Expected: PASS.

- [ ] **Step 3: Commit the helper**

```bash
git add cli/kalshi_daemon.py
git commit -m "feat: _compute_chop_metrics helper (BBW + ATR pct, per-asset + market)"
```

---

### Task 3: Edge-case tests

**Files:**
- Modify: `tests/test_chop_metrics.py` — append tests for the 5 edge cases enumerated in the spec's "Testing" section.

- [ ] **Step 1: Add the tests**

Append these to `tests/test_chop_metrics.py`:

```python
def test_compute_chop_metrics_missing_1h():
    """1h df absent for the bet's asset — 1h fields are None; 15m fields still computed."""
    cache = {"BTC/USDT": _make_df(n=30, atr_variable=True)}
    for pair in ("ETH/USDT", "SOL/USDT", "XRP/USDT"):
        cache[pair] = _make_df(n=30, atr_variable=True)
        cache[f"{pair}_1h"] = _make_df(n=30, atr_variable=True)

    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert out["bbw_15m"] is not None
    assert out["atr_pct_15m"] is not None
    assert out["bbw_1h"] is None
    assert out["atr_pct_1h"] is None
    # Market 1h still has 3 other assets — should still compute
    assert out["bbw_1h_mkt"] is not None


def test_compute_chop_metrics_short_15m():
    """15m df has fewer than 20 rows — 15m fields are None."""
    cache = {"BTC/USDT": _make_df(n=5)}
    cache["BTC/USDT_1h"] = _make_df(n=30, atr_variable=True)
    for pair in ("ETH/USDT", "SOL/USDT", "XRP/USDT"):
        cache[pair] = _make_df(n=30, atr_variable=True)
        cache[f"{pair}_1h"] = _make_df(n=30, atr_variable=True)

    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert out["bbw_15m"] is None
    assert out["atr_pct_15m"] is None
    assert out["bbw_1h"] is not None


def test_compute_chop_metrics_one_asset_missing_from_market():
    """Only one asset has data — market-wide fields are None (needs ≥2)."""
    cache = {
        "BTC/USDT": _make_df(n=30, atr_variable=True),
        "BTC/USDT_1h": _make_df(n=30, atr_variable=True),
    }
    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert out["bbw_15m"] is not None
    assert out["bbw_15m_mkt"] is None
    assert out["atr_pct_15m_mkt"] is None
    assert out["bbw_1h_mkt"] is None
    assert out["atr_pct_1h_mkt"] is None


def test_compute_chop_metrics_all_assets_missing():
    """Empty cache — every field is None and helper does not raise."""
    d = _make_daemon(cache={})
    out = d._compute_chop_metrics("BTC")

    assert set(out.keys()) == {
        "bbw_15m", "atr_pct_15m", "bbw_1h", "atr_pct_1h",
        "bbw_15m_mkt", "atr_pct_15m_mkt", "bbw_1h_mkt", "atr_pct_1h_mkt",
    }
    assert all(v is None for v in out.values())


def test_compute_chop_metrics_degenerate_bbw():
    """sma_20 == 0 — BBW is None (guards against divide-by-zero)."""
    df = _make_df(n=30, sma_20=0.0)
    cache = {"BTC/USDT": df, "BTC/USDT_1h": df}
    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert out["bbw_15m"] is None
    assert out["bbw_1h"] is None
```

- [ ] **Step 2: Run all chop-metric tests**

```
./venv/bin/pytest tests/test_chop_metrics.py -v
```

Expected: 5 passes, 0 fails.

- [ ] **Step 3: Commit**

```bash
git add tests/test_chop_metrics.py
git commit -m "test: chop metrics edge cases"
```

---

### Task 4: Wire chop_metrics into MARKET_SELECT log and `_pending_bets` stash

**Files:**
- Modify: `cli/kalshi_daemon.py:2605-2621` — `MARKET_SELECT` emission in `_kalshi_execute_bet`.
- Modify: `cli/kalshi_daemon.py:2838-2853` — `_pending_bets.append` in `_kalshi_execute_bet`.

- [ ] **Step 1: Compute the dict once, emit in MARKET_SELECT**

Replace the existing `MARKET_SELECT` emission:

```python
            # Log market selection for debugging demo/live issues
            market_strike = market.get("floor_strike") or market.get("custom_strike")
            market_close = market.get("close_time", "?")
            model_strike = signal.strike_price if isinstance(signal, KalshiV3Signal) else 0
            self._log_trade_debug(
                asset=asset, action="MARKET_SELECT",
                details={
                    "ticker": ticker,
                    "market_strike": market_strike,
                    "model_strike": model_strike,
                    "close_time": market_close,
                    "side": side,
                    "direction": direction_label,
                    "n_markets": len(all_markets),
                    "demo": self.demo,
                }
            )
```

with:

```python
            # Log market selection for debugging demo/live issues
            market_strike = market.get("floor_strike") or market.get("custom_strike")
            market_close = market.get("close_time", "?")
            model_strike = signal.strike_price if isinstance(signal, KalshiV3Signal) else 0
            chop_metrics = self._compute_chop_metrics(asset)
            self._log_trade_debug(
                asset=asset, action="MARKET_SELECT",
                details={
                    "ticker": ticker,
                    "market_strike": market_strike,
                    "model_strike": model_strike,
                    "close_time": market_close,
                    "side": side,
                    "direction": direction_label,
                    "n_markets": len(all_markets),
                    "demo": self.demo,
                    **chop_metrics,
                }
            )
```

- [ ] **Step 2: Stash on the `_pending_bets` entry**

Find the `_pending_bets.append({...})` block (around line 2838) and add one line so the entry carries the dict:

```python
                    self._pending_bets.append({
                        "asset": asset,
                        "symbol": symbol,
                        "side": side,
                        "direction": direction_label,
                        "strike": signal.strike_price,
                        "confidence": conf_display,
                        "bet_time": datetime.now(timezone.utc),
                        "settle_time": settle_time,
                        "fill_price": fill_price,
                        "count": int(actual_filled),
                        "contract_price": fill_price,
                        "order_id": order_id,
                        "ticker": ticker,
                        "live": True,
                        "chop_metrics": chop_metrics,
                    })
```

- [ ] **Step 3: Verify daemon still imports cleanly**

```
./venv/bin/python -c "import ast; ast.parse(open('cli/kalshi_daemon.py').read()); print('OK')"
./venv/bin/python -c "from cli.kalshi_daemon import KalshiDaemon; print('OK')"
```

Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add cli/kalshi_daemon.py
git commit -m "feat: emit chop_metrics on MARKET_SELECT; stash on pending bet"
```

---

### Task 5: Emit chop_metrics in `SETTLED` log

**Files:**
- Modify: `cli/kalshi_daemon.py:1606-1621` — `SETTLED` emission.

- [ ] **Step 1: Replace the SETTLED `_log_trade_debug` call**

```python
                        # Log settlement details for debugging
                        self._log_trade_debug(
                            asset=asset, action="SETTLED",
                            details={
                                "result": result_str,
                                "bet_side": bet.get("side"),
                                "bet_direction": bet.get("direction"),
                                "kalshi_result": result,
                                "strike": bet.get("strike"),
                                "settled_value": settled_value,
                                "entry_price": entry_price,
                                "count": count,
                                "pnl_cents": pnl_cents,
                                "ticker": m.get("ticker"),
                                "demo": self.demo,
                                **(bet.get("chop_metrics") or {}),
                            }
                        )
```

The `**(bet.get("chop_metrics") or {})` pattern is defensive: if the bet has no `chop_metrics` (recovered positions that never had entry-time compute), the spread is empty and no keys are added. If it exists (all subsequent bets), the 8 fields land alongside the settlement data.

- [ ] **Step 2: Verify syntax**

```
./venv/bin/python -c "import ast; ast.parse(open('cli/kalshi_daemon.py').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add cli/kalshi_daemon.py
git commit -m "feat: carry chop_metrics into SETTLED log"
```

---

### Task 6: Emit chop_metrics in `M10_EXIT` and `M10_RESTING_SELL` logs

**Files:**
- Modify: `cli/kalshi_daemon.py:1028-1037` — `M10_RESTING_SELL` emission.
- Modify: `cli/kalshi_daemon.py:1080-1089` — `M10_EXIT` emission.

- [ ] **Step 1: Update `M10_RESTING_SELL`**

Replace:

```python
                # Log the resting sell
                self._log_trade_debug(
                    asset=asset, action="M10_RESTING_SELL",
                    details={
                        "side": side, "direction": dir_label,
                        "m10_prob": prob, "distance": distance,
                        "bid": current_bid, "sell_price": self.MIN_SELL_PRICE,
                        "entry": entry, "count": bet.get("count", 0),
                        "ticker": ticker,
                    }
                )
```

with:

```python
                # Log the resting sell
                self._log_trade_debug(
                    asset=asset, action="M10_RESTING_SELL",
                    details={
                        "side": side, "direction": dir_label,
                        "m10_prob": prob, "distance": distance,
                        "bid": current_bid, "sell_price": self.MIN_SELL_PRICE,
                        "entry": entry, "count": bet.get("count", 0),
                        "ticker": ticker,
                        **(bet.get("chop_metrics") or {}),
                    }
                )
```

- [ ] **Step 2: Update `M10_EXIT`**

Replace:

```python
            # Bid >= 10c: exit at market
            self._log_trade_debug(
                asset=asset, action="M10_EXIT",
                details={
                    "side": side, "direction": dir_label,
                    "m10_prob": prob, "m10_side": m10_side,
                    "distance": distance, "bid": current_bid,
                    "entry": entry, "count": bet.get("count", 0),
                    "ticker": ticker,
                }
            )
```

with:

```python
            # Bid >= 10c: exit at market
            self._log_trade_debug(
                asset=asset, action="M10_EXIT",
                details={
                    "side": side, "direction": dir_label,
                    "m10_prob": prob, "m10_side": m10_side,
                    "distance": distance, "bid": current_bid,
                    "entry": entry, "count": bet.get("count", 0),
                    "ticker": ticker,
                    **(bet.get("chop_metrics") or {}),
                }
            )
```

- [ ] **Step 3: Verify syntax**

```
./venv/bin/python -c "import ast; ast.parse(open('cli/kalshi_daemon.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add cli/kalshi_daemon.py
git commit -m "feat: carry chop_metrics into M10 exit logs"
```

---

### Task 7: Tag recovered positions with an all-None chop_metrics dict

**Files:**
- Modify: `cli/kalshi_daemon.py` — `_recover_positions`, both the open-position append (~line 3409) and the resting-order append (~line 3509).

- [ ] **Step 1: Add an all-None constant near the top of `_recover_positions`**

Find the `def _recover_positions` method. At the top of the method body, before any position/order iteration, add:

```python
        _empty_chop = {
            "bbw_15m": None, "atr_pct_15m": None,
            "bbw_1h": None, "atr_pct_1h": None,
            "bbw_15m_mkt": None, "atr_pct_15m_mkt": None,
            "bbw_1h_mkt": None, "atr_pct_1h_mkt": None,
        }
```

This is scoped to the method — no class-level constant needed. One dict used in both appends.

- [ ] **Step 2: Update the open-position `bet_entry` append**

Find the `bet_entry = {...}` block near line 3400. Add one entry:

```python
            bet_entry = {
                # ... existing fields ...
                "ticker": ticker,
                "live": True,
                "needs_fill_check": False,
                "_recovered": True,
                "chop_metrics": _empty_chop,
            }
```

- [ ] **Step 3: Update the resting-order `resting_entry` append**

Find `resting_entry = {...}` near line 3509. Add:

```python
            resting_entry = {
                # ... existing fields ...
                "ticker": ticker,
                "live": True,
                "needs_fill_check": True,
                "_ws_detected": False,
                "_recovered": True,
                "chop_metrics": _empty_chop,
            }
```

- [ ] **Step 4: Verify syntax**

```
./venv/bin/python -c "import ast; ast.parse(open('cli/kalshi_daemon.py').read()); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add cli/kalshi_daemon.py
git commit -m "feat: tag recovered positions with all-None chop_metrics"
```

---

### Task 8: Integration smoke test

**Files:**
- No code changes — this task verifies the whole chain end-to-end with a small synthetic script.

- [ ] **Step 1: Run the daemon's full test suite**

```
./venv/bin/pytest tests/test_chop_metrics.py tests/test_daemon.py tests/test_kalshi.py -v
```

Expected: all pass.

- [ ] **Step 2: Inline schema check via python**

Run this one-liner to verify the helper returns the expected schema on a daemon with an empty cache (proves the keys are always present):

```
./venv/bin/python -c "
from cli.kalshi_daemon import KalshiDaemon
d = KalshiDaemon.__new__(KalshiDaemon)
d._kalshi_cached_dataframes = {}
out = d._compute_chop_metrics('BTC')
expected = {'bbw_15m', 'atr_pct_15m', 'bbw_1h', 'atr_pct_1h',
            'bbw_15m_mkt', 'atr_pct_15m_mkt', 'bbw_1h_mkt', 'atr_pct_1h_mkt'}
assert set(out.keys()) == expected, out
assert all(v is None for v in out.values()), out
print('schema OK')
"
```

Expected: `schema OK`.

- [ ] **Step 3: Dry-run readiness check**

Optional, requires live dashboard access — start the daemon in `--dry-run` for one 15-min window, then verify:

```
./venv/bin/python -c "
import json
from collections import Counter
recent = []
for line in open('data/store/trade_debug.jsonl'):
    try: recent.append(json.loads(line))
    except: pass
recent = recent[-200:]
ms = [r for r in recent if r.get('action') == 'MARKET_SELECT']
if not ms:
    print('no recent MARKET_SELECT — run a dry-run window first')
else:
    r = ms[-1]
    missing = [k for k in ['bbw_15m','atr_pct_15m','bbw_1h','atr_pct_1h',
                           'bbw_15m_mkt','atr_pct_15m_mkt','bbw_1h_mkt','atr_pct_1h_mkt']
               if k not in r]
    print('missing keys:', missing or 'none')
    print('sample:', {k: r.get(k) for k in ['bbw_15m','atr_pct_15m','bbw_1h','atr_pct_1h']})
"
```

Expected after a dry-run window: `missing keys: none` and a sample dict of actual numeric values. This step is informational — it does not gate the commit.

- [ ] **Step 4: Final commit (if any cleanup)**

No changes expected in this task; if the integration check revealed an issue, fix inline, add a regression test, then:

```bash
git add -A
git commit -m "chore: integration verification for chop_metrics logging"
```

---

## Self-Review

**Spec coverage:**
- 8 fields with exact formulas → Task 2 (`_bbw_from_df`, `_atr_pct_from_df`, `_compute_chop_metrics`).
- `_compute_chop_metrics` helper signature + behavior → Task 2.
- Call site at bet placement → Task 4.
- Emit on MARKET_SELECT → Task 4.
- Stash on `_pending_bets` → Task 4.
- Emit on SETTLED → Task 5.
- Emit on M10_EXIT and M10_RESTING_SELL → Task 6.
- `ORDER_PLACED` / `EARLY_EXIT` / `EXIT_REVIEW` NOT modified → implicit (no changes to those log sites).
- No-fallback failure behavior → Task 2's None returns.
- Recovered positions get all-None dict → Task 7.
- Dashboard unchanged → implicit (no dashboard files modified).
- Unit tests for all 5 enumerated edge cases → Task 3 (full-data in Task 1 + 5 edges in Task 3 = 6 total; spec listed 5 — the extra `degenerate_bbw` test is over-delivery, not a gap).
- Integration check → Task 8.

**Placeholder scan:** No TBDs, no "handle appropriately," no unexplained "similar to". Every code step has concrete code.

**Type consistency:** `_compute_chop_metrics` returns dict with 8 named keys; all downstream uses spread via `**chop_metrics` or `**(bet.get("chop_metrics") or {})`. Sub-helpers `_bbw_from_df` and `_atr_pct_from_df` return `float | None` and are used consistently.

All good.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-04-13-chop-detection-logging.md`.
