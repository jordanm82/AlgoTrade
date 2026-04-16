#!/usr/bin/env python3
"""Kalshi 15-minute prediction daemon — standalone K15 UpDown bot.

Usage:
    python -m cli.kalshi_daemon --dry-run --predictor v3 --cycles 10
"""

import argparse
import fcntl
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from termcolor import colored

from config.production import MAX_CONCURRENT_KALSHI_BETS, KALSHI_RISK_PER_BET_PCT
from config.settings import CDP_KEY_FILE, DATA_DIR
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.m10_feature_builder import (
    build_common_feature_vector,
    compute_m10_intra_from_window_candles,
    filter_completed_candles,
)
from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3, KalshiV3Signal, MAX_BET_PRICE
from strategy.snapshot import build_minute3_snapshot, compute_btc_confluence

# Intervals in seconds
TICK_INTERVAL = 60

# Kalshi multi-timeframe evaluation
KALSHI_CUTOFF_MINUTES = 13     # no new bets after this minute in the 15m window
KALSHI_LASTLOOK_MINUTE = 12    # elevated threshold window
KALSHI_THRESHOLD_BOOST = 10    # added to per-asset threshold for last-look


class KalshiDaemon:
    """Standalone Kalshi 15-minute prediction bot."""

    # Kalshi series tickers — BTC, ETH, SOL, XRP (no BNB — not on Coinbase)
    KALSHI_PAIRS = {
        "BTC/USDT": "KXBTC15M",
        "ETH/USDT": "KXETH15M",
        "SOL/USDT": "KXSOL15M",
        "XRP/USDT": "KXXRP15M",
    }

    # Per-asset M0 entry thresholds (33 features, no rsi_15m)
    KALSHI_THRESHOLDS = {
        "BTC/USDT": 60,
        "ETH/USDT": 60,
        "SOL/USDT": 60,
        "XRP/USDT": 60,
    }

    # Per-asset M10 exit thresholds (39 features, with rsi_15m + 5m intra-window)
    KALSHI_M10_THRESHOLDS = {
        "BTC/USDT": 85,
        "ETH/USDT": 85,
        "SOL/USDT": 85,
        "XRP/USDT": 85,
    }

    # Coinbase symbol mapping for live price (matches BRTI settlement source)
    COINBASE_PRICE_MAP = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "XRP/USDT": "XRP-USD",
    }

    # Live indicator warmup depth (must be sufficiently deep to match backtest/training indicators).
    CANDLE_LIMIT_15M = 300
    CANDLE_LIMIT_1H = 200
    CANDLE_LIMIT_4H = 120

    def __init__(self, dry_run: bool = True, predictor_version: str = "v3", demo: bool = False,
                 max_bets: int = 0, max_size_pct: float = 0):
        self.dry_run = dry_run
        self.demo = demo  # use Kalshi demo exchange (real orders, play money)
        self.kalshi_predictor_version = predictor_version
        # CLI overrides — 0 means use defaults
        self._cli_max_bets = max_bets      # max concurrent bets (0 = use default)
        self._cli_max_size_pct = max_size_pct / 100 if max_size_pct > 0 else 0  # e.g. 2.5 → 0.025
        self.fetcher = DataFetcher()
        self._running = False

        # Dataframes keyed by binance symbol (e.g. "BTC/USDT")
        self._dataframes: dict[str, pd.DataFrame] = {}

        # Kalshi predictor + client
        if predictor_version == "v3":
            self.kalshi_predictor = KalshiPredictorV3()
        elif predictor_version == "v2":
            from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
            self.kalshi_predictor = KalshiPredictorV2()
        else:
            from strategy.strategies.kalshi_predictor import KalshiPredictor
            self.kalshi_predictor = KalshiPredictor()

        self.kalshi_client = None  # lazy init
        self.kalshi_ws = None      # WebSocket for real-time prices
        self._brti_proxy = None    # multi-exchange BRTI approximation
        self._m10_model = None     # M10 confirmation model (loaded lazily)
        self._m10_scaler = None
        self.kalshi_threshold = 30  # minimum confidence to bet
        self.kalshi_predictions: list[dict] = []  # latest predictions for dashboard
        self._active_kalshi_bets = {}  # {ticker: placement_time}
        self._kalshi_pending_signals = {}   # {asset: {direction, base_conf, last_5m_conf, ...}}
        self._last_kalshi_eval = 0          # timestamp of last eval
        self._kalshi_cached_dataframes = {}  # {symbol: DataFrame} cached 15m data
        self._btc_cached_1m: pd.DataFrame | None = None  # BTC 1m candles for confluence
        self._resting_orders: list[dict] = []  # orders waiting for price to dip
        self._coinbase_auth_hint_shown = False
        self._parity_warn_window_token: str | None = None
        self._parity_warn_once_keys: set[str] = set()
        self._perf_idle_window_token: str | None = None
        self._window_heartbeat_token: str | None = None
        self._resting_eow_cancel_token: str | None = None
        self._price_skip_log_state: dict[str, dict[str, float | int]] = {}
        self._kalshi_market_strike_cache: dict[str, float] = {}
        self._kalshi_event_target_cache: dict[str, float | None] = {}
        self._kalshi_prefetch_window_token: str | None = None
        self._m0_anchor_window_token: str | None = None
        self._m0_anchor_prices: dict[str, float] = {}
        self._kx_extra_window_token: str | None = None
        self._kx_extra_cache: dict[tuple[str, float, str], dict] = {}

        # Bet tracking — records ALL bets (live + dry-run) and checks settlement
        self._pending_bets: list[dict] = []
        self._completed_bets: list[dict] = []
        self._session_wins = 0
        self._session_losses = 0       # full losses (held to settlement and lost)
        self._session_partial_losses = 0  # early exits (model changed, exited at a loss)
        self._session_good_exits = 0     # early exits that saved money (would have lost more)
        self._session_bad_exits = 0      # early exits that cost money (would have won)
        self._session_bets_placed = 0

        # Dry-run simulated balance — starts from actual Kalshi balance or $100
        # Compounds with wins/losses so sizing is realistic
        self._dry_balance_cents = 10000  # updated in startup from actual balance

        # Runtime safety: single-instance lock + settlement idempotency cache
        self._instance_lock_path = "data/store/kalshi_daemon.lock"
        self._instance_lock_fd = None
        self._settlement_idempotency_keys: set[str] = set()

        # Dashboard compatibility stubs (spot trading attributes not used)
        self.kalshi_only = True
        self._equity = 0.0
        self._pnl_today = 0.0
        self.executor = None
        self.tracker = None

    def _update_equity(self):
        """Stub — KalshiDaemon doesn't track equity."""
        pass

    def _enforce_stops(self):
        """Stub — KalshiDaemon doesn't have stop-losses."""
        pass

    def signal_cycle(self):
        """Stub — KalshiDaemon uses _kalshi_eval() instead."""
        return []

    # ------------------------------------------------------------------
    # Runtime safety
    # ------------------------------------------------------------------

    def _acquire_instance_lock(self) -> bool:
        """Acquire non-blocking process lock to prevent duplicate daemon instances."""
        if self._instance_lock_fd is not None:
            return True

        os.makedirs(os.path.dirname(self._instance_lock_path), exist_ok=True)
        fd = open(self._instance_lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = ""
            try:
                fd.seek(0)
                holder = fd.read().strip()
            except Exception:
                holder = ""
            try:
                fd.close()
            except Exception:
                pass
            holder_msg = f" (holder={holder})" if holder else ""
            print(colored(
                f"[LOCK] Another dashboard/daemon instance is already running{holder_msg}. "
                f"Lock file: {self._instance_lock_path}",
                "red",
            ))
            return False

        fd.seek(0)
        fd.truncate()
        fd.write(str(os.getpid()))
        fd.flush()
        self._instance_lock_fd = fd
        return True

    def _release_instance_lock(self):
        """Release process lock if held."""
        if self._instance_lock_fd is None:
            return
        try:
            self._instance_lock_fd.seek(0)
            self._instance_lock_fd.truncate()
            self._instance_lock_fd.flush()
        except Exception:
            pass
        try:
            fcntl.flock(self._instance_lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._instance_lock_fd.close()
        except Exception:
            pass
        self._instance_lock_fd = None

    @staticmethod
    def _settlement_idempotency_key(bet: dict, market_ticker: str | None = None) -> str:
        """Stable idempotency key for one bet settlement event."""
        ticker = market_ticker or bet.get("ticker") or "unknown_ticker"
        settle_time = bet.get("settle_time")
        if isinstance(settle_time, datetime):
            settle_token = settle_time.astimezone(timezone.utc).isoformat()
        else:
            settle_token = str(settle_time or "unknown_settle_time")
        side = (bet.get("side") or "unknown_side").lower()
        order_id = str(bet.get("order_id") or "").strip()
        bet_time = bet.get("bet_time")
        if order_id:
            uniq = f"order:{order_id}"
        elif isinstance(bet_time, datetime):
            uniq = f"bet_time:{bet_time.astimezone(timezone.utc).isoformat()}"
        else:
            uniq = f"count:{int(bet.get('count') or 0)}"
        return f"{ticker}|{settle_token}|{side}|{uniq}"

    def _maybe_log_coinbase_auth_hint(self, err) -> None:
        """Emit a one-time hint for common Coinbase auth/IP allowlist failures."""
        if self._coinbase_auth_hint_shown:
            return
        msg = str(err)
        if "transaction_summary" in msg and "401" in msg and "Unauthorized" in msg:
            print(colored(
                "  [COINBASE AUTH] Coinbase returned 401 on authenticated endpoint "
                "(transaction_summary). Likely API key auth/IP allowlist mismatch "
                "after IP change. Update allowlist/key, then restart daemon.",
                "red",
            ))
            self._coinbase_auth_hint_shown = True

    def _log_parity_warn_once(
        self,
        scope: str,
        detail: str,
        asset: str | None = None,
        window_start: datetime | None = None,
    ) -> None:
        """Log parity warning once per unique root cause per 15-minute window."""
        if window_start is None:
            now_utc = datetime.now(timezone.utc)
            minute_in_window = now_utc.minute % 15
            window_start = now_utc.replace(
                minute=now_utc.minute - minute_in_window, second=0, microsecond=0
            )
        if window_start.tzinfo is None:
            window_start = window_start.replace(tzinfo=timezone.utc)
        else:
            window_start = window_start.astimezone(timezone.utc)
        token = window_start.strftime("%Y-%m-%dT%H:%MZ")
        if token != self._parity_warn_window_token:
            self._parity_warn_window_token = token
            self._parity_warn_once_keys.clear()

        key = f"{token}:{scope}:{detail}"
        if key in self._parity_warn_once_keys:
            return
        self._parity_warn_once_keys.add(key)
        asset_hint = f" (first asset={asset})" if asset else ""
        print(colored(
            f"  [PARITY WARN] {scope} failed: {detail}{asset_hint} (further duplicates suppressed)",
            "red",
        ))

    def _log_price_skip_throttled(self, asset: str, side: str, ask_cents: int, max_cents: int) -> None:
        """Log high-ask skip at most once per asset/window/ask (or every 10s if unchanged)."""
        pending = self._kalshi_pending_signals.get(asset, {})
        ws = pending.get("window_start")
        if isinstance(ws, datetime):
            if ws.tzinfo is None:
                ws = ws.replace(tzinfo=timezone.utc)
            else:
                ws = ws.astimezone(timezone.utc)
            ws_token = ws.strftime("%Y-%m-%dT%H:%MZ")
        else:
            ws_token = "unknown"

        key = f"{asset}:{side}:{ws_token}:{max_cents}"
        now_ts = time.time()
        state = self._price_skip_log_state.get(key)
        should_log = (
            state is None
            or int(state.get("ask", -1)) != int(ask_cents)
            or (now_ts - float(state.get("ts", 0))) >= 10
        )
        if not should_log:
            return

        self._price_skip_log_state[key] = {"ask": int(ask_cents), "ts": now_ts}
        print(colored(
            f"  [KALSHI REST] {asset} {side.upper()} ask {ask_cents}c > {max_cents}c — skipping (no fill)",
            "yellow",
        ))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    _EVENT_TARGET_RE = re.compile(
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*target",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _parse_event_target_strike(cls, title: str | None) -> float | None:
        if not title:
            return None
        m = cls._EVENT_TARGET_RE.search(title)
        if not m:
            return None
        try:
            value = float(m.group(1).replace(",", ""))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _get_event_target_strike(self, event_ticker: str) -> float | None:
        """Fetch strike target from Kalshi event metadata."""
        if not event_ticker:
            return None
        if event_ticker in self._kalshi_event_target_cache:
            return self._kalshi_event_target_cache[event_ticker]

        self._init_kalshi_client()
        if not self.kalshi_client:
            self._kalshi_event_target_cache[event_ticker] = None
            return None

        strike = None
        try:
            event_resp = self.kalshi_client.get_event(event_ticker)
            event = event_resp.get("event", event_resp)
            strike = self._parse_event_target_strike(event.get("title"))
        except Exception:
            strike = None

        self._kalshi_event_target_cache[event_ticker] = strike
        return strike

    def _extract_market_strike(self, symbol: str, market: dict) -> float | None:
        """Extract strike/anchor for a Kalshi market.

        Supports legacy strike contracts (`floor_strike`/`custom_strike`) and
        K15 event-target strikes from Kalshi event metadata.
        """
        raw_floor = market.get("floor_strike")
        if raw_floor is not None:
            try:
                floor = float(raw_floor)
                if floor > 0:
                    return floor
            except (TypeError, ValueError):
                pass

        raw_custom = market.get("custom_strike")
        if raw_custom is not None:
            try:
                custom = float(raw_custom)
                if custom > 0:
                    return custom
            except (TypeError, ValueError):
                pass

        ticker = market.get("ticker", "")
        if ticker and ticker in self._kalshi_market_strike_cache:
            return self._kalshi_market_strike_cache[ticker]

        event_ticker = market.get("event_ticker", "")
        anchor = self._get_event_target_strike(event_ticker)
        if not anchor:
            return None
        if ticker:
            self._kalshi_market_strike_cache[ticker] = anchor
        return anchor

    def _fetch_all(self, include_higher_timeframes: bool = True):
        """Fetch and cache 15m candles for all pairs (parallel).

        Optionally refresh 1h + 4h caches. Entry-time boundary refresh should
        stay lightweight, so callers can skip higher timeframes.

        Uses explicit `since` parameter to ensure we get the LATEST candles
        including any that just completed. limit=50 alone can miss the most
        recent candle due to API lag.
        """
        from concurrent.futures import ThreadPoolExecutor
        import time as _time

        # Compute 'since' timestamps to guarantee fresh data
        now_ms = int(_time.time() * 1000)
        since_15m = now_ms - self.CANDLE_LIMIT_15M * 900_000
        since_1h = now_ms - self.CANDLE_LIMIT_1H * 3_600_000
        since_4h = now_ms - self.CANDLE_LIMIT_4H * 14_400_000

        def _fetch_symbol(symbol):
            """Fetch all timeframes for one symbol."""
            results = {}
            try:
                df = self.fetcher.ohlcv(
                    symbol, timeframe="15m", limit=self.CANDLE_LIMIT_15M, since=since_15m
                )
                if df is None or df.empty:
                    df = self.fetcher.ohlcv(symbol, timeframe="15m", limit=self.CANDLE_LIMIT_15M)
                df = add_indicators(df)
                pct = df["close"].pct_change()
                df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
                df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
                df["ema_slope"] = df["ema_12"].pct_change(3) * 100
                df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
                df["hourly_return"] = df["close"].pct_change(4) * 100
                results["15m"] = df
            except Exception as e:
                self._maybe_log_coinbase_auth_hint(e)
                print(colored(f"  [WARN] 15m fetch failed for {symbol}: {e}", "yellow"))
            if include_higher_timeframes:
                try:
                    df_1h = self.fetcher.ohlcv(symbol, "1h", limit=self.CANDLE_LIMIT_1H, since=since_1h)
                    if df_1h is None or df_1h.empty:
                        df_1h = self.fetcher.ohlcv(symbol, "1h", limit=self.CANDLE_LIMIT_1H)
                    if df_1h is not None and not df_1h.empty:
                        results["1h"] = add_indicators(df_1h)
                except Exception as e:
                    self._maybe_log_coinbase_auth_hint(e)
                    print(colored(f"  [FETCH] {symbol} 1h failed: {e}", "yellow"))
                try:
                    df_4h = self.fetcher.ohlcv(symbol, "4h", limit=self.CANDLE_LIMIT_4H, since=since_4h)
                    if df_4h is None or df_4h.empty:
                        df_4h = self.fetcher.ohlcv(symbol, "4h", limit=self.CANDLE_LIMIT_4H)
                    if df_4h is not None and not df_4h.empty:
                        results["4h"] = add_indicators(df_4h)
                except Exception as e:
                    self._maybe_log_coinbase_auth_hint(e)
                    print(colored(f"  [FETCH] {symbol} 4h failed: {e}", "yellow"))
            return symbol, results

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = list(pool.map(_fetch_symbol, self.KALSHI_PAIRS.keys()))

        for symbol, results in futures:
            if "15m" in results:
                self._dataframes[symbol] = results["15m"]
                self._kalshi_cached_dataframes[symbol] = results["15m"]
            if "1h" in results:
                self._kalshi_cached_dataframes[f"{symbol}_1h"] = results["1h"]
            if "4h" in results:
                self._kalshi_cached_dataframes[f"{symbol}_4h"] = results["4h"]

    def _refresh_higher_timeframes(self):
        """Lightweight refresh of 1h/4h candles.

        Keep raw rows (including in-progress). Selection of completed rows is
        done at scoring time via `filter_completed_candles(...)` for parity.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_ht(symbol):
            results = {}
            try:
                df_1h = self.fetcher.ohlcv(symbol, "1h", limit=self.CANDLE_LIMIT_1H)
                if df_1h is not None and len(df_1h) > 0:
                    results["1h"] = add_indicators(df_1h)
            except Exception as e:
                self._maybe_log_coinbase_auth_hint(e)
                print(colored(f"  [REFRESH] {symbol} 1h failed: {e}", "yellow"))
            try:
                df_4h = self.fetcher.ohlcv(symbol, "4h", limit=self.CANDLE_LIMIT_4H)
                if df_4h is not None and len(df_4h) > 0:
                    results["4h"] = add_indicators(df_4h)
            except Exception as e:
                self._maybe_log_coinbase_auth_hint(e)
                print(colored(f"  [REFRESH] {symbol} 4h failed: {e}", "yellow"))
            return symbol, results

        with ThreadPoolExecutor(max_workers=4) as pool:
            for symbol, results in pool.map(_fetch_ht, self.KALSHI_PAIRS.keys()):
                if "1h" in results:
                    self._kalshi_cached_dataframes[f"{symbol}_1h"] = results["1h"]
                if "4h" in results:
                    self._kalshi_cached_dataframes[f"{symbol}_4h"] = results["4h"]

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

    def _prefetch_kalshi_markets(self, *, force: bool = False):
        """Batch-fetch all K15 open + settled markets in 2 API calls.

        Caches results so per-asset strike/extra queries don't need individual API calls.
        """
        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15
        window_start = now_utc.replace(minute=now_utc.minute - minute_in_window, second=0, microsecond=0)
        window_token = window_start.strftime("%Y-%m-%dT%H:%MZ")
        if not force and self._kalshi_prefetch_window_token == window_token:
            return

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        self._kalshi_open_markets_cache = {}  # {series_ticker: [markets]}
        self._kalshi_settled_markets_cache = {}  # {series_ticker: [markets]}

        try:
            # Parallelize per-series open+settled calls to reduce boundary latency.
            from concurrent.futures import ThreadPoolExecutor

            def _get_markets(series: str, status: str) -> list:
                fast_getter = getattr(self.kalshi_client, "get_markets_fast", None)
                if callable(fast_getter):
                    return fast_getter(series_ticker=series, status=status, timeout=1.5)
                return self.kalshi_client.get_markets(series_ticker=series, status=status)

            def _fetch_series(series: str):
                errors = []
                open_m = None
                settled_m = None
                try:
                    open_m = _get_markets(series, "open")
                except Exception as e:
                    errors.append(f"{series} open: {e}")
                try:
                    settled_m = _get_markets(series, "settled")
                except Exception as e:
                    errors.append(f"{series} settled: {e}")
                return series, open_m, settled_m, errors

            series_list = list(self.KALSHI_PAIRS.values())
            with ThreadPoolExecutor(max_workers=min(4, len(series_list) or 1)) as pool:
                for series, open_m, settled_m, errors in pool.map(_fetch_series, series_list):
                    if open_m is not None:
                        self._kalshi_open_markets_cache[series] = open_m or []
                    if settled_m is not None:
                        self._kalshi_settled_markets_cache[series] = settled_m or []
                    for err in errors:
                        self._log_parity_warn_once("_prefetch_kalshi_markets", err)
            self._kalshi_prefetch_window_token = window_token
        except Exception as e:
            print(colored(f"  [WARN] Kalshi market prefetch failed: {e}", "yellow"))

    # ------------------------------------------------------------------
    # Coinbase price
    # ------------------------------------------------------------------

    def _get_kalshi_extra(
        self,
        asset: str,
        symbol: str,
        strike: float,
        *,
        ws_override: datetime | None = None,
        alt_price_mode: str = "m0_open",
        anchor_prices: dict[str, float] | None = None,
    ) -> dict | None:
        """Compute Kalshi-specific + cross-asset confluence + regime detection features.

        Strict no-fallback semantics:
        if required context cannot be built, return None so caller skips scoring.
        """
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        extra = {"asset": asset, "hour": hour}

        series = self.KALSHI_PAIRS.get(symbol, "")
        if not series:
            self._log_parity_warn_once("_get_kalshi_extra", f"no Kalshi series for {symbol}", asset)
            return None

        if ws_override is not None:
            if ws_override.tzinfo is not None:
                ws_kx = ws_override.astimezone(timezone.utc).replace(tzinfo=None)
            else:
                ws_kx = ws_override
        else:
            now_ws_kx = datetime.now(timezone.utc)
            ws_kx = now_ws_kx.replace(
                minute=now_ws_kx.minute - (now_ws_kx.minute % 15), second=0, microsecond=0
            ).replace(tzinfo=None)
        ws_kx_utc = ws_kx.replace(tzinfo=timezone.utc)

        def _fetch_markets(series_ticker: str, status: str) -> list:
            cache_attr = "_kalshi_settled_markets_cache" if status == "settled" else "_kalshi_open_markets_cache"
            cache_obj = getattr(self, cache_attr, None)
            if isinstance(cache_obj, dict):
                cached = cache_obj.get(series_ticker)
                # Empty lists should NOT be treated as authoritative; retry live.
                if cached:
                    return cached
            if self.kalshi_client is None:
                self._init_kalshi_client()
            if self.kalshi_client is None:
                raise ValueError(f"Kalshi client unavailable ({status}:{series_ticker})")
            fast_getter = getattr(self.kalshi_client, "get_markets_fast", None)
            if callable(fast_getter):
                markets = fast_getter(series_ticker=series_ticker, status=status, timeout=1.5)
            else:
                markets = self.kalshi_client.get_markets(series_ticker=series_ticker, status=status)
            markets = markets or []
            if isinstance(cache_obj, dict):
                cache_obj[series_ticker] = markets
            return markets

        try:
            settled = _fetch_markets(series, "settled")
            if not settled:
                raise ValueError(f"no settled markets for {series}")
            settled = sorted(settled, key=lambda m: m.get("close_time", ""), reverse=True)

            # ATR + regime source data must exist at this point-in-time.
            df_15m = self._kalshi_cached_dataframes.get(symbol)
            if df_15m is None or len(df_15m) < 50:
                raise ValueError(f"insufficient 15m cache for {symbol}")
            df_15m_filt = df_15m[df_15m.index < pd.Timestamp(ws_kx)]
            if len(df_15m_filt) < 50:
                raise ValueError(f"insufficient filtered 15m rows for {symbol}")

            atr = float(df_15m_filt.iloc[-1].get("atr", np.nan))
            if pd.isna(atr) or atr <= 0:
                raise ValueError(f"invalid ATR for {symbol}")

            atr_s = df_15m_filt["atr"].dropna()
            if len(atr_s) < 20:
                raise ValueError(f"insufficient ATR history for {symbol}")
            r20 = atr_s.rolling(20)
            mn, mx = float(r20.min().iloc[-1]), float(r20.max().iloc[-1])
            if not (mx > mn):
                raise ValueError(f"degenerate ATR percentile window for {symbol}")
            extra["atr_percentile"] = (atr - mn) / (mx - mn)

            if not strike or strike <= 0:
                raise ValueError(f"invalid strike for {symbol}: {strike}")

            prev_results = []
            prev_strikes = []
            for mk in settled[:3]:
                result = (mk.get("result") or "").lower()
                mk_strike = self._extract_market_strike(symbol, mk)
                if result in ("yes", "no"):
                    prev_results.append(1 if result == "yes" else 0)
                if mk_strike and mk_strike > 0:
                    prev_strikes.append(float(mk_strike))

            if not prev_results:
                raise ValueError(f"missing previous settlement results for {series}")
            if not prev_strikes:
                raise ValueError(f"missing previous settlement strikes for {series}")

            extra["prev_result"] = prev_results[0]
            extra["prev_3_yes_pct"] = sum(prev_results) / len(prev_results)

            streak = 0
            last_r = prev_results[0]
            for r in prev_results:
                if r == last_r:
                    streak += 1
                else:
                    break
            extra["streak_length"] = streak if last_r == 1 else -streak

            extra["strike_delta"] = (strike - prev_strikes[0]) / atr
            if len(prev_strikes) >= 2:
                deltas = [strike - prev_strikes[0]]
                for idx in range(len(prev_strikes) - 1):
                    deltas.append(prev_strikes[idx] - prev_strikes[idx + 1])
                extra["strike_trend_3"] = sum(d / atr for d in deltas) / len(deltas)
            else:
                extra["strike_trend_3"] = extra["strike_delta"]

            # === Cross-asset confluence features ===
            all_assets = list(self.KALSHI_PAIRS.keys())
            alt_symbols = [s for s in all_assets if s.split("/")[0] != asset]
            alt_rsi_15m, alt_rsi_1h, alt_momentum = [], [], []
            alt_prev_results, alt_distances = [], []

            for alt_sym in alt_symbols:
                alt_15m = self._kalshi_cached_dataframes.get(alt_sym)
                if alt_15m is None or len(alt_15m) < 2:
                    raise ValueError(f"missing 15m cache for {alt_sym}")
                alt_15m_filt = alt_15m[alt_15m.index < pd.Timestamp(ws_kx)]
                if len(alt_15m_filt) < 2:
                    raise ValueError(f"insufficient filtered 15m rows for {alt_sym}")
                alt_rsi_val = float(alt_15m_filt.iloc[-1].get("rsi", np.nan))
                if pd.isna(alt_rsi_val):
                    raise ValueError(f"invalid 15m RSI for {alt_sym}")
                alt_rsi_15m.append(alt_rsi_val)
                alt_momentum.append(1 if alt_rsi_val >= 50 else -1)

                alt_1h = self._kalshi_cached_dataframes.get(f"{alt_sym}_1h")
                if alt_1h is None or len(alt_1h) < 2:
                    raise ValueError(f"missing 1h cache for {alt_sym}")
                alt_1h_filt = filter_completed_candles(alt_1h, ws_kx, "1h")
                if len(alt_1h_filt) < 2:
                    raise ValueError(f"insufficient filtered 1h rows for {alt_sym}")
                alt_rsi_1h_val = float(alt_1h_filt.iloc[-1].get("rsi", np.nan))
                if pd.isna(alt_rsi_1h_val):
                    raise ValueError(f"invalid 1h RSI for {alt_sym}")
                alt_rsi_1h.append(alt_rsi_1h_val)

                alt_series = self.KALSHI_PAIRS.get(alt_sym, "")
                if not alt_series:
                    raise ValueError(f"missing Kalshi series for {alt_sym}")

                alt_settled = _fetch_markets(alt_series, "settled")
                if not alt_settled:
                    raise ValueError(f"no settled markets for {alt_series}")
                alt_sorted = sorted(alt_settled, key=lambda m: m.get("close_time", ""), reverse=True)
                alt_res = (alt_sorted[0].get("result") or "").lower()
                if alt_res not in ("yes", "no"):
                    raise ValueError(f"missing latest settlement result for {alt_series}")
                alt_prev_results.append(1 if alt_res == "yes" else 0)

                if self._brti_proxy is None:
                    from data.brti_proxy import BRTIProxy
                    self._brti_proxy = BRTIProxy()
                usd_sym = alt_sym.replace("/USDT", "/USD")
                alt_price = None
                if alt_price_mode == "m10_open":
                    alt_window = self._brti_proxy.get_5m_window_candles(usd_sym, ws_kx_utc)
                    if alt_window and alt_window.get("minute_10") is not None:
                        alt_price = float(alt_window["minute_10"]["open"])
                elif alt_price_mode == "m0_open":
                    if anchor_prices:
                        alt_price = anchor_prices.get(alt_sym)
                    if alt_price is None:
                        alt_price = self._brti_proxy.get_5m_open_at(usd_sym, ws_kx_utc)
                elif alt_price_mode == "spot":
                    alt_price = self._brti_proxy.get_price(usd_sym)
                else:
                    raise ValueError(f"unsupported alt_price_mode={alt_price_mode}")
                if alt_price is None:
                    # Training path tolerates missing alt anchor for distance confluence.
                    # Keep strict behavior for required fields, but skip this optional term.
                    continue

                alt_open = _fetch_markets(alt_series, "open")
                alt_strike = None
                for mk in alt_open:
                    s = self._extract_market_strike(alt_sym, mk)
                    if s and s > 0:
                        alt_strike = float(s)
                        break
                if alt_strike is None:
                    # Optional for alt_distance_avg only.
                    continue

                alt_atr = float(alt_15m_filt.iloc[-1].get("atr", np.nan))
                if pd.isna(alt_atr) or alt_atr <= 0:
                    continue
                alt_distances.append((alt_price - alt_strike) / alt_atr)

            if not alt_rsi_15m or not alt_rsi_1h or not alt_prev_results:
                raise ValueError(f"incomplete confluence bundle for {asset}")

            extra["alt_rsi_avg"] = sum(alt_rsi_15m) / len(alt_rsi_15m)
            extra["alt_rsi_1h_avg"] = sum(alt_rsi_1h) / len(alt_rsi_1h)
            extra["alt_momentum_align"] = sum(alt_momentum)
            all_prev = [extra["prev_result"]] + alt_prev_results
            extra["prev_result_consensus"] = sum(1 for r in all_prev if r == 1) / len(all_prev)
            extra["alt_distance_avg"] = sum(alt_distances) / len(alt_distances) if alt_distances else 0.0

            # === Regime features ===
            filt = df_15m_filt
            if len(filt) < 48:
                raise ValueError(f"insufficient regime history for {symbol}")
            close_now = float(filt.iloc[-1]["close"])
            close_4h = float(filt.iloc[-16]["close"])
            close_12h = float(filt.iloc[-48]["close"])
            if close_4h == 0 or close_12h == 0:
                raise ValueError(f"invalid regime denominator for {symbol}")
            extra["return_4h"] = (close_now - close_4h) / close_4h * 100
            extra["return_12h"] = (close_now - close_12h) / close_12h * 100

            alt_1h_self = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
            if alt_1h_self is None:
                raise ValueError(f"missing 1h cache for {symbol}")
            h1f = filter_completed_candles(alt_1h_self, ws_kx, "1h")
            if len(h1f) < 20:
                raise ValueError(f"insufficient filtered 1h rows for {symbol}")
            atr_val = float(filt.iloc[-1].get("atr", np.nan))
            if pd.isna(atr_val) or atr_val <= 0:
                raise ValueError(f"invalid ATR for {symbol}")
            h1_sma = float(h1f["close"].rolling(20).mean().iloc[-1])
            extra["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - h1_sma) / atr_val

            alt_4h_self = self._kalshi_cached_dataframes.get(f"{symbol}_4h")
            if alt_4h_self is None:
                raise ValueError(f"missing 4h cache for {symbol}")
            h4f = filter_completed_candles(alt_4h_self, ws_kx, "4h")
            if len(h4f) < 10:
                raise ValueError(f"insufficient filtered 4h rows for {symbol}")
            extra["lower_lows_4h"] = sum(
                1 for i in range(-3, 0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i - 1]["low"])
            )
            h4_sma = float(h4f["close"].rolling(10).mean().iloc[-1])
            extra["trend_strength"] = (float(h4f.iloc[-1]["close"]) - h4_sma) / atr_val

            return extra
        except Exception as e:
            self._log_parity_warn_once("_get_kalshi_extra", str(e), asset, ws_kx_utc)
            return None

    def _get_kalshi_strike(self, series_ticker: str) -> tuple:
        """Get strike price and close time for current open market.

        Parity rule: use Kalshi `floor_strike` only. No substitution from settled data.
        Returns: (strike, close_time_dt, ticker) or (None, None, None)
        """
        self._init_kalshi_client()
        if not self.kalshi_client:
            return None, None, None
        symbol = next((sym for sym, series in self.KALSHI_PAIRS.items() if series == series_ticker), None)
        if not symbol:
            return None, None, None

        try:
            def _fetch_open_markets_live() -> list:
                fast_getter = getattr(self.kalshi_client, "get_markets_fast", None)
                if callable(fast_getter):
                    return fast_getter(series_ticker=series_ticker, status="open", timeout=1.5)
                return self.kalshi_client.get_markets(series_ticker=series_ticker, status="open")

            def _is_future_candidate(mk: dict, now_chk: datetime) -> bool:
                ct = mk.get("close_time", "")
                if not (ct and "T" in ct):
                    return False
                try:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                except Exception:
                    return False
                if close_dt <= now_chk:
                    return False
                # Legacy strike field present
                raw_floor = mk.get("floor_strike")
                if raw_floor is not None:
                    try:
                        if float(raw_floor) > 0:
                            return True
                    except (TypeError, ValueError):
                        pass
                raw_custom = mk.get("custom_strike")
                if raw_custom is not None:
                    try:
                        if float(raw_custom) > 0:
                            return True
                    except (TypeError, ValueError):
                        pass
                # New up/down schema: recover anchor from open_time path.
                return bool(mk.get("open_time"))

            def _has_future_strike(mkts: list) -> bool:
                now_chk = datetime.now(timezone.utc)
                for mk in mkts or []:
                    if _is_future_candidate(mk, now_chk):
                        return True
                return False

            # Use prefetched cache if available AND non-empty, otherwise query API live
            if (hasattr(self, '_kalshi_open_markets_cache')
                    and series_ticker in self._kalshi_open_markets_cache
                    and self._kalshi_open_markets_cache[series_ticker]):
                markets = self._kalshi_open_markets_cache[series_ticker]
                # Cache can be momentarily stale/incomplete right after window rollover.
                # If no usable future strike exists, force a live refresh now.
                if not _has_future_strike(markets):
                    markets = _fetch_open_markets_live()
                    if hasattr(self, '_kalshi_open_markets_cache'):
                        self._kalshi_open_markets_cache[series_ticker] = markets or []
            else:
                markets = _fetch_open_markets_live()
                # Update cache with fresh results
                if hasattr(self, '_kalshi_open_markets_cache'):
                    self._kalshi_open_markets_cache[series_ticker] = markets or []
            if not markets:
                return None, None, None

            # Only consider markets whose close_time is in the future
            # (prevents using a just-settled market that Kalshi hasn't transitioned yet)
            now_utc = datetime.now(timezone.utc)
            future_markets = []
            for mk in markets:
                ct = mk.get("close_time", "")
                if ct and "T" in ct:
                    close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if close_dt > now_utc:
                        future_markets.append(mk)
            if not future_markets:
                # Cache had stale data — clear it so next call queries live
                if hasattr(self, '_kalshi_open_markets_cache') and series_ticker in self._kalshi_open_markets_cache:
                    del self._kalshi_open_markets_cache[series_ticker]
                return None, None, None

            # Pick the soonest future market with a valid floor_strike.
            future_markets.sort(key=lambda x: x.get("close_time", "9999"))
            for m in future_markets:
                strike = self._extract_market_strike(symbol, m)
                if not strike:
                    continue
                ticker = m.get("ticker", "")
                ct = m.get("close_time") or m.get("expiration_time", "")
                close_time_dt = None
                if ct and "T" in ct:
                    close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                return float(strike), close_time_dt, ticker

            return None, None, None

        except Exception as e:
            print(colored(f"  [V3] Kalshi API error for {series_ticker}: {e}", "yellow"))
            return None, None, None

    def _get_coinbase_price(self, symbol: str) -> float | None:
        """Get live price from Coinbase (closer to BRTI than BinanceUS)."""
        try:
            from coinbase.rest import RESTClient
            cb_symbol = self.COINBASE_PRICE_MAP.get(symbol)
            if not cb_symbol:
                return None
            client = RESTClient(key_file=str(CDP_KEY_FILE))
            product = client.get_product(cb_symbol)
            return float(product["price"])
        except Exception as e:
            self._maybe_log_coinbase_auth_hint(e)
            return None

    # ------------------------------------------------------------------
    # BTC confluence scoring
    # ------------------------------------------------------------------

    def _fetch_btc_1m(self):
        """Fetch recent BTC 1m candles for confluence scoring."""
        try:
            df = self.fetcher.ohlcv(self.BTC_SYMBOL, "1m", limit=10)
            if df is not None and not df.empty:
                self._btc_cached_1m = df
        except Exception:
            pass

    def compute_btc_score(self, bet_side: str) -> int:
        """Compute BTC confluence score (0-100) for an alt bet.

        Uses BTC's recent 1m returns to measure how strongly BTC's
        momentum supports the proposed bet direction.

        50 = neutral, 70+ = BTC confirms, 30- = BTC opposes
        """
        if self._btc_cached_1m is None or len(self._btc_cached_1m) < 4:
            return 50  # neutral if no data

        df = self._btc_cached_1m
        closes = df["close"].values
        if len(closes) < 4:
            return 50

        # Recent 1m returns (1-bar, 2-bar, 3-bar)
        c0 = float(closes[-1])
        c1 = float(closes[-2])
        c2 = float(closes[-3])
        c3 = float(closes[-4])

        if c1 <= 0 or c2 <= 0 or c3 <= 0:
            return 50

        r1 = (c0 - c1) / c1 * 100
        r2 = (c0 - c2) / c2 * 100
        r3 = (c0 - c3) / c3 * 100

        score = 50
        dm = 1 if bet_side == "YES" else -1

        # 1-bar (most recent 1m) — up to ±20 pts
        a1 = r1 * dm
        score += min(20, max(-20, a1 * 67))

        # 2-bar (2 min trend) — up to ±15 pts
        score += min(15, max(-15, r2 * dm * 30))

        # 3-bar (3 min trend) — up to ±10 pts
        score += min(10, max(-10, r3 * dm * 15))

        # Multi-timeframe agreement bonus ±5
        all_aligned = a1 > 0 and r2 * dm > 0 and r3 * dm > 0
        all_opposing = a1 < 0 and r2 * dm < 0 and r3 * dm < 0
        if all_aligned:
            score += 5
        elif all_opposing:
            score -= 5

        return max(0, min(100, round(score)))

    # ------------------------------------------------------------------
    # Kalshi client
    # ------------------------------------------------------------------

    def _init_kalshi_client(self):
        """Lazy-initialize the Kalshi client.

        Demo mode: uses demo API + demo keys (real orders, play money).
        Dry-run: uses production API for market data, skips order placement.
        Live: uses production API for everything.
        """
        if self.kalshi_client is not None:
            return
        try:
            from exchange.kalshi import KalshiClient

            if self.demo:
                from config.settings import KALSHI_DEMO_KEY_FILE
                # Demo key file format: first line = "API: <key-id>" or just key-id
                with open(KALSHI_DEMO_KEY_FILE) as f:
                    first_line = f.readline().strip()
                # Strip "API: " or "API:" prefix if present
                demo_key_id = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
                self.kalshi_client = KalshiClient(
                    api_key_id=demo_key_id,
                    private_key_path=str(KALSHI_DEMO_KEY_FILE),
                    demo=True,
                )
            else:
                from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
                self.kalshi_client = KalshiClient(
                    api_key_id=KALSHI_API_KEY_ID,
                    private_key_path=str(KALSHI_KEY_FILE),
                    demo=False,
                )
        except Exception as e:
            print(colored(f"  [WARN] Kalshi client init failed: {e}", "yellow"))

    # ------------------------------------------------------------------
    # Kalshi WebSocket
    # ------------------------------------------------------------------

    def _start_kalshi_ws(self):
        """Start WebSocket for real-time Kalshi contract prices."""
        try:
            from exchange.kalshi_ws import KalshiWebSocket

            if self.demo:
                from config.settings import KALSHI_DEMO_KEY_FILE
                with open(KALSHI_DEMO_KEY_FILE) as f:
                    first_line = f.readline().strip()
                key_id = first_line.split(":", 1)[-1].strip() if ":" in first_line else first_line
                key_path = str(KALSHI_DEMO_KEY_FILE)
                is_demo = True
            else:
                from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
                key_id = KALSHI_API_KEY_ID
                key_path = str(KALSHI_KEY_FILE)
                is_demo = False

            self.kalshi_ws = KalshiWebSocket(
                api_key_id=key_id,
                private_key_path=key_path,
                demo=is_demo,
            )
            self.kalshi_ws.start()
            print(colored("  [WS] Kalshi WebSocket started", "green"))

            # Subscribe to current open markets
            self._ws_subscribe_current_markets()

        except Exception as e:
            print(colored(f"  [WS] Failed to start: {e}", "yellow"))
            self.kalshi_ws = None

    def _ws_subscribe_current_markets(self):
        """Subscribe to ticker updates for current open K15 markets."""
        if not self.kalshi_ws:
            return
        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        for series_ticker in self.KALSHI_PAIRS.values():
            try:
                markets = None
                if (hasattr(self, "_kalshi_open_markets_cache")
                        and isinstance(self._kalshi_open_markets_cache, dict)
                        and self._kalshi_open_markets_cache.get(series_ticker)):
                    markets = self._kalshi_open_markets_cache.get(series_ticker)
                if not markets:
                    markets = self.kalshi_client.get_markets(
                        series_ticker=series_ticker, status="open"
                    )
                if markets:
                    markets.sort(key=lambda m: m.get("close_time", "9999"))
                    ticker = markets[0].get("ticker", "")
                    if ticker:
                        self.kalshi_ws.subscribe_ticker(ticker)
            except Exception:
                pass

    def _get_ws_price(self, symbol: str, side: str = "yes") -> int | None:
        """Get real-time contract price from WebSocket. Returns cents or None."""
        if not self.kalshi_ws:
            return None

        series = self.KALSHI_PAIRS.get(symbol, "")
        if not series:
            return None

        # Find the ticker for current open market
        tickers = self.kalshi_ws.get_all_tickers()
        for ticker, data in tickers.items():
            if series[:6] in ticker:  # e.g. "KXBTC1" matches "KXBTC15M-..."
                age = time.time() - data.get("ts", 0)
                if age > 60:  # stale data
                    continue
                if side == "yes":
                    return data.get("yes_ask", 0)
                else:
                    return data.get("no_ask", 0)
        return None

    # ------------------------------------------------------------------
    # M10 confirmation — separate model trained on minute-10 data
    # ------------------------------------------------------------------

    MIN_SELL_PRICE = 10  # floor: won't sell below 10c, place resting sell instead

    def _load_m10_model(self):
        """Lazy-load per-asset M10 models (39 features each)."""
        if self._m10_model is not None:
            return
        try:
            import pickle
            from pathlib import Path
            # Load per-asset M10 models
            self._m10_per_asset = {}
            for asset_code in ["btc", "eth", "sol", "xrp"]:
                pa_path = Path(f"models/m10_{asset_code}.pkl")
                if pa_path.exists():
                    with open(pa_path, "rb") as f:
                        pa_data = pickle.load(f)
                        self._m10_per_asset[asset_code.upper()] = (
                            pa_data.get("model") or pa_data.get("knn"),
                            pa_data["scaler"],
                            pa_data["feature_names"],
                        )
            if self._m10_per_asset:
                # Use first per-asset model as sentinel so lazy-load doesn't re-trigger
                first = next(iter(self._m10_per_asset.values()))
                self._m10_model = first[0]
                self._m10_scaler = first[1]
                print(colored(f"  [M10] Loaded {len(self._m10_per_asset)} per-asset models", "green"))
            else:
                print(colored("  [M10] No per-asset models found", "yellow"))
        except Exception as e:
            print(colored(f"  [M10] Model load failed: {e}", "yellow"))

    def _m10_confirm(self):
        """At minute 10, run per-asset M10 model (39 features) to decide: hold or exit.

        M10 model has 39 features: 33 M0 features + rsi_15m + 5 intra-window 5m features.
        Threshold 90/10 — only exits on very high-conviction disagreement.
        """
        if not self._pending_bets:
            return

        self._load_m10_model()
        if self._m10_model is None:
            return

        if self._brti_proxy is None:
            from data.brti_proxy import BRTIProxy
            self._brti_proxy = BRTIProxy()

        for bet in list(self._pending_bets):
            if bet.get("result") or bet.get("count", 0) <= 0:
                continue
            if bet.get("_m10_checked"):
                continue

            asset = bet.get("asset", "?")
            side = bet.get("side", "yes")
            symbol = bet.get("symbol", "")
            strike = bet.get("strike", 0)
            ticker = bet.get("ticker", "")
            entry = bet.get("contract_price", bet.get("fill_price", 0))

            if not strike or not symbol:
                continue

            # Window anchors for strict point-in-time feature calculation.
            now_m10 = datetime.now(timezone.utc)
            ws_m10_utc = now_m10.replace(
                minute=now_m10.minute - (now_m10.minute % 15), second=0, microsecond=0
            )
            ws_m10 = ws_m10_utc.replace(tzinfo=None)

            # Strict parity: use Coinbase+Bitstamp 5m candles for minute-0/5/10.
            usd_sym = symbol.replace("/USDT", "/USD")
            window_5m = self._brti_proxy.get_5m_window_candles(usd_sym, ws_m10_utc)
            if not window_5m:
                continue
            c1 = window_5m.get("minute_0")
            c2 = window_5m.get("minute_5")
            c3 = window_5m.get("minute_10")
            if c1 is None or c2 is None or c3 is None:
                continue
            current_price = float(c3["open"])

            # Get ATR from cached 15m data — filter to BEFORE window start (matches backtest)
            df_15m = self._kalshi_cached_dataframes.get(symbol)
            if df_15m is None or len(df_15m) < 5:
                continue
            df_15m_filtered = df_15m[df_15m.index < pd.Timestamp(ws_m10)]
            if len(df_15m_filtered) < 5:
                continue
            pr = df_15m_filtered.iloc[-1]
            atr = float(pr.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            distance = (current_price - strike) / atr

            df_1h = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
            df_4h = self._kalshi_cached_dataframes.get(f"{symbol}_4h")
            # Kalshi + confluence extras at ws_m10 with minute-10 anchor pricing.
            kx = self._get_kalshi_extra(
                asset,
                symbol,
                strike,
                ws_override=ws_m10,
                alt_price_mode="m10_open",
            )
            if not kx:
                print(colored(f"  [M10 SKIP] {asset}: missing Kalshi context — skipping M10 score", "yellow"))
                continue

            feat = build_common_feature_vector(
                pr,
                df_1h,
                df_4h,
                pd.Timestamp(ws_m10),
                distance,
                kalshi_extra=kx,
                atr_pctile_val=kx.get("atr_percentile", 0.5),
            )
            if feat is None:
                continue

            intra = compute_m10_intra_from_window_candles(c1, c2, atr)
            if intra is None:
                continue
            feat.update(intra)

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            # Select per-asset M10 model (39 features) or fall back to unified
            if hasattr(self, '_m10_per_asset') and asset in self._m10_per_asset:
                pa_model, pa_scaler, pa_features = self._m10_per_asset[asset]
                missing = [f for f in pa_features if f not in feat]
                if missing:
                    print(colored(f"  [M10] Missing features for {asset}: {missing}", "yellow"))
                    continue
                X = np.array([feat[f] for f in pa_features], dtype=float).reshape(1, -1)
                # XGBoost models take raw features; linear-model artifacts use scaled.
                is_tree = hasattr(pa_model, 'get_booster') or 'XGB' in type(pa_model).__name__
                X_pred = X if is_tree else pa_scaler.transform(X)
                prob = float(pa_model.predict_proba(X_pred)[0][1])
            else:
                feature_names = self._m10_scaler.feature_names_in_ if hasattr(self._m10_scaler, 'feature_names_in_') else [
                    "macd_15m", "norm_return", "ema_slope", "roc_5",
                    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
                    "vol_ratio", "adx", "rsi_1h", "rsi_4h", "distance_from_strike",
                ]
                X = np.array([feat[f] for f in feature_names]).reshape(1, -1)
                prob = float(self._m10_model.predict_proba(self._m10_scaler.transform(X))[0][1])
            pct = int(prob * 100)

            m10_thresh = self.KALSHI_M10_THRESHOLDS.get(symbol, 70)
            m10_side = "yes" if pct >= m10_thresh else "no" if pct <= (100 - m10_thresh) else "skip"
            dir_label = "YES" if side == "yes" else "NO"
            bet["_m10_checked"] = True

            # Store M10 score for dashboard display
            m10_display = pct if m10_side == "yes" else (100 - pct) if m10_side == "no" else pct
            if asset in self._kalshi_pending_signals:
                self._kalshi_pending_signals[asset]["confirmed_m10"] = True
                self._kalshi_pending_signals[asset]["m10_score"] = m10_display
                self._kalshi_pending_signals[asset]["m10_side"] = m10_side

            # Does M10 agree with our bet? Skip = uncertain = hold.
            if m10_side == "skip" or side == m10_side:
                # M10 confirms or is uncertain — hold
                print(colored(
                    f"  [M10 CONFIRM] {asset} {dir_label} dist={distance:+.2f} M10={pct}% — holding",
                    "green",
                ))
                continue

            # M10 disagrees — exit
            # Get current bid for sell price
            current_bid = None
            if self.kalshi_ws:
                ws_data = self.kalshi_ws.get_ticker(ticker)
                if ws_data and time.time() - ws_data.get("ts", 0) < 30:
                    current_bid = ws_data.get("yes_bid") if side == "yes" else ws_data.get("no_bid")
            if current_bid is None:
                self._init_kalshi_client()
                if self.kalshi_client:
                    try:
                        mkt = self.kalshi_client.get_market(ticker)
                        mkt_data = mkt.get("market", mkt)
                        if side == "yes":
                            current_bid = int(float(mkt_data.get("yes_bid_dollars", 0)) * 100)
                        else:
                            current_bid = int(float(mkt_data.get("no_bid_dollars", 0)) * 100)
                    except Exception:
                        pass

            if not current_bid or current_bid <= 0:
                # No bid available — treat as below floor, place resting sell at floor
                current_bid = 0

            # 10c floor (or no bid): place resting sell at floor price
            if current_bid < self.MIN_SELL_PRICE:
                print(colored(
                    f"  [M10 EXIT] {asset} {dir_label} M10={pct}% bid={current_bid}c < {self.MIN_SELL_PRICE}c — "
                    f"resting sell @ {self.MIN_SELL_PRICE}c",
                    "yellow",
                ))
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

                if not (self.dry_run and not self.demo):
                    # Live: place resting sell at floor price
                    sell_count = bet.get("count", 0)
                    if sell_count <= 0:
                        # Verify actual position from exchange
                        try:
                            positions = self.kalshi_client.get_positions()
                            for p in positions:
                                if p.get("ticker") == ticker:
                                    sell_count = abs(int(float(p.get("position_fp", p.get("position", 0)))))
                                    break
                        except Exception:
                            pass
                    if sell_count > 0:
                        try:
                            self.kalshi_client.place_order(
                                ticker=ticker, side=side, count=sell_count,
                                price_cents=self.MIN_SELL_PRICE,
                                order_type="limit", action="sell",
                            )
                        except Exception as e:
                            print(colored(f"  [M10 SELL ERR] {asset}: {e}", "red"))
                    else:
                        print(colored(f"  [M10 SELL] {asset} no position to sell", "yellow"))

                # Mark as exited — prevents settlement from double-counting
                pnl_cents = bet.get("count", 0) * (self.MIN_SELL_PRICE - entry)
                pnl_dollars = pnl_cents / 100
                bet["result"] = "PL"
                bet["pnl_cents"] = pnl_cents
                bet["pnl_dollars"] = pnl_dollars
                bet["exited_early"] = True
                self._completed_bets.append(bet)
                self._session_partial_losses += 1
                # Update dry-run balance (was missing — caused P&L vs balance mismatch)
                if self.dry_run and not self.demo:
                    self._dry_balance_cents += pnl_cents
                self._pending_bets = [b for b in self._pending_bets if b is not bet]
                continue

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
            self._exit_position(bet, current_bid,
                f"M10 EXIT (M10={pct}% {m10_side.upper()}, dist={distance:+.2f})")

    # ------------------------------------------------------------------
    # Stop loss + minute-5 confirmation (legacy, disabled)
    # ------------------------------------------------------------------

    STOP_LOSS_PCT = 0.50  # sell if contract drops to 50% of entry

    def _check_stop_losses(self):
        """Check pending bets for stop loss — sell if contract at 50% of entry."""
        if not self._pending_bets:
            return

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        for bet in list(self._pending_bets):
            if bet.get("result") or bet.get("count", 0) <= 0:
                continue
            if not bet.get("live") and not self.dry_run:
                continue

            asset = bet.get("asset", "?")
            side = bet.get("side", "yes")
            entry = bet.get("contract_price", bet.get("fill_price", 0))
            ticker = bet.get("ticker", "")
            count = bet.get("count", 0)

            if not entry or not ticker:
                continue

            stop_price = int(entry * self.STOP_LOSS_PCT)

            # Get current contract price from WebSocket or API
            current_price = None
            if self.kalshi_ws:
                ws_data = self.kalshi_ws.get_ticker(ticker)
                if ws_data and time.time() - ws_data.get("ts", 0) < 30:
                    current_price = ws_data.get("yes_bid") if side == "yes" else ws_data.get("no_bid")

            if current_price is None:
                # Fallback to REST API
                try:
                    mkt = self.kalshi_client.get_market(ticker)
                    mkt_data = mkt.get("market", mkt)
                    if side == "yes":
                        current_price = int(float(mkt_data.get("yes_bid_dollars", 0)) * 100)
                    else:
                        current_price = int(float(mkt_data.get("no_bid_dollars", 0)) * 100)
                except Exception:
                    continue

            if current_price and current_price <= stop_price:
                # STOP LOSS HIT — sell
                self._exit_position(bet, current_price, "STOP LOSS")

    def _confirm_with_distance(self, minute: int):
        """At minute 5 and 10, recheck with real distance data. Exit if model flipped."""
        if not self._pending_bets:
            return

        checkpoint = f"{minute}m"
        flag_key = f"confirmed_{checkpoint}"

        # Get fresh multi-exchange prices
        if self._brti_proxy is None:
            from data.brti_proxy import BRTIProxy
            self._brti_proxy = BRTIProxy()

        for bet in list(self._pending_bets):
            if bet.get("result") or bet.get("count", 0) <= 0:
                continue
            if bet.get(flag_key):
                continue  # already checked at this checkpoint

            asset = bet.get("asset", "?")
            side = bet.get("side", "yes")
            symbol = bet.get("symbol", "")
            strike = bet.get("strike", 0)
            ticker = bet.get("ticker", "")

            if not strike or not symbol:
                continue

            usd_sym = symbol.replace("/USDT", "/USD")
            current_crypto = self._brti_proxy.get_price(usd_sym)
            if not current_crypto:
                continue

            df_15m = self._kalshi_cached_dataframes.get(symbol)
            if df_15m is None or len(df_15m) < 5:
                continue
            atr = float(df_15m.iloc[-1].get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            distance = (current_crypto - strike) / atr

            # Re-run the model with actual distance to see if prediction changed
            # Build features from cached indicators + real distance
            prev = df_15m.iloc[-1]
            sma_val = float(prev.get("sma_20", 0))
            adx_val = float(prev.get("adx", 20))
            close_val = float(prev.get("close", 0))
            ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
            pve = float(prev.get("price_vs_ema", 0)) if pd.notna(prev.get("price_vs_ema")) else 0
            hr = float(prev.get("hourly_return", 0)) if pd.notna(prev.get("hourly_return")) else 0

            import numpy as np
            feat = {
                "macd_15m": float(prev.get("macd_hist", 0)),
                "norm_return": float(prev.get("norm_return", 0)) if pd.notna(prev.get("norm_return")) else 0,
                "ema_slope": float(prev.get("ema_slope", 0)) if pd.notna(prev.get("ema_slope")) else 0,
                "roc_5": float(prev.get("roc_5", 0)),
                "macd_1h": 0.0, "price_vs_ema": pve, "hourly_return": hr,
                "trend_direction": adx_val * ts_sign,
                "vol_ratio": float(prev.get("vol_ratio", 1)) if pd.notna(prev.get("vol_ratio")) else 1,
                "adx": adx_val, "rsi_1h": 50.0, "rsi_4h": 50.0,
                "distance_from_strike": distance,
            }

            # Get 1h/4h context
            df_1h = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
            df_4h = self._kalshi_cached_dataframes.get(f"{symbol}_4h")
            ws_naive = pd.Timestamp.now()
            if df_1h is not None:
                m1h = filter_completed_candles(df_1h, ws_naive, "1h")
                if len(m1h) >= 20:
                    feat["rsi_1h"] = float(m1h.iloc[-1].get("rsi", 50))
                    feat["macd_1h"] = float(m1h.iloc[-1].get("macd_hist", 0))
            if df_4h is not None:
                m4h = filter_completed_candles(df_4h, ws_naive, "4h")
                if len(m4h) >= 10:
                    feat["rsi_4h"] = float(m4h.iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            # Run model with real distance
            feature_names = self.kalshi_predictor._knn_scaler.feature_names_in_ if hasattr(self.kalshi_predictor._knn_scaler, 'feature_names_in_') else [
                "macd_15m", "norm_return", "ema_slope", "roc_5",
                "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
                "vol_ratio", "adx", "rsi_1h", "rsi_4h", "distance_from_strike",
            ]
            X = np.array([feat[f] for f in feature_names]).reshape(1, -1)
            prob = float(self.kalshi_predictor._knn.predict_proba(
                self.kalshi_predictor._knn_scaler.transform(X))[0][1])
            pct = int(prob * 100)

            # Did the model change? Exit if it flipped OR lost confidence (went to SKIP)
            model_side = "yes" if pct >= 55 else "no" if pct <= 45 else "skip"
            flipped = (side != model_side)  # exit if model no longer agrees with our bet

            dir_label = "YES" if side == "yes" else "NO"

            if flipped:
                # Model changed its mind — exit
                current_bid = None
                if self.kalshi_ws:
                    ws_data = self.kalshi_ws.get_ticker(ticker)
                    if ws_data and time.time() - ws_data.get("ts", 0) < 30:
                        current_bid = ws_data.get("yes_bid") if side == "yes" else ws_data.get("no_bid")
                if current_bid is None:
                    try:
                        mkt = self.kalshi_client.get_market(ticker)
                        mkt_data = mkt.get("market", mkt)
                        if side == "yes":
                            current_bid = int(float(mkt_data.get("yes_bid_dollars", 0)) * 100)
                        else:
                            current_bid = int(float(mkt_data.get("no_bid_dollars", 0)) * 100)
                    except Exception:
                        current_bid = None

                if current_bid and current_bid > 0:
                    # Log early exit to debug file
                    self._log_trade_debug(
                        asset=asset, action="EARLY_EXIT",
                        details={
                            "reason": f"MIN{minute}",
                            "bet_side": side,
                            "bet_direction": dir_label,
                            "model_new_side": model_side,
                            "model_prob": prob,
                            "distance": distance,
                            "entry_price": bet.get("contract_price", bet.get("fill_price", 0)),
                            "exit_price": current_bid,
                            "count": bet.get("count", 0),
                            "strike": bet.get("strike", 0),
                            "ticker": ticker,
                            "settle_time": str(bet.get("settle_time", "")),
                        }
                    )
                    # Track for post-settlement comparison
                    if not hasattr(self, '_early_exits'):
                        self._early_exits = []
                    self._early_exits.append({
                        "asset": asset,
                        "side": side,
                        "direction": dir_label,
                        "strike": bet.get("strike", 0),
                        "entry_price": bet.get("contract_price", bet.get("fill_price", 0)),
                        "exit_price": current_bid,
                        "exit_pnl_cents": bet.get("count", 0) * (current_bid - bet.get("contract_price", bet.get("fill_price", 0))),
                        "settle_time": bet.get("settle_time"),
                        "ticker": ticker,
                        "symbol": bet.get("symbol", ""),
                        "exit_time": datetime.now(timezone.utc),
                    })

                    self._exit_position(bet, current_bid,
                        f"MIN{minute} EXIT (model flipped to {model_side.upper()}, dist={distance:+.2f})")
                else:
                    print(colored(
                        f"  [MIN{minute} EXIT] {asset} {dir_label} model flipped but no bid — holding",
                        "yellow"))
            else:
                bet[flag_key] = True
                print(colored(
                    f"  [MIN{minute} CONFIRM] {asset} {dir_label} dist={distance:+.2f} "
                    f"model={pct}% — holding",
                    "green",
                ))

        # Cancel resting orders if model flipped
        for order in list(self._resting_orders):
            asset = order.get("asset", "?")
            symbol = order.get("symbol", "")
            side = order.get("side", "yes")
            strike = order.get("strike", 0)
            order_id = order.get("order_id", "")

            if not symbol or not strike or not order_id:
                continue

            usd_sym = symbol.replace("/USDT", "/USD")
            current_crypto = self._brti_proxy.get_price(usd_sym) if self._brti_proxy else None
            if not current_crypto:
                continue

            df_15m = self._kalshi_cached_dataframes.get(symbol)
            if df_15m is None or len(df_15m) < 5:
                continue
            atr = float(df_15m.iloc[-1].get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            distance = (current_crypto - strike) / atr
            against = (side == "yes" and distance < -0.1) or (side == "no" and distance > 0.1)

            if against:
                try:
                    self.kalshi_client.cancel_order_safe(order_id)
                    dir_label = "YES" if side == "yes" else "NO"
                    print(colored(
                        f"  [MIN{minute} CANCEL] {asset} {dir_label} resting — model against (dist={distance:+.2f})",
                        "yellow",
                    ))
                    self._pending_bets = [
                        b for b in self._pending_bets if b.get("order_id") != order_id
                    ]
                except Exception:
                    pass

        self._resting_orders = [
            o for o in self._resting_orders
            if o.get("order_id") and o.get("order_id") not in
            [b.get("order_id") for b in self._pending_bets if b.get("result")]
        ]

    def _exit_position(self, bet: dict, sell_price: int, reason: str):
        """Sell a position — either stop loss or minute-5 exit."""
        asset = bet.get("asset", "?")
        side = bet.get("side", "yes")
        ticker = bet.get("ticker", "")
        count = bet.get("count", 0)
        entry = bet.get("contract_price", bet.get("fill_price", 0))
        dir_label = "YES" if side == "yes" else "NO"

        pnl_cents = count * (sell_price - entry)
        pnl_dollars = pnl_cents / 100

        if self.dry_run and not self.demo:
            # Dry-run: simulate the exit
            print(colored(
                f"  [{reason}] {asset} {dir_label} sell x{count} @ {sell_price}c "
                f"(entry {entry}c) P&L ${pnl_dollars:+.2f}",
                "red" if pnl_cents < 0 else "green",
            ))
            bet["result"] = "WIN" if pnl_cents > 0 else "PL"
            bet["pnl_cents"] = pnl_cents
            bet["pnl_dollars"] = pnl_dollars
            bet["settle_price"] = sell_price
            bet["exited_early"] = True
            self._completed_bets.append(bet)
            if pnl_cents > 0:
                self._session_wins += 1
            else:
                self._session_partial_losses += 1
            self._dry_balance_cents += pnl_cents
            # Remove from pending so settlement doesn't double-count
            self._pending_bets = [b for b in self._pending_bets if b is not bet]
        else:
            # Live: query actual position size from Kalshi before selling
            try:
                positions = self.kalshi_client.get_positions()
                for p in positions:
                    if p.get("ticker") == ticker:
                        actual = int(float(p.get("position_fp", 0)))
                        if actual > count:
                            count = actual  # sell everything we hold
                            bet["count"] = count
                            pnl_cents = count * (sell_price - entry)
                            pnl_dollars = pnl_cents / 100
                        break
            except Exception:
                pass

            try:
                # Sell loop — reprice and retry until filled
                remaining = count
                total_filled = 0
                order_id = None

                for attempt in range(5):
                    # Get fresh bid price each attempt
                    if attempt > 0:
                        time.sleep(1)
                        # Cancel previous unfilled order
                        if order_id:
                            try:
                                self.kalshi_client.cancel_order_safe(order_id)
                            except Exception:
                                pass

                        # Refresh bid price
                        sell_price = None
                        if self.kalshi_ws:
                            ws_data = self.kalshi_ws.get_ticker(ticker)
                            if ws_data and time.time() - ws_data.get("ts", 0) < 30:
                                sell_price = ws_data.get("yes_bid") if side == "yes" else ws_data.get("no_bid")
                        if not sell_price:
                            try:
                                mkt = self.kalshi_client.get_market(ticker)
                                mkt_data = mkt.get("market", mkt)
                                if side == "yes":
                                    sell_price = int(float(mkt_data.get("yes_bid_dollars", 0)) * 100)
                                else:
                                    sell_price = int(float(mkt_data.get("no_bid_dollars", 0)) * 100)
                            except Exception:
                                sell_price = 1

                        # Recalc P&L with actual sell price
                        pnl_cents = count * (sell_price - entry)
                        pnl_dollars = pnl_cents / 100

                    result = self.kalshi_client.place_order(
                        ticker=ticker, side=side, count=remaining,
                        price_cents=max(1, sell_price),
                        order_type="limit",
                        action="sell",
                    )
                    order = result.get("order", {})
                    order_id = order.get("order_id")
                    fill_count = int(float(order.get("fill_count_fp", 0)))
                    total_filled += fill_count
                    remaining -= fill_count

                    print(colored(
                        f"  [{reason}] {asset} {dir_label} sell x{fill_count}/{count} @ {sell_price}c "
                        f"(entry {entry}c) attempt {attempt+1}",
                        "yellow" if remaining > 0 else ("red" if pnl_cents < 0 else "green"),
                    ))

                    if remaining <= 0:
                        break

                if total_filled > 0:
                    pnl_cents = total_filled * (sell_price - entry)
                    pnl_dollars = pnl_cents / 100
                    bet["result"] = "WIN" if pnl_cents > 0 else "PL"
                    bet["pnl_cents"] = pnl_cents
                    bet["pnl_dollars"] = pnl_dollars
                    bet["settle_price"] = sell_price
                    bet["exited_early"] = True
                    self._completed_bets.append(bet)
                    if pnl_cents > 0:
                        self._session_wins += 1
                    else:
                        self._session_partial_losses += 1
                    self._pending_bets = [b for b in self._pending_bets if b is not bet]
                    print(colored(
                        f"  [{reason}] {asset} {dir_label} EXITED x{total_filled} P&L ${pnl_dollars:+.2f}",
                        "red" if pnl_cents < 0 else "green",
                    ))
                else:
                    print(colored(
                        f"  [{reason}] {asset} {dir_label} EXIT FAILED — 0 fills after 5 attempts",
                        "red",
                    ))
            except Exception as e:
                print(colored(f"  [EXIT ERR] {asset}: {e}", "red"))

    # ------------------------------------------------------------------
    # Trade debug logging
    # ------------------------------------------------------------------

    def _log_trade_debug(self, asset: str, action: str, details: dict):
        """Log trade lifecycle events to file for debugging live/demo issues."""
        try:
            import json as _json
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "asset": asset,
                "action": action,
                **details,
            }
            with open("data/store/trade_debug.jsonl", "a") as f:
                f.write(_json.dumps(entry) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Settlement checking
    # ------------------------------------------------------------------

    def _check_dryrun_settlements(self):
        """Check if any pending bets have settled using Kalshi's actual results.

        Queries Kalshi for settled markets to get the authoritative outcome.
        No Coinbase price approximation — uses Kalshi's own settlement data.
        """
        if not self._pending_bets:
            return
        now = datetime.now(timezone.utc)
        settled = []

        # Check bets past their settlement time (15s buffer for Kalshi to process)
        due_bets = [b for b in self._pending_bets
                    if now >= b["settle_time"] + pd.Timedelta(seconds=15)]
        if not due_bets:
            return

        # Query Kalshi for settled markets
        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        for bet in due_bets:
            try:
                asset = bet["asset"]
                series = self.KALSHI_PAIRS.get(bet["symbol"], "")
                if not series:
                    continue

                # Idempotency guard: never count the same market+window+side twice
                base_key = self._settlement_idempotency_key(bet)
                if base_key in self._settlement_idempotency_keys:
                    print(colored(
                        f"  [SETTLED SKIP] Duplicate settlement ignored for {asset} "
                        f"(key={base_key})",
                        "yellow",
                    ))
                    settled.append(bet)
                    continue

                # Live/demo: check if resting order filled before settlement
                if not (self.dry_run and not self.demo) and bet.get("needs_fill_check") and bet.get("order_id"):
                    try:
                        order_info = self.kalshi_client.get_order_status(bet["order_id"])
                        filled = float(order_info.get("fill_count_fp", 0))
                        if filled > 0:
                            bet["count"] = int(filled)
                            bet["needs_fill_check"] = False
                        else:
                            # Never filled — remove without counting as W/L
                            print(colored(
                                f"  [EXPIRED] {asset} {bet.get('direction', '?')} "
                                f"@ {bet.get('contract_price', 0)}c — never filled, removing",
                                "dark_grey"))
                            settled.append(bet)
                            continue
                    except Exception:
                        pass

                # Skip unfilled live/demo orders — no W/L for unfilled bets
                if not (self.dry_run and not self.demo) and bet.get("count", 0) == 0:
                    print(colored(
                        f"  [EXPIRED] {asset} {bet.get('direction', '?')} — unfilled, removing",
                        "dark_grey"))
                    settled.append(bet)
                    continue

                # Find the settled market matching our bet's window
                settled_markets = self.kalshi_client.get_markets(
                    series_ticker=series, status="settled"
                )

                for m in settled_markets:
                    ct = m.get("close_time", "")
                    if not ct:
                        continue
                    market_close = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if abs((market_close - bet["settle_time"]).total_seconds()) < 60:
                        result = m.get("result", "")
                        if not result:
                            continue

                        market_key = self._settlement_idempotency_key(
                            bet,
                            market_ticker=m.get("ticker"),
                        )
                        if market_key in self._settlement_idempotency_keys:
                            print(colored(
                                f"  [SETTLED SKIP] Duplicate market settlement ignored for {asset} "
                                f"(key={market_key})",
                                "yellow",
                            ))
                            settled.append(bet)
                            break

                        entry_price = bet.get("contract_price", bet.get("fill_price", 0))
                        count = bet.get("count", 1)
                        settled_value = float(m.get("expiration_value", 0))

                        # Compute P&L from our known entry price + settlement result
                        # Don't use settlements API costs — they mix buy+sell transactions
                        # and produce wrong entry prices when M10 early exits happened
                        won = bet["side"] == result
                        if won:
                            pnl_cents = count * (100 - entry_price)
                        else:
                            pnl_cents = -(count * entry_price)
                        pnl_dollars = pnl_cents / 100

                        result_str = "WIN" if won else "LOSS"
                        color = "green" if won else "red"

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

                        print(colored(
                            f"  [SETTLED] {asset} {bet['direction']} "
                            f"@ {entry_price}c x{count} "
                            f"strike=${bet['strike']:,.2f} "
                            f"settled=${settled_value:,.2f} "
                            f"-> {result_str} ${pnl_dollars:+.2f}",
                            color,
                        ))

                        bet["result"] = result_str
                        bet["settle_price"] = settled_value
                        bet["kalshi_result"] = result
                        bet["pnl_cents"] = pnl_cents
                        bet["pnl_dollars"] = pnl_dollars
                        bet["contract_price"] = entry_price
                        bet["count"] = count
                        self._completed_bets.append(bet)
                        if won:
                            self._session_wins += 1
                        else:
                            self._session_losses += 1

                        # Compound dry-run simulated balance (not demo — demo uses real balance)
                        if self.dry_run and not self.demo:
                            self._dry_balance_cents += pnl_cents

                        self._settlement_idempotency_keys.add(base_key)
                        self._settlement_idempotency_keys.add(market_key)
                        settled.append(bet)
                        break

            except Exception as e:
                print(colored(f"  [SETTLE ERR] {bet['asset']}: {e}", "yellow"))

        # Remove settled bets from pending
        if settled:
            self._pending_bets = [b for b in self._pending_bets if b not in settled]

        # Check early exits against actual settlement — would we have won if we held?
        if hasattr(self, '_early_exits') and self._early_exits:
            still_pending = []
            for ex in self._early_exits:
                settle_time = ex.get("settle_time")
                if settle_time and now > settle_time + pd.Timedelta(minutes=1):
                    # Find the settlement result for this window
                    series = self.KALSHI_PAIRS.get(ex.get("symbol", ""), "")
                    if series and self.kalshi_client:
                        try:
                            mkts = self.kalshi_client.get_markets(series_ticker=series, status="settled")
                            for m in mkts:
                                ct = m.get("close_time", "")
                                if ct:
                                    mkt_close = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                                    if abs((mkt_close - settle_time).total_seconds()) < 60:
                                        result = m.get("result", "")
                                        would_win = (ex["side"] == result)
                                        entry = ex["entry_price"]
                                        count = ex.get("count", 1) or 1
                                        if would_win:
                                            would_pnl = count * (100 - entry)
                                        else:
                                            would_pnl = -(count * entry)
                                        exit_pnl = ex["exit_pnl_cents"]

                                        saved = exit_pnl > would_pnl  # early exit saved us money
                                        label = "GOOD EXIT" if saved else "BAD EXIT"
                                        color = "green" if saved else "red"
                                        if saved:
                                            self._session_good_exits += 1
                                        else:
                                            self._session_bad_exits += 1

                                        print(colored(
                                            f"  [{label}] {ex['asset']} {ex['direction']} — "
                                            f"exited at {ex['exit_price']}c (PL {exit_pnl:+d}c) | "
                                            f"would have {'WON' if would_win else 'LOST'} "
                                            f"({would_pnl:+d}c) if held",
                                            color,
                                        ))

                                        self._log_trade_debug(
                                            asset=ex["asset"], action="EXIT_REVIEW",
                                            details={
                                                "label": label,
                                                "bet_side": ex["side"],
                                                "exit_pnl_cents": exit_pnl,
                                                "would_win": would_win,
                                                "would_pnl_cents": would_pnl,
                                                "kalshi_result": result,
                                                "strike": ex["strike"],
                                                "entry_price": entry,
                                                "exit_price": ex["exit_price"],
                                            }
                                        )
                                        break
                        except Exception:
                            pass
                else:
                    still_pending.append(ex)
            self._early_exits = still_pending

    # ------------------------------------------------------------------
    # Kalshi evaluation lifecycle
    # ------------------------------------------------------------------

    def _kalshi_eval(self):
        """Kalshi evaluation — per-asset models, minute-0 entry only.

        Uses per-asset M0 models (33 features, no rsi_15m) for entry at minute 0-1,
        and per-asset M10 models (39 features, with rsi_15m + 5m intra-window) for
        exit decisions at minute 10+. Cross-asset confluence and regime detection
        features are computed for all assets. In-progress 1h/4h candles are dropped.

        Lifecycle within each 15m window:
        - ENTRY (min 0-1): Per-asset M0 signal at threshold 65, reprice up to 60c.
        - MONITORING (min 2-9): Hold position, no new entries.
        - M10 CONFIRM (min 10+): Per-asset M10 model decides hold or exit (threshold 90).
        """
        from data.market_data import get_order_book_imbalance, get_trade_flow

        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15
        eval_started = time.perf_counter()

        # Check resting orders — monitor fills, cancel at minute 10+
        if self._resting_orders and minute_in_window >= 1:
            self._check_resting_orders()

        # Minute 10+ confirmation — uses per-asset M10 models (39 features incl rsi_15m + 5m intra-window).
        # Use >= 10 instead of == 10 so we don't miss it if eval doesn't land exactly at min 10.
        # Each bet has _m10_checked flag to prevent double-runs.
        if minute_in_window >= 10:
            self._m10_confirm()

        # Settlement check moved to dashboard refresh cycle (every 15s)

        # Compute current window start (round down to :00/:15/:30/:45)
        window_minute = now_utc.minute - minute_in_window
        current_window_start = now_utc.replace(minute=window_minute, second=0, microsecond=0)
        # Pandas-compatible timestamp for DataFrame index comparisons
        current_window_start_pd = pd.Timestamp(current_window_start.replace(tzinfo=None))
        window_token = current_window_start.strftime("%Y-%m-%dT%H:%MZ")
        if window_token != self._window_heartbeat_token:
            self._window_heartbeat_token = window_token
            print(colored(
                f"  [WINDOW] {current_window_start.strftime('%H:%M')} UTC window started",
                "cyan",
            ))

        # End-of-window safety: cancel any resting orders before rollover.
        # Runs once per window in live/demo at minute 14+.
        if (not self.dry_run or self.demo) and minute_in_window >= 14:
            if self._resting_eow_cancel_token != window_token:
                self._resting_eow_cancel_token = window_token
                self._cancel_all_resting_orders("end of window")

        # At window boundary: cancel stale resting orders, refresh data once, re-subscribe WS
        boundary_refreshed = False
        if minute_in_window <= 1 and not self._kalshi_cached_dataframes.get("_refreshed_" + str(current_window_start)):
            if not self.dry_run or self.demo:
                # Always hard-reset resting orders at window rollover in live/demo,
                # including untracked exchange-side orders (e.g., 10c resting sells).
                self._cancel_all_resting_orders("window boundary")
            refresh_15m_started = time.perf_counter()
            # Keep minute-0 path lightweight: refresh 15m cache only.
            self._fetch_all(include_higher_timeframes=False)
            refresh_15m_ms = (time.perf_counter() - refresh_15m_started) * 1000
            prefetch_started = time.perf_counter()
            self._prefetch_kalshi_markets()  # batch Kalshi queries for this window
            prefetch_ms = (time.perf_counter() - prefetch_started) * 1000
            self._kalshi_cached_dataframes["_refreshed_" + str(current_window_start)] = True
            boundary_refreshed = True
            if refresh_15m_ms > 800 or prefetch_ms > 800:
                print(colored(
                    f"  [PERF] boundary refresh: 15m={refresh_15m_ms:.0f}ms "
                    f"kalshi_prefetch={prefetch_ms:.0f}ms",
                    "yellow",
                ))
            if self.kalshi_ws:
                self._ws_subscribe_current_markets()
        # Higher-timeframe (1h/4h) cache refresh — epoch-aligned to actual candle
        # completions, not fixed time buckets.
        #
        # Empirical finding: cached RSI/MACD diverge from a training-style fresh
        # fetch by up to 12 RSI points near HTF boundaries. Root cause is timing:
        # the daemon's cached 1h/4h df is missing candles that completed AFTER
        # the cache was loaded. Direct test confirms add_indicators is
        # deterministic — daemon-style and parity-style fetches at the SAME
        # moment produce identical RSI to 6 decimal places.
        #
        # Fix: key the refresh on the last-completed 1h and 4h candle OPEN times.
        # As soon as a new candle completes, the key changes and the next eval
        # triggers a refresh. The boundary_refreshed guard prevents stacking
        # this on top of the minute-0 15m-only boundary refresh in the same
        # eval tick, so M0 critical path stays lean (HTF fetch runs on the
        # SECOND eval of the new window, usually within a couple seconds).
        last_complete_1h_open = (
            now_utc.replace(minute=0, second=0, microsecond=0)
            - timedelta(hours=1)
        )
        last_complete_4h_open = (
            now_utc.replace(
                hour=(now_utc.hour // 4) * 4, minute=0, second=0, microsecond=0
            )
            - timedelta(hours=4)
        )
        htf_epoch_key = (
            f"_htf_epoch_"
            f"{last_complete_1h_open.isoformat()}_"
            f"{last_complete_4h_open.isoformat()}"
        )
        if (
            not boundary_refreshed
            and not self._kalshi_cached_dataframes.get(htf_epoch_key)
        ):
            # First eval after a 1h/4h candle completed. Fire even at minute 0
            # of hour-aligned windows — M0 scoring needs the just-completed
            # 1h candle to be present in the cache. The minute-0 15m-only
            # boundary refresh already ran in an earlier eval tick of this
            # window (boundary_refreshed guard above), so this adds one HTF
            # fetch per epoch rollover without blocking the 15m critical path.
            self._refresh_higher_timeframes()
            self._kalshi_cached_dataframes[htf_epoch_key] = True

        # Prune active bets — clear at window boundary so new window can bet immediately
        now_ts = time.time()
        window_start_ts = current_window_start.replace(tzinfo=timezone.utc).timestamp()
        self._active_kalshi_bets = {
            t: placed for t, placed in self._active_kalshi_bets.items()
            if placed >= window_start_ts  # only keep bets from THIS window
        }

        if minute_in_window >= KALSHI_CUTOFF_MINUTES:
            # Too close to settlement — show EXPIRED, but preserve BET_PLACED for bets taken this window
            expired_preds = []
            for sym in self.KALSHI_PAIRS:
                asset = sym.split("/")[0]
                pending = self._kalshi_pending_signals.get(asset, {})
                if pending.get("bet_placed") and pending.get("window_start") == current_window_start:
                    expired_preds.append({
                        "symbol": sym, "asset": asset,
                        "direction": pending.get("direction", "--"),
                        "confidence": pending.get("last_5m_conf", 0),
                        "reason": "bet placed -- awaiting settlement",
                        "ob": 0, "flow": 0, "state": "SETTLING",
                    })
                else:
                    expired_preds.append({
                        "symbol": sym, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "expired -- too close to settlement",
                        "ob": 0, "flow": 0, "state": "EXPIRED",
                    })
            self.kalshi_predictions = expired_preds
            return

        predictions = []
        actionable_signals = []

        if self._kx_extra_window_token != window_token:
            self._kx_extra_window_token = window_token
            self._kx_extra_cache = {}

        # Pre-fetch pricing data
        _prefetched_prices = {}  # minute-0 5m OPEN (Coinbase+Bitstamp), matches training M0 anchor

        # Multi-exchange anchor prices for M0: exact 5m candle OPEN at window start.
        if self._brti_proxy is None:
            from data.brti_proxy import BRTIProxy
            self._brti_proxy = BRTIProxy()
        from concurrent.futures import ThreadPoolExecutor

        anchor_cache_reset = self._m0_anchor_window_token != window_token
        if anchor_cache_reset:
            self._m0_anchor_window_token = window_token
            self._m0_anchor_prices = {}

        def _fetch_m0_anchor_open(sym):
            try:
                usd_sym = sym.replace("/USDT", "/USD")
                px = self._brti_proxy.get_5m_open_at(usd_sym, current_window_start)
                return sym, px
            except Exception:
                return sym, None

        # Fetch M0 anchors once per window; retry missing symbols during entry minute.
        missing_anchors = [
            sym for sym in self.KALSHI_PAIRS
            if anchor_cache_reset or sym not in self._m0_anchor_prices
        ]
        if minute_in_window > 1 and not anchor_cache_reset:
            missing_anchors = []
        if missing_anchors:
            with ThreadPoolExecutor(max_workers=min(4, len(missing_anchors))) as pool:
                for sym, px in pool.map(_fetch_m0_anchor_open, missing_anchors):
                    if px is not None:
                        self._m0_anchor_prices[sym] = px
        _prefetched_prices = dict(self._m0_anchor_prices)

        # Strike discovery is latency-critical at minute 0-1. Fetch all series in parallel first
        # so one slow asset doesn't delay readiness detection for the rest.
        strike_snapshot: dict[str, tuple[float | None, datetime | None, str | None]] = {}
        if minute_in_window <= 1:
            def _fetch_strike_snapshot(item):
                sym, series = item
                try:
                    return sym, self._get_kalshi_strike(series)
                except Exception:
                    return sym, (None, None, None)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for sym, strike_tuple in pool.map(_fetch_strike_snapshot, self.KALSHI_PAIRS.items()):
                    strike_snapshot[sym] = strike_tuple

        def _get_cached_m0_extra(asset_name: str, sym: str, strike_val: float):
            kx_key = (asset_name, float(strike_val), "m0_open")
            cached = self._kx_extra_cache.get(kx_key)
            if cached is None:
                fresh = self._get_kalshi_extra(
                    asset_name,
                    sym,
                    float(strike_val),
                    ws_override=current_window_start,
                    alt_price_mode="m0_open",
                    anchor_prices=_prefetched_prices,
                )
                if not fresh:
                    return None
                self._kx_extra_cache[kx_key] = fresh
                cached = fresh
            out = dict(cached)
            out["window_start_naive"] = current_window_start_pd
            return out

        for symbol, series_ticker in self.KALSHI_PAIRS.items():
            asset = symbol.split("/")[0]

            # Check if pending signal belongs to current window
            pending = self._kalshi_pending_signals.get(asset)
            if pending and pending.get("window_start") != current_window_start:
                # New window — clear old signal only.
                # Do NOT evict symbol candle caches here: confluence for other assets
                # depends on all symbol caches being present in the same eval cycle.
                self._kalshi_pending_signals.pop(asset, None)
                pending = None

            # Skip if already holding a position on this asset (any window)
            has_open_position = any(
                b.get("asset") == asset and not b.get("result") and b.get("count", 0) > 0
                for b in self._pending_bets
            )
            if has_open_position and not (pending and pending.get("bet_placed")):
                # Position from a prior window still open — don't double up
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": "--", "confidence": 0,
                    "knn_score": 0, "knn_thresh": 0,
                    "tbl_score": 0, "tbl_thresh": 0,
                    "btc_score": 0, "btc_thresh": 0,
                    "reason": "position already open from prior window",
                    "ob": 0, "flow": 0, "state": "POSITION_HELD",
                })
                continue

            # Skip if already bet OR already attempted this window — show stored scores.
            # bet_attempted covers unfilled/cancelled/price-skipped attempts so we don't re-fire.
            if pending and (pending.get("bet_placed") or pending.get("bet_attempted")):
                was_placed = bool(pending.get("bet_placed"))
                status_reason = (
                    "already bet this window"
                    if was_placed
                    else pending.get("skip_reason", "entry skipped this window")
                )
                status_state = "BET_PLACED" if was_placed else "BET_SKIPPED"
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending.get("direction", "--"),
                    "confidence": pending.get("last_5m_conf", pending.get("base_conf", 0)),
                    "knn_score": pending.get("bet_knn_score", 0),
                    "knn_thresh": self.KALSHI_THRESHOLDS.get(symbol, 60),
                    "tbl_score": 0,
                    "tbl_thresh": 0,
                    "btc_score": pending.get("bet_btc_score", 0),
                    "btc_thresh": 0,
                    "reason": status_reason,
                    "ob": 0, "flow": 0, "state": status_state,
                })
                continue

            # Lifecycle:
            # Min 0-1:   CONFIRMED — bet at window open (one shot + reprice)
            # Min 2-9:   MONITORING — hold position
            # Min 10+:   M10 CONFIRM — M10 model decides hold or exit
            if minute_in_window <= 1 and not (pending and (pending.get("bet_placed") or pending.get("bet_attempted"))):
                state = "CONFIRMED"
            elif minute_in_window >= 10 and pending and pending.get("bet_placed") and not pending.get("confirmed_m10"):
                state = "M10_CONFIRM"
            else:
                state = "MONITORING"

            # --- BET_PLACED or past window: show stored data ---
            if state == "MONITORING" and pending and pending.get("bet_placed"):
                # Already bet — show stored scores, state = SETTLING
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending.get("direction", "--"),
                    "confidence": pending.get("last_5m_conf", pending.get("base_conf", 0)),
                    "knn_score": pending.get("bet_knn_score", 0),
                    "knn_thresh": self.KALSHI_THRESHOLDS.get(symbol, 60),
                    "tbl_score": 0,
                    "tbl_thresh": 0,
                    "btc_score": pending.get("bet_btc_score", 0),
                    "btc_thresh": 0,
                    "reason": "bet placed — awaiting settlement",
                    "ob": 0, "flow": 0, "state": "SETTLING",
                })
                continue

            # MONITORING without bet: fall through to full eval for display
            # (the CONFIRMED code below handles both CONFIRMED and MONITORING)

            # --- SETUP: use pre-cached 15m/1h, just fetch Kalshi strike ---
            if state == "SETUP":
                df_15m = self._kalshi_cached_dataframes.get(symbol)
                if df_15m is None or len(df_15m) < 50:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "no 15m data", "ob": 0, "flow": 0, "state": state,
                    })
                    continue

                df_1h = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
                market_data = None
                ob_imb = 0
                net_flow = 0

                if self.kalshi_predictor_version == "v3":
                    # V3: query Kalshi for strike price + time remaining
                    strike, close_time_dt, market_ticker = strike_snapshot.get(
                        symbol,
                        (None, None, None),
                    )
                    if not strike:
                        strike, close_time_dt, market_ticker = self._get_kalshi_strike(series_ticker)

                    # Use pre-filtered 15m data (last completed candle = backtest parity)
                    # Strict parity price path: Coinbase+Bitstamp only (no source substitution).
                    setup_price = _prefetched_prices.get(symbol)
                    if setup_price is None:
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": "--", "confidence": 0,
                            "knn_score": 0, "knn_thresh": self.KALSHI_THRESHOLDS.get(symbol, 60),
                            "tbl_score": 0, "tbl_thresh": 0,
                            "btc_score": 0, "btc_thresh": 0,
                            "reason": "no CB+BS price for model input",
                            "ob": 0, "flow": 0, "state": state,
                        })
                        continue

                    signal = None
                    if strike and close_time_dt:
                        # Compute Kalshi-specific features for per-asset model.
                        # Strict mode: if extras fail, skip this scoring attempt.
                        kx = _get_cached_m0_extra(asset, symbol, float(strike))
                        if not kx:
                            predictions.append({
                                "symbol": symbol, "asset": asset,
                                "direction": "--", "confidence": 0,
                                "knn_score": 0, "knn_thresh": self.KALSHI_THRESHOLDS.get(symbol, 60),
                                "tbl_score": 0, "tbl_thresh": 0,
                                "btc_score": 0, "btc_thresh": 0,
                                "reason": "setup: missing Kalshi context — skipped scoring",
                                "ob": 0, "flow": 0, "state": state,
                            })
                            continue
                        try:
                            # Filter to candles BEFORE window start (matches backtest exactly)
                            df_15m_filtered = df_15m[df_15m.index < current_window_start_pd]
                            if df_15m_filtered is not None and len(df_15m_filtered) >= 20:
                                signal = self.kalshi_predictor.predict(
                                    df_15m_filtered, strike_price=float(strike),
                                    minutes_remaining=12,
                                    market_data=market_data, df_1h=df_1h,
                                    current_price=setup_price,
                                    df_4h=self._kalshi_cached_dataframes.get(f"{symbol}_4h"),
                                    kalshi_extra=kx,
                                )
                        except Exception:
                            pass
                        # Store strike info in pending signal for V3
                        if signal and asset not in self._kalshi_pending_signals:
                            # Store directional confidence (YES=raw prob, NO=100-prob)
                            _raw = int(signal.probability * 100)
                            _dir_conf = _raw if signal.recommended_side == "YES" else (100 - _raw)
                            self._kalshi_pending_signals[asset] = {
                                "strike_price": float(strike),
                                "close_time": close_time_dt,
                                "_signal_ticker": market_ticker,  # exact market for this signal
                                "probability": signal.probability,
                                "recommended_side": signal.recommended_side,
                                "max_price_cents": signal.max_price_cents,
                                "distance_atr": signal.distance_atr,
                                "bet_placed": False,
                                "bet_attempted": False,
                                "window_start": current_window_start,
                                # V1/V2 compat fields
                                "direction": signal.recommended_side if signal.recommended_side != "SKIP" else "--",
                                "base_conf": _dir_conf,
                                "last_5m_conf": _dir_conf,
                                "setup_time": now_utc,
                                "confirmed": False,
                            }
                    else:
                        signal = None

                    side = signal.recommended_side if signal and signal.recommended_side != "SKIP" else "--"
                    raw_pct = int(signal.probability * 100) if signal else 0
                    # Display confidence in the recommended direction
                    setup_conf = raw_pct if side == "YES" else (100 - raw_pct) if side == "NO" else raw_pct

                    # M0-only preview: score is directional confidence from M0.
                    m0_score = setup_conf if signal else 0
                    m0_thresh = self.KALSHI_THRESHOLDS.get(symbol, 60)

                    # BTC score preview
                    btc_preview = self.compute_btc_score(side) if side in ("YES", "NO") else 50

                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": side,
                        "confidence": setup_conf,
                        "knn_score": m0_score, "knn_thresh": m0_thresh,
                        "tbl_score": 0, "tbl_thresh": 0,
                        "btc_score": btc_preview, "btc_thresh": 0,
                        "reason": "setup [M0] — waiting for confirmation",
                        "ob": ob_imb, "flow": net_flow, "state": state,
                    })
                else:
                    # V1/V2: existing score() call
                    signal = self.kalshi_predictor.score(df_15m, market_data=market_data, df_1h=df_1h)

                    # Store direction + confidence, no betting
                    if signal is not None and asset not in self._kalshi_pending_signals:
                        self._kalshi_pending_signals[asset] = {
                            "direction": signal.direction,
                            "base_conf": signal.confidence,
                            "last_5m_conf": signal.confidence,
                            "setup_time": now_utc,
                            "confirmed": False,
                            "bet_placed": False,
                            "bet_attempted": False,
                            "window_start": current_window_start,
                        }
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": signal.direction if signal else "--",
                        "confidence": signal.confidence if signal else 0,
                        "reason": "setup -- waiting for confirmation (leading indicators will refresh)",
                        "ob": ob_imb, "flow": net_flow, "state": state,
                    })

            # --- CONFIRMED or MONITORING: full eval. Only bet if CONFIRMED. ---
            elif state in ("CONFIRMED", "MONITORING"):
                if not pending and self.kalshi_predictor_version != "v3":
                    # No SETUP signal — nothing to re-score (V1/V2 only)
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": f"{state.lower()}: no SETUP signal",
                        "ob": 0, "flow": 0, "state": state,
                    })
                    continue

                # Use CACHED data — no extra API calls in the critical path
                market_data = None
                # Multi-exchange price for distance (Coinbase+Bitstamp avg, matches training)
                cb_price = _prefetched_prices.get(symbol)
                if cb_price is None:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"] if pending else "--",
                        "confidence": 0,
                        "reason": "no CB+BS price for model input",
                        "ob": 0, "flow": 0, "state": state,
                    })
                    continue
                df_15m = self._kalshi_cached_dataframes.get(symbol)

                # If no cache (late start), run full fetch with derived features
                if df_15m is None or len(df_15m) < 50:
                    try:
                        raw = self.fetcher.ohlcv(symbol, "15m", limit=self.CANDLE_LIMIT_15M)
                        if raw is not None and not raw.empty:
                            df_15m = add_indicators(raw)
                            pct = df_15m["close"].pct_change()
                            df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
                            df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
                            df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
                            df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
                            df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100
                            self._kalshi_cached_dataframes[symbol] = df_15m
                    except Exception:
                        pass

                # Filter to candles BEFORE window start (matches backtest exactly)
                if df_15m is None or len(df_15m) < 50:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "no cached 15m data", "ob": 0, "flow": 0, "state": state,
                    })
                    continue
                df_15m = df_15m[df_15m.index < current_window_start_pd]
                if len(df_15m) < 20:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "no cached 15m data", "ob": 0, "flow": 0, "state": state,
                    })
                    continue
                if df_15m is None or len(df_15m) < 50:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"] if pending else "--",
                        "confidence": 0,
                        "reason": f"{state.lower()}: no cached 15m data",
                        "ob": 0, "flow": 0, "state": state,
                    })
                    continue

                # Use cached 1h and 4h from SETUP
                df_1h = self._kalshi_cached_dataframes.get(f"{symbol}_1h")
                df_4h = self._kalshi_cached_dataframes.get(f"{symbol}_4h")

                ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
                net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

                if self.kalshi_predictor_version == "v3":
                    # V3: get strike from pending or query Kalshi fresh
                    strike = pending.get("strike_price") if pending else None
                    close_time_dt = pending.get("close_time") if pending else None

                    signal_ticker = pending.get("_signal_ticker") if pending else None
                    if not strike:
                        # No SETUP ran — query Kalshi for strike
                        strike, close_time_dt, signal_ticker = strike_snapshot.get(
                            symbol,
                            (None, None, None),
                        )
                        if not strike:
                            strike, close_time_dt, signal_ticker = self._get_kalshi_strike(series_ticker)

                    if not strike and state == "CONFIRMED":
                        # Log once per asset per window
                        wait_key = f"_wait_logged_{asset}_{current_window_start}"
                        if minute_in_window <= 1 and not self._kalshi_cached_dataframes.get(wait_key):
                            print(colored(
                                f"  [WAIT] {asset}: no Kalshi market yet (min {minute_in_window})",
                                "dark_grey"))
                            self._kalshi_cached_dataframes[wait_key] = True
                        # If still missing by the final entry minute, log explicit skip reason.
                        final_wait_key = f"_wait_final_{asset}_{current_window_start}"
                        if minute_in_window >= 1 and not self._kalshi_cached_dataframes.get(final_wait_key):
                            print(colored(
                                f"  [WAIT] {asset}: no Kalshi market by final entry minute "
                                f"(min {minute_in_window}) — skipping window",
                                "yellow"))
                            self._kalshi_cached_dataframes[final_wait_key] = True

                    if strike and close_time_dt:
                        ready_key = f"_ready_logged_{asset}_{current_window_start}"
                        if state == "CONFIRMED" and not self._kalshi_cached_dataframes.get(ready_key):
                            lag_s = int(max(0, (now_utc - current_window_start).total_seconds()))
                            if lag_s >= 2:
                                print(colored(
                                    f"  [READY] {asset}: Kalshi market became tradable at +{lag_s}s",
                                    "dark_grey",
                                ))
                            self._kalshi_cached_dataframes[ready_key] = True
                        mins_left = max(0, (close_time_dt - now_utc).total_seconds() / 60)
                        kx = _get_cached_m0_extra(asset, symbol, float(strike))
                        if not kx:
                            predictions.append({
                                "symbol": symbol, "asset": asset,
                                "direction": pending["direction"] if pending else "--",
                                "confidence": 0,
                                "reason": f"{state.lower()}: missing Kalshi context — skipped scoring",
                                "ob": ob_imb, "flow": net_flow, "state": state,
                            })
                            continue
                        signal = self.kalshi_predictor.predict(
                            df_15m, strike_price=float(strike),
                            minutes_remaining=mins_left,
                            market_data=market_data, df_1h=df_1h,
                            current_price=cb_price,
                            df_4h=df_4h,
                            kalshi_extra=kx,
                        )
                        if signal and pending:
                            pending["probability"] = signal.probability
                            pending["recommended_side"] = signal.recommended_side
                            pending["max_price_cents"] = signal.max_price_cents
                            # Store directional confidence
                            _r = int(signal.probability * 100)
                            pending["last_5m_conf"] = _r if signal.recommended_side == "YES" else (100 - _r)
                            pending["confirmed"] = True
                        elif signal and not pending:
                            # Create pending signal on the fly (late start)
                            _r2 = int(signal.probability * 100)
                            _dc = _r2 if signal.recommended_side == "YES" else (100 - _r2)
                            self._kalshi_pending_signals[asset] = {
                                "strike_price": float(strike),
                                "close_time": close_time_dt,
                                "probability": signal.probability,
                                "recommended_side": signal.recommended_side,
                                "max_price_cents": signal.max_price_cents,
                                "distance_atr": signal.distance_atr,
                                "bet_placed": False,
                                "bet_attempted": False,
                                "window_start": current_window_start,
                                "direction": signal.recommended_side if signal.recommended_side != "SKIP" else "--",
                                "base_conf": _dc,
                                "last_5m_conf": _dc,
                                "setup_time": now_utc,
                                "confirmed": True,
                            }
                            pending = self._kalshi_pending_signals[asset]
                    else:
                        signal = None
                else:
                    signal = self.kalshi_predictor.score(df_15m, market_data=market_data, df_1h=df_1h)

                if signal is None:
                    # Signal disappeared with new leading data — skip, signal survives for next eval
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"] if pending else "--",
                        "confidence": 0,
                        "reason": f"{state.lower()}: score returned None with fresh OB/flow, signal survives",
                        "ob": ob_imb, "flow": net_flow, "state": state,
                    })
                    continue

                # V3: check recommended_side; V1/V2: check confidence threshold
                if self.kalshi_predictor_version == "v3":

                    if isinstance(signal, KalshiV3Signal) and signal.recommended_side != "SKIP":
                        pending["confirmed"] = True
                        # Apply per-asset M0 confidence threshold.
                        asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, 60)
                        prob_pct = int(signal.probability * 100)
                        meets_m0 = prob_pct >= asset_threshold or prob_pct <= (100 - asset_threshold)

                        if state == "CONFIRMED" and meets_m0:
                            dir_conf = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)
                            # Only log SIGNAL once per asset per window
                            signal_log_key = f"_signal_logged_{asset}_{current_window_start}"
                            if not self._kalshi_cached_dataframes.get(signal_log_key):
                                print(colored(
                                    f"  [SIGNAL] {asset} {signal.recommended_side} "
                                    f"M0={dir_conf}% → actionable",
                                    "cyan",
                                ))
                                self._kalshi_cached_dataframes[signal_log_key] = True
                            actionable_signals.append({
                                "symbol": symbol, "series_ticker": series_ticker,
                                "signal": signal, "market_data": market_data,
                                "state": state,
                                "signal_ticker": signal_ticker,  # exact market from _get_kalshi_strike
                            })
                        elif state == "CONFIRMED":
                            # Explicitly log why no entry happened this window.
                            no_sig_key = f"_no_signal_logged_{asset}_{current_window_start}"
                            if not self._kalshi_cached_dataframes.get(no_sig_key):
                                dir_conf = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)
                                print(colored(
                                    f"  [NO SIGNAL] {asset} {signal.recommended_side} "
                                    f"M0={dir_conf}% < {asset_threshold}% threshold",
                                    "dark_grey",
                                ))
                                self._kalshi_cached_dataframes[no_sig_key] = True
                        reason_suffix = ""
                        if not meets_m0:
                            reason_suffix = f" (M0 below {asset_threshold}%)"
                        elif state == "CONFIRMED":
                            reason_suffix = " -> BETTING (M0)"
                        elif state == "MONITORING":
                            reason_suffix = " (monitoring)"
                        else:
                            reason_suffix = " (waiting)"

                        # Display confidence in the RECOMMENDED direction, not raw UP probability
                        display_conf = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)

                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": signal.recommended_side,
                            "confidence": display_conf,
                            "knn_score": display_conf, "knn_thresh": asset_threshold,
                            "tbl_score": 0, "tbl_thresh": 0,
                            "btc_score": 0, "btc_thresh": 0,
                            "reason": f"{state.lower()} [M0]: prob={signal.probability:.2f} side={signal.recommended_side}" + reason_suffix,
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })
                    else:
                        # SKIP — keep dashboard visibility without extra model calls.
                        raw_pct = int(signal.probability * 100) if hasattr(signal, 'probability') else 50
                        # Show how close to threshold (e.g., 52% → "52%")
                        skip_m0 = raw_pct if raw_pct >= 50 else (100 - raw_pct)

                        if state == "CONFIRMED":
                            no_sig_key = f"_no_signal_logged_{asset}_{current_window_start}"
                            if not self._kalshi_cached_dataframes.get(no_sig_key):
                                print(colored(
                                    f"  [NO SIGNAL] {asset} SKIP prob={signal.probability:.2f}",
                                    "dark_grey",
                                ))
                                self._kalshi_cached_dataframes[no_sig_key] = True

                        # BTC score (neutral since no direction chosen)
                        skip_btc = self.compute_btc_score("YES")  # arbitrary side, shows magnitude

                        asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, 60)

                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": "--",
                            "confidence": raw_pct,
                            "knn_score": skip_m0, "knn_thresh": asset_threshold,
                            "tbl_score": 0, "tbl_thresh": 0,
                            "btc_score": skip_btc, "btc_thresh": 0,
                            "reason": f"{state.lower()} [M0]: SKIP (prob={signal.probability:.2f})",
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })
                else:
                    # V1/V2: update pending with fresh confidence
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
                            "reason": f"{state.lower()}: conf {signal.confidence} >= {asset_threshold} (fresh OB/flow)",
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })
                    else:
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": signal.direction,
                            "confidence": signal.confidence,
                            "reason": f"{state.lower()}: conf {signal.confidence} < {asset_threshold}",
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })

        # Execute top actionable signals by confidence
        def _signal_score(x):
            sig = x["signal"]
            if hasattr(sig, 'probability'):
                return sig.probability
            return getattr(sig, 'confidence', 0) / 100
        actionable_signals.sort(key=_signal_score, reverse=True)

        # Count only ACTUAL fills toward max_bets, not price-blocked attempts
        actual_fills_this_eval = 0
        for vs in actionable_signals:
            if self._cli_max_bets > 0:
                max_bets = self._cli_max_bets
            else:
                max_bets = MAX_CONCURRENT_KALSHI_BETS
            if len(self._active_kalshi_bets) + actual_fills_this_eval >= max_bets:
                break
            entry_exec_started = time.perf_counter()
            result = self._kalshi_execute_bet(
                vs["symbol"], vs["series_ticker"], vs["signal"], vs["market_data"],
                signal_ticker=vs.get("signal_ticker"),
            )
            entry_exec_ms = (time.perf_counter() - entry_exec_started) * 1000
            asset = vs["symbol"].split("/")[0]
            reason = result.get("reason", "") if result else ""
            has_fill = "filled=" in reason and "filled=0" not in reason
            is_dry_bet = "would bet" in reason
            was_skipped = "skipping" in reason or "price too high" in reason

            # Mark attempted on any real order placement (filled or not) to prevent
            # check_price_watches/eval from re-firing the same signal after an
            # unfilled-and-cancelled order. Does NOT include pre-order skips —
            # those stay eligible for price_watch retry.
            if not was_skipped and asset in self._kalshi_pending_signals:
                self._kalshi_pending_signals[asset]["bet_attempted"] = True

            if (has_fill or is_dry_bet) and not was_skipped:
                actual_fills_this_eval += 1
                if asset in self._kalshi_pending_signals:
                    self._kalshi_pending_signals[asset]["bet_placed"] = True
                    sig_obj = vs["signal"]
                    if hasattr(sig_obj, "probability"):
                        p = int(sig_obj.probability * 100)
                        d = sig_obj.recommended_side
                        self._kalshi_pending_signals[asset]["bet_knn_score"] = p if d == "YES" else (100 - p)
                        self._kalshi_pending_signals[asset]["bet_btc_score"] = self.compute_btc_score(d)
            elif was_skipped:
                # Price too high — try once at min 0, once at min 1, then stop
                if asset in self._kalshi_pending_signals:
                    pending_sig = self._kalshi_pending_signals[asset]
                    last_skip_min = pending_sig.get("_last_skip_min", -1)
                    current_min = datetime.now(timezone.utc).minute % 15
                    if last_skip_min != current_min:
                        # First skip this minute — record it.
                        pending_sig["_last_skip_min"] = current_min
                    # Minute 0 skip: keep eligible for one retry at minute 1.
                    # Minute 1+ skip: lock this window as attempted, but never mark as placed.
                    if current_min >= 1:
                        pending_sig["bet_attempted"] = True
                        pending_sig["skip_reason"] = "entry skipped (ask > 60c at final entry minute)"
                    else:
                        pending_sig["skip_reason"] = "entry skipped (ask > 60c, retrying at minute 1)"

            if entry_exec_ms > 1200:
                print(colored(
                    f"  [PERF] {asset} entry execution took {entry_exec_ms:.0f}ms",
                    "yellow",
                ))

        self.kalshi_predictions = predictions
        eval_ms = (time.perf_counter() - eval_started) * 1000
        # PERF logging relevance filter:
        # - always log severe slow evals
        # - log moderate slowness only when there was actual entry activity
        # - emit at most one idle sample per 15m window (keeps observability without spam)
        severe_slow = eval_ms > 6000
        moderate_slow = eval_ms > 1500
        has_entry_activity = (len(actionable_signals) > 0) or (actual_fills_this_eval > 0)
        now_perf_utc = datetime.now(timezone.utc)
        perf_window = now_perf_utc.replace(
            minute=now_perf_utc.minute - (now_perf_utc.minute % 15), second=0, microsecond=0
        )
        perf_window_token = perf_window.strftime("%Y-%m-%dT%H:%MZ")
        idle_sample = False
        if moderate_slow and not has_entry_activity and perf_window_token != self._perf_idle_window_token:
            idle_sample = True
            self._perf_idle_window_token = perf_window_token

        if severe_slow or (moderate_slow and has_entry_activity) or idle_sample:
            suffix = " [idle sample]" if idle_sample else ""
            print(colored(
                f"  [PERF] _kalshi_eval took {eval_ms:.0f}ms "
                f"(signals={len(actionable_signals)}, preds={len(predictions)}){suffix}",
                "yellow",
            ))

    # ------------------------------------------------------------------
    # Bet execution
    # ------------------------------------------------------------------

    def _kalshi_execute_bet(self, symbol: str, series_ticker: str, signal, market_data: dict | None,
                            signal_ticker: str | None = None) -> dict:
        """Execute a single Kalshi bet. Returns a pred dict with outcome details.

        signal_ticker: exact market ticker from signal generation (_get_kalshi_strike).
        If provided, uses this ticker directly instead of re-discovering the market.
        """
        asset = symbol.split("/")[0]
        chop_metrics = self._compute_chop_metrics(asset)
        ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
        net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

        # Detect V3 signal type and extract side/confidence accordingly

        if isinstance(signal, KalshiV3Signal):
            direction_label = signal.recommended_side  # "YES" or "NO"
            side = "yes" if signal.recommended_side == "YES" else "no"
            raw_prob = int(signal.probability * 100)
            # Show confidence in the bet direction, not raw UP probability
            conf_display = raw_prob if side == "yes" else (100 - raw_prob)
            MAX_ENTRY_CENTS = min(85, signal.max_price_cents) if signal.max_price_cents > 0 else 85
        else:
            direction_label = signal.direction
            side = "yes" if signal.direction == "UP" else "no"
            conf_display = signal.confidence
            MAX_ENTRY_CENTS = 50

        pred = {
            "symbol": symbol, "asset": asset,
            "direction": direction_label,
            "confidence": conf_display,
            "ob": ob_imb, "flow": net_flow, "reason": "",
        }

        RISK_PER_BET_PCT = KALSHI_RISK_PER_BET_PCT

        pred["reason"] = f"would bet {side.upper()} (conf={conf_display})"

        if self.dry_run and not self.demo:
            # Record dry-run bet with ACTUAL contract price from Kalshi
            strike = 0
            settle_time = None
            contract_price = 0
            ticker = ""
            price_lookup_error = None

            if isinstance(signal, KalshiV3Signal):
                strike = signal.strike_price
                pending = self._kalshi_pending_signals.get(asset, {})
                settle_time = pending.get("close_time")

            # Query Kalshi for actual contract price — use signal's ticker
            self._init_kalshi_client()
            if self.kalshi_client and strike:
                try:
                    if signal_ticker:
                        ticker = signal_ticker
                        m = self.kalshi_client.get_market(ticker)
                        m = m.get("market", m)
                    else:
                        series = self.KALSHI_PAIRS.get(symbol, "")
                        markets = self.kalshi_client.get_markets(series_ticker=series, status="open")
                        m = markets[0] if markets else None
                        ticker = m.get("ticker", "") if m else ""
                    if m:
                        # Get the ask price for our side
                        if side == "yes":
                            contract_price = int(float(m.get("yes_ask_dollars", 0)) * 100)
                        else:
                            contract_price = int(float(m.get("no_ask_dollars", 0)) * 100)

                        # Also try orderbook for better price
                        try:
                            book = self.kalshi_client.get_orderbook(ticker)
                            ob_data = book.get("orderbook", {})
                            asks = ob_data.get(side, [])
                            if asks and len(asks) > 0:
                                best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
                                contract_price = int(best_ask)
                        except Exception as e:
                            price_lookup_error = f"orderbook lookup failed: {e}"
                except Exception as e:
                    price_lookup_error = str(e)

            # Log dry-run market selection (same schema as live, plus chop_metrics)
            self._log_trade_debug(
                asset=asset, action="MARKET_SELECT",
                details={
                    "ticker": ticker,
                    "market_strike": strike,
                    "model_strike": strike,
                    "close_time": settle_time.isoformat() if settle_time else "?",
                    "side": side,
                    "direction": direction_label,
                    "n_markets": 1,
                    "demo": self.demo,
                    "dry_run": True,
                    **chop_metrics,
                }
            )

            # Strict no-fallback: never simulate fills with missing/invalid ask price.
            if contract_price <= 0:
                err_hint = f" ({price_lookup_error})" if price_lookup_error else ""
                print(colored(
                    f"  [KALSHI DRY] {asset} {side.upper()} missing ask price{err_hint} — skipping",
                    "yellow",
                ))
                pred["reason"] = "missing ask price"
                return pred

            # If ask > max, DON'T fill — match live behavior exactly
            # Live would place a resting order that may never fill
            if contract_price > MAX_ENTRY_CENTS:
                self._log_price_skip_throttled(asset, side, contract_price, MAX_ENTRY_CENTS)
                pred["reason"] = f"price too high ({contract_price}c > {MAX_ENTRY_CENTS}c)"
                return pred

            # Ask <= MAX_ENTRY_CENTS — fill at the ask price (same as live fill)
            # Position sizing — CLI override or default 5%
            size_pct = self._cli_max_size_pct if self._cli_max_size_pct > 0 else 0.05
            risk_budget = int(self._dry_balance_cents * size_pct)
            count = max(1, risk_budget // contract_price) if contract_price > 0 else 1
            cost = count * contract_price
            potential_profit = count * (100 - contract_price)

            if strike and settle_time:
                self._pending_bets.append({
                    "asset": asset,
                    "symbol": symbol,
                    "side": side,
                    "direction": direction_label,
                    "strike": strike,
                    "confidence": conf_display,
                    "bet_time": datetime.now(timezone.utc),
                    "settle_time": settle_time,
                    "contract_price": contract_price,
                    "count": count,
                    "cost_cents": cost,
                    "potential_profit_cents": potential_profit,
                    "ticker": ticker,
                    "chop_metrics": chop_metrics,
                })
                self._session_bets_placed += 1

            # Track active bet for concurrency limit (same as live path)
            bet_key = f"dry_{asset}_{int(time.time())}"
            self._active_kalshi_bets[bet_key] = time.time()

            print(colored(
                f"  [KALSHI DRY] {asset} {direction_label} "
                f"prob={conf_display}% | "
                f"{side.upper()} x{count} @ {contract_price}c "
                f"risk=${cost/100:.2f} profit=${potential_profit/100:.2f} "
                f"strike=${strike:,.2f}",
                "magenta",
            ))
            return pred

        # Live: place the bet
        self._init_kalshi_client()
        if self.kalshi_client is None:
            pred["reason"] = "client init failed"
            return pred
        try:
            # Kalshi is source of truth for balance
            balance_resp = self.kalshi_client.get_balance()
            balance_cents = balance_resp.get("balance", 0)

            # Use the SAME market ticker that the signal was generated for.
            # Previously re-discovered markets here, which could pick a DIFFERENT
            # market than the signal predicted — causing systematic losses.
            market_count = 1
            if signal_ticker:
                ticker = signal_ticker
                market = self.kalshi_client.get_market(ticker)
                market = market.get("market", market)
            else:
                # Fallback: discover market (same logic as _get_kalshi_strike)
                all_markets = self.kalshi_client.get_markets(
                    series_ticker=series_ticker, status="open")
                if not all_markets:
                    pred["reason"] = f"no {series_ticker} markets found"
                    print(colored(f"  [KALSHI] No markets for {series_ticker}", "yellow"))
                    return pred
                now_utc_exec = datetime.now(timezone.utc)
                future = [m for m in all_markets
                          if m.get("close_time", "") and
                          datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) > now_utc_exec]
                if not future:
                    pred["reason"] = f"no future {series_ticker} markets"
                    return pred
                future.sort(key=lambda m: m.get("close_time", "9999"))
                market = future[0]
                ticker = market.get("ticker", "")
                market_count = len(all_markets)

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
                    "n_markets": market_count,
                    "demo": self.demo,
                    **chop_metrics,
                }
            )

            # Get current ask price — bid AT the ask for guaranteed fill
            if side == "yes":
                ask_raw = market.get("yes_ask_dollars") or market.get("yes_ask")
            else:
                ask_raw = market.get("no_ask_dollars") or market.get("no_ask")

            if ask_raw:
                fill_price = int(float(ask_raw) * 100) if float(ask_raw) < 1.5 else int(float(ask_raw))
            else:
                pred["reason"] = "missing ask price"
                return pred

            fill_price = max(1, min(99, fill_price))

            # If ask > max, SKIP — don't place resting order below the ask.
            # Adverse selection: fills on resting orders below market ask are
            # systematically losers (price dropped TO our limit = market turned
            # against us). Dry-run skips these, so must live.
            if fill_price > MAX_ENTRY_CENTS:
                self._log_price_skip_throttled(asset, side, fill_price, MAX_ENTRY_CENTS)
                pred["reason"] = f"ask too high ({fill_price}c > {MAX_ENTRY_CENTS}c)"
                return pred

            # Position sizing — CLI override or default 5%
            size_pct = self._cli_max_size_pct if self._cli_max_size_pct > 0 else 0.05
            risk_budget_cents = int(balance_cents * size_pct)
            count = max(1, risk_budget_cents // fill_price) if fill_price > 0 else 1
            potential_profit = count * (100 - fill_price)
            potential_loss = count * fill_price
            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0

            # Verify against actual Kalshi balance (exchange is source of truth)
            cost_cents = count * fill_price
            if cost_cents > balance_cents:
                count = balance_cents // fill_price
                if count < 1:
                    # Can't afford even 1 contract — buy 1 anyway if balance > 0
                    if balance_cents >= fill_price:
                        count = 1
                    else:
                        pred["reason"] = f"insufficient Kalshi balance (${balance_cents/100:.2f})"
                        return pred
                cost_cents = count * fill_price
                potential_profit = count * (100 - fill_price)
                rr_ratio = potential_profit / cost_cents if cost_cents > 0 else 0

            result = self.kalshi_client.place_order(
                ticker=ticker,
                side=side,
                count=count,
                price_cents=fill_price,
                order_type="limit",
            )
            order = result.get("order", {})
            order_id = order.get("order_id", "?")
            fill_count = order.get("fill_count_fp", "0")
            order_status = order.get("status", "?")

            # Log order placement details
            self._log_trade_debug(
                asset=asset, action="ORDER_PLACED",
                details={
                    "ticker": ticker,
                    "side": side,
                    "direction": direction_label,
                    "count": count,
                    "fill_price": fill_price,
                    "fill_count": fill_count,
                    "status": order_status,
                    "order_id": order_id,
                    "model_prob": signal.probability if isinstance(signal, KalshiV3Signal) else 0,
                    "model_dist": signal.distance_atr if isinstance(signal, KalshiV3Signal) else 0,
                    "model_strike": signal.strike_price if isinstance(signal, KalshiV3Signal) else 0,
                    "market_yes_ask": market.get("yes_ask_dollars") or market.get("yes_ask"),
                    "market_no_ask": market.get("no_ask_dollars") or market.get("no_ask"),
                    "demo": self.demo,
                }
            )

            # Aggressive fill loop — if resting, reprice every 2s until filled or ask > MAX
            if float(fill_count) < count and order_status == "resting":
                import time as _t
                MAX_REPRICE_ATTEMPTS = 2  # 2 retries × 1s = ~2s total
                for reprice_attempt in range(MAX_REPRICE_ATTEMPTS):
                    _t.sleep(1)

                    # Check if filled since last attempt
                    try:
                        cur_status = self.kalshi_client.get_order_status(order_id)
                        cur_filled = float(cur_status.get("fill_count_fp", 0))
                        cur_state = cur_status.get("status", "")
                        if cur_filled >= count or cur_state in ("executed", "filled", "closed"):
                            fill_count = str(int(cur_filled)) if cur_filled > 0 else str(count)
                            order_status = cur_state or "executed"
                            print(colored(
                                f"  [KALSHI FILL] {asset} {direction_label} filled on check "
                                f"(attempt {reprice_attempt + 1})",
                                "green"))
                            break
                    except Exception:
                        pass

                    # Get fresh ask price
                    try:
                        mkt_resp = self.kalshi_client.get_market(ticker)
                        mkt_data = mkt_resp.get("market", mkt_resp)
                        if side == "yes":
                            new_ask_raw = mkt_data.get("yes_ask_dollars") or mkt_data.get("yes_ask")
                        else:
                            new_ask_raw = mkt_data.get("no_ask_dollars") or mkt_data.get("no_ask")
                        if new_ask_raw:
                            new_ask = int(float(new_ask_raw) * 100) if float(new_ask_raw) < 1.5 else int(float(new_ask_raw))
                        else:
                            continue  # no ask available, wait
                    except Exception:
                        continue

                    new_ask = max(1, min(99, new_ask))

                    if new_ask > MAX_ENTRY_CENTS:
                        # Cancel the resting order — no resting below market
                        try:
                            self.kalshi_client.cancel_order_safe(order_id)
                        except Exception:
                            pass
                        order_status = "cancelled"
                        fill_count = "0"
                        print(colored(
                            f"  [KALSHI REPRICE] {asset} {direction_label} ask now {new_ask}c > "
                            f"{MAX_ENTRY_CENTS}c — cancelled order",
                            "yellow"))
                        break

                    if new_ask == fill_price:
                        continue  # same price, no point repricing

                    # Cancel old order and place at new ask
                    try:
                        cancel_result = self.kalshi_client.cancel_order_safe(order_id)
                        if cancel_result.get("status") == "filled":
                            # Filled between our check and cancel
                            fill_count = str(count)
                            order_status = "executed"
                            print(colored(
                                f"  [KALSHI FILL] {asset} {direction_label} filled on cancel "
                                f"(attempt {reprice_attempt + 1})",
                                "green"))
                            break
                    except Exception as e:
                        print(colored(f"  [REPRICE] Cancel failed: {e}", "red"))
                        break

                    # Place new order at current ask
                    fill_price = new_ask
                    cost_cents = count * fill_price
                    potential_profit = count * (100 - fill_price)
                    rr_ratio = potential_profit / cost_cents if cost_cents > 0 else 0
                    try:
                        result = self.kalshi_client.place_order(
                            ticker=ticker, side=side, count=count,
                            price_cents=fill_price, order_type="limit",
                        )
                        order = result.get("order", {})
                        order_id = order.get("order_id", order_id)
                        fill_count = order.get("fill_count_fp", "0")
                        order_status = order.get("status", "?")
                        print(colored(
                            f"  [KALSHI REPRICE] {asset} {direction_label} repriced to {fill_price}c "
                            f"(attempt {reprice_attempt + 1}) status={order_status}",
                            "cyan"))
                        if float(fill_count) >= count or order_status in ("executed", "filled"):
                            break
                    except Exception as e:
                        print(colored(f"  [REPRICE] Re-place failed: {e}", "red"))
                        break

            # If still unfilled after reprice loop, cancel — no resting orders
            actual_filled = float(fill_count) if fill_count else 0
            if actual_filled < count and order_status not in ("executed", "filled", "cancelled"):
                try:
                    cancel_result = self.kalshi_client.cancel_order_safe(order_id)
                    if cancel_result.get("status") == "filled":
                        fill_count = str(count)
                        order_status = "executed"
                    else:
                        order_status = "cancelled"
                        fill_count = "0"
                        print(colored(
                            f"  [KALSHI CANCEL] {asset} {direction_label} unfilled after reprice — cancelled",
                            "yellow"))
                except Exception:
                    pass

            pred["reason"] = (
                f"placed {side.upper()} x{count} @ {fill_price}c "
                f"risk=${cost_cents/100:.2f} profit=${potential_profit/100:.2f} R:R={rr_ratio:.1f}:1 "
                f"filled={fill_count} status={order_status} (#{order_id})"
            )
            print(colored(
                f"  [KALSHI BET] {asset} {direction_label} "
                f"conf={conf_display} | {side.upper()} x{count} @ {fill_price}c "
                f"| risk=${cost_cents/100:.2f} profit=${potential_profit/100:.2f} R:R={rr_ratio:.1f}:1 "
                f"| filled={fill_count} status={order_status}",
                "magenta",
            ))
            self._active_kalshi_bets[ticker] = time.time()
            self._session_bets_placed += 1

            # Track for settlement — only if order actually filled
            actual_filled = float(fill_count) if fill_count else 0  # recompute after cancel logic
            if isinstance(signal, KalshiV3Signal) and actual_filled > 0:
                pending = self._kalshi_pending_signals.get(asset, {})
                settle_time = pending.get("close_time")
                if settle_time:
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
            # No resting order tracking — unfilled orders are always cancelled

        except Exception as e:
            pred["reason"] = f"order failed: {e}"
            print(colored(f"  [KALSHI ERR] {asset}: {e}", "red"))

        return pred

    # ------------------------------------------------------------------
    # Price monitoring for blocked signals
    # ------------------------------------------------------------------

    def check_price_watches(self):
        """Check price-blocked signals for entry opportunity. Called every 5s from dashboard."""
        now_utc = datetime.now(timezone.utc)
        min_in = now_utc.minute % 15
        if min_in < 3 or min_in > 10:
            return  # only monitor during entry window

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        # Sort by confidence (highest first) so best signals get priority
        watches = [(a, p) for a, p in self._kalshi_pending_signals.items()
                   if p.get("price_watch") and not p.get("bet_placed") and not p.get("bet_attempted")]
        watches.sort(key=lambda x: x[1].get("probability", 0), reverse=True)

        for asset, pending in watches:

            series = pending.get("watch_series", "")
            side = pending.get("watch_side", "")
            if not series or not side:
                continue

            # Skip if already holding a position on this asset from any window
            if any(b.get("asset") == asset and not b.get("result") and b.get("count", 0) > 0
                   for b in self._pending_bets):
                continue

            try:
                markets = self.kalshi_client.get_markets(series_ticker=series, status="open")
                if not markets:
                    continue
                m = max(markets, key=lambda x: x.get("close_time", ""))

                if side == "YES":
                    ask = float(m.get("yes_ask_dollars", 0)) * 100
                else:
                    ask = float(m.get("no_ask_dollars", 0)) * 100
                ask = int(ask)

                symbol = f"{asset}/USDT"

                if ask <= MAX_BET_PRICE:
                    # Check max positions before entering
                    if self._cli_max_bets > 0:
                        max_bets = self._cli_max_bets
                    else:
                        max_bets = MAX_CONCURRENT_KALSHI_BETS
                    active_count = sum(1 for p in self._kalshi_pending_signals.values()
                                       if p.get("bet_placed") and not p.get("result"))
                    if active_count >= max_bets:
                        continue  # at capacity, skip this one

                    # Price dropped! Execute the bet
                    print(colored(
                        f"  [PRICE DROP] {asset} {side} ask={ask}c ≤ {MAX_BET_PRICE}c — entering!",
                        "green", attrs=["bold"],
                    ))

                    # Build a minimal signal for execution
                    strike = pending.get("strike_price", 0)
                    prob = pending.get("probability", 0.5)
                    from strategy.strategies.kalshi_predictor_v3 import KalshiV3Signal
                    sig = KalshiV3Signal(
                        asset=asset, probability=prob,
                        recommended_side=side,
                        max_price_cents=MAX_BET_PRICE,
                        distance_atr=pending.get("distance_atr", 0),
                        base_prob=prob, adjustments={"mode": "knn_early_entry"},
                        current_price=0, strike_price=strike,
                        minutes_remaining=15 - min_in,
                    )
                    result = self._kalshi_execute_bet(symbol, series, sig, None)
                    reason = result.get("reason", "") if result else ""
                    has_fill = "filled=" in reason and "filled=0" not in reason
                    is_dry = "would bet" in reason
                    was_skipped = "skipping" in reason or "price too high" in reason
                    # Lock this ticker as attempted — unfilled+cancelled orders
                    # otherwise re-fire on the next 5s tick and stack positions.
                    if not was_skipped:
                        pending["bet_attempted"] = True
                        pending["price_watch"] = False
                    if has_fill or is_dry:
                        pending["bet_placed"] = True
                        pending["price_watch"] = False
                        p = int(prob * 100)
                        d = side
                        pending["bet_knn_score"] = p if d == "YES" else (100 - p)
                        pending["bet_btc_score"] = self.compute_btc_score(d)
                else:
                    if min_in % 2 == 0:  # don't spam every 5s
                        print(colored(
                            f"  [WATCHING] {asset} {side} ask={ask}c > {MAX_BET_PRICE}c (min {min_in}/10)",
                            "dark_grey",
                        ))

            except Exception:
                pass

    # ------------------------------------------------------------------
    # Resting order management
    # ------------------------------------------------------------------

    def _cancel_all_resting_orders(self, reason: str = "shutdown"):
        """Cancel ALL resting orders on Kalshi. Called on shutdown and window boundary."""
        # Dry-run does not place real orders on Kalshi.
        if self.dry_run and not self.demo:
            return

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        cancelled = 0
        known_ids = set()

        # Cancel orders we're tracking in memory
        for order in list(self._resting_orders):
            order_id = order.get("order_id", "")
            asset = order.get("asset", "?")
            if not order_id:
                continue
            known_ids.add(order_id)
            try:
                result = self.kalshi_client.cancel_order_safe(order_id)
                if result.get("status") == "filled":
                    print(colored(
                        f"  [CANCEL] {asset} order {order_id} — already filled", "green"))
                else:
                    print(colored(
                        f"  [CANCEL] {asset} order {order_id} — cancelled ({reason})", "yellow"))
                    cancelled += 1
            except Exception as e:
                print(colored(f"  [CANCEL ERR] {asset}: {e}", "red"))

        # Also query exchange for any resting orders we might not be tracking
        try:
            exchange_resting = self.kalshi_client.get_orders(status="resting")
            for order in exchange_resting:
                oid = order.get("order_id", "")
                if oid and oid not in known_ids:
                    ticker = order.get("ticker", "?")
                    try:
                        self.kalshi_client.cancel_order_safe(oid)
                        print(colored(
                            f"  [CANCEL] {ticker} order {oid} — cancelled ({reason}, untracked)",
                            "yellow"))
                        cancelled += 1
                    except Exception:
                        pass
        except Exception:
            pass

        self._resting_orders.clear()
        if cancelled > 0:
            print(colored(f"  [CANCEL] Cancelled {cancelled} resting orders ({reason})", "yellow"))

    def _check_resting_orders(self):
        """Check resting orders for fills, cancel at minute 10."""
        if not self._resting_orders:
            return

        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        # Check WebSocket fill events (instant detection trigger)
        # WS can send duplicate events — use as a TRIGGER, then verify actual count via API
        if self.kalshi_ws:
            ws_fills = self.kalshi_ws.get_pending_fills()
            for fill in ws_fills:
                fill_ticker = fill.get("ticker", "")
                for order in self._resting_orders:
                    if order.get("ticker") == fill_ticker or order.get("order_id", "") in str(fill):
                        if order.get("_ws_detected"):
                            continue  # already processed this order's fill
                        asset = order.get("asset", "?")
                        order_id = order.get("order_id", "")

                        # Query actual fill count from API (source of truth)
                        actual_count = 0
                        if order_id and self.kalshi_client:
                            try:
                                status = self.kalshi_client.get_order_status(order_id)
                                actual_count = int(float(status.get("fill_count_fp", 0)))
                            except Exception:
                                actual_count = max(int(fill.get("count", 1)), 1)
                        else:
                            actual_count = max(int(fill.get("count", 1)), 1)

                        if actual_count > 0:
                            order["count"] = actual_count
                            order["needs_fill_check"] = False
                            order["_ws_detected"] = True
                            if asset in self._kalshi_pending_signals:
                                self._kalshi_pending_signals[asset]["bet_placed"] = True
                            print(colored(
                                f"  [WS FILL] {asset} {order['side'].upper()} "
                                f"x{actual_count} @ {order['fill_price']}c — filled!",
                                "green",
                            ))

        still_resting = []
        for order in self._resting_orders:
            order_id = order.get("order_id", "")
            asset = order.get("asset", "?")
            if not order_id or order_id == "?":
                continue

            # Already filled via WebSocket?
            if order.get("count", 0) > 0 and not order.get("needs_fill_check", True):
                continue  # already processed above

            try:
                status = self.kalshi_client.get_order_status(order_id)
                filled = float(status.get("fill_count_fp", 0))
                order_status = status.get("status", "")

                # CHECK FILLS — accept fills at any time
                # Kalshi statuses: resting, canceled, executed, pending
                if filled > 0 or order_status in ("executed", "filled", "closed"):
                    fill_count = max(int(filled), 1)
                    order["count"] = fill_count
                    order["needs_fill_check"] = False
                    if asset in self._kalshi_pending_signals:
                        self._kalshi_pending_signals[asset]["bet_placed"] = True
                    print(colored(
                        f"  [RESTING FILL] {asset} {order['side'].upper()} "
                        f"x{fill_count} @ {order['fill_price']}c — filled! (status={order_status})",
                        "green",
                    ))
                    continue  # don't add back to resting list

                # THEN cancel if past deadline and still unfilled
                order_expired = False
                settle_time = order.get("settle_time")
                if settle_time and now_utc > settle_time:
                    order_expired = True

                if (minute_in_window >= 10 or order_expired) and order_status != "executed":
                    reason = "previous window" if order_expired else f"min {minute_in_window}"
                    try:
                        result = self.kalshi_client.cancel_order_safe(order_id)
                        # cancel_order_safe returns {"status": "filled"} on 404
                        if result.get("status") == "filled":
                            # Actually filled! Don't cancel.
                            order["count"] = int(order.get("count", 0)) or 1
                            order["needs_fill_check"] = False
                            if asset in self._kalshi_pending_signals:
                                self._kalshi_pending_signals[asset]["bet_placed"] = True
                            print(colored(
                                f"  [RESTING FILL] {asset} {order['side'].upper()} "
                                f"@ {order['fill_price']}c — filled (detected on cancel attempt)!",
                                "green",
                            ))
                            continue
                        print(colored(
                            f"  [RESTING CANCEL] {asset} {order['side'].upper()} "
                            f"@ {order['fill_price']}c — cancelled ({reason})",
                            "yellow",
                        ))
                        self._pending_bets = [
                            b for b in self._pending_bets
                            if b.get("order_id") != order_id
                        ]
                    except Exception as e:
                        print(colored(f"  [CANCEL ERR] {asset}: {e}", "red"))
                    continue  # don't add back

                # Still resting — query the ORDER's status for current market ask
                current_ask = "?"
                try:
                    # Use the order status we already fetched — check the order's own ticker
                    order_ticker = status.get("ticker") or order.get("ticker", "")
                    if order_ticker:
                        mkt = self.kalshi_client.get_market(order_ticker)
                        market_data = mkt.get("market", mkt)
                        side = order.get("side", "yes")
                        if side == "yes":
                            raw = market_data.get("yes_ask") or market_data.get("yes_ask_dollars")
                        else:
                            raw = market_data.get("no_ask") or market_data.get("no_ask_dollars")
                        if raw:
                            ask_val = float(raw) * 100 if float(raw) < 1.5 else float(raw)
                            current_ask = f"{int(ask_val)}c"
                except Exception:
                    pass

                print(colored(
                    f"  [RESTING] {asset} {order['side'].upper()} "
                    f"@ {order['fill_price']}c — still waiting (ask={current_ask}, min {minute_in_window})",
                    "dark_grey"))
                still_resting.append(order)

            except Exception:
                still_resting.append(order)  # keep tracking on error

        self._resting_orders = still_resting

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _check_model_freshness(self):
        """Check if per-asset M0 models are older than 7 days. Retrain if stale."""
        from pathlib import Path
        import os

        max_age_days = 7
        per_asset_paths = [
            Path("models/m0_btc.pkl"),
            Path("models/m0_eth.pkl"),
            Path("models/m0_sol.pkl"),
            Path("models/m0_xrp.pkl"),
        ]
        legacy_path = Path("models/knn_kalshi.pkl")

        if all(p.exists() for p in per_asset_paths):
            # Use oldest file in the bundle as freshness anchor.
            age_seconds = max(time.time() - os.path.getmtime(p) for p in per_asset_paths)
            model_label = "M0 per-asset bundle"
        elif legacy_path.exists():
            age_seconds = time.time() - os.path.getmtime(legacy_path)
            model_label = "M0 unified model (legacy)"
        else:
            print(colored("  [MODEL] M0 models not found — training...", "yellow"))
            self._retrain_model()
            return

        age_days = age_seconds / 86400

        if age_days > max_age_days:
            print(colored(
                f"  [MODEL] {model_label} is {age_days:.1f} days old (max {max_age_days}) — retraining...",
                "yellow",
            ))
            self._retrain_model()
        else:
            print(colored(
                f"  [MODEL] {model_label} is {age_days:.1f} days old — OK",
                "green",
            ))

    def _retrain_model(self):
        """Retrain per-asset models with Kalshi settlement labels + multi-exchange price.

        Uses scripts/retrain_kalshi_labels.py which:
        - Labels: Kalshi settled result (yes/no) — ground truth
        - Strike: Kalshi floor_strike — actual settlement strike
        - Price: Coinbase + Bitstamp 5m average — BRTI proxy
        - Per-asset models with cross-asset confluence + regime detection
        - 33 M0 features (no rsi_15m), 35 M10 features (with rsi_15m + 5m intra-window)
        - Trained on 12 months synthetic data (~160K samples)
        """
        import subprocess

        print(colored("  [MODEL] Starting Kalshi-labeled retrain (~15 min)...", "yellow"))
        print(colored("  [MODEL] Labels: Kalshi settlements | Price: Coinbase+Bitstamp avg", "yellow"))

        try:
            result = subprocess.run(
                ["./venv/bin/python", "scripts/retrain_kalshi_labels.py",
                 "--output", "models/knn_kalshi.pkl"],
                capture_output=True, text=True, timeout=1200,
            )

            # Print key output lines
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line and any(k in line for k in [
                    "Train WR", "Feature", "distance_from_strike",
                    "Best config", "Saved", "samples", "Total:",
                ]):
                    print(f"  [MODEL] {line}")

            if result.returncode == 0:
                print(colored("  [MODEL] Strike-relative retrain complete!", "green"))
                # Reload — re-init predictor to pick up new model format
                from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
                self.kalshi_predictor = KalshiPredictorV3()
                mt = self.kalshi_predictor._model_type or "unknown"
                print(colored(f"  [MODEL] Predictor reloaded (type: {mt})", "green"))
            else:
                print(colored(f"  [MODEL] Retrain failed: {result.stderr[-300:]}", "red"))

        except subprocess.TimeoutExpired:
            print(colored("  [MODEL] Retrain timed out (>15 min)", "red"))
        except Exception as e:
            print(colored(f"  [MODEL] Retrain error: {e}", "red"))

    def startup(self):
        """Fetch initial data, check model freshness, and run first eval."""
        if not self._acquire_instance_lock():
            return False

        mode = "DEMO" if self.demo else ("DRY-RUN" if self.dry_run else "LIVE")
        if self.kalshi_predictor_version == 'v3':
            label = 'V3 Strike-Relative'
        elif self.kalshi_predictor_version == 'v2':
            label = 'V2 Continuation'
        else:
            label = 'V1 Mean-Reversion'

        print(colored(f"{'='*70}", "cyan"))
        print(colored(f"  Kalshi 15m Prediction Daemon -- {mode} MODE", "cyan"))
        print(colored(f"  Predictor: {label}", "cyan"))
        print(colored(f"  Pairs: {', '.join(self.KALSHI_PAIRS.keys())}", "cyan"))
        print(colored(f"  Thresholds: {self.KALSHI_THRESHOLDS}", "cyan"))
        if self._cli_max_bets > 0:
            print(colored(f"  Max concurrent bets: {self._cli_max_bets} (CLI override)", "yellow"))
        if self._cli_max_size_pct > 0:
            print(colored(f"  Position size: {self._cli_max_size_pct*100:.1f}% (CLI override)", "yellow"))
        else:
            print(colored(f"  Position size: 5.0% (default)", "cyan"))
        print(colored(f"{'='*70}", "cyan"))

        # Check model freshness before anything else
        if self.kalshi_predictor_version == "v3":
            self._check_model_freshness()

        # Initialize balance
        if self.demo:
            # Demo mode: query demo exchange for real play-money balance
            self._init_kalshi_client()
            if self.kalshi_client:
                try:
                    bal = self.kalshi_client.get_balance()
                    balance = bal.get("balance", 0) / 100
                    print(colored(
                        f"  [BALANCE] Demo exchange balance: ${balance:.2f}",
                        "green"))
                except Exception as e:
                    print(colored(f"  [BALANCE] Demo balance query failed: {e}", "red"))
        elif self.dry_run:
            # Dry-run: seed simulated balance from production Kalshi
            self._init_kalshi_client()
            if self.kalshi_client:
                try:
                    bal = self.kalshi_client.get_balance()
                    self._dry_balance_cents = bal.get("balance", 10000)
                    print(colored(
                        f"  [BALANCE] Dry-run starting balance: ${self._dry_balance_cents/100:.2f} (from Kalshi)",
                        "green"))
                except Exception:
                    print(colored("  [BALANCE] Dry-run using default $100.00", "yellow"))

        print("\n[STARTUP] Fetching initial 15m data for Kalshi pairs...")
        self._fetch_all()
        print(f"[STARTUP] Loaded data for {len(self._dataframes)}/{len(self.KALSHI_PAIRS)} pairs")
        for sym, df in self._dataframes.items():
            last = df.iloc[-1]
            rsi = last.get("rsi", 0)
            close = float(last["close"])
            print(f"  {sym}: ${close:.4f} | RSI={rsi:.1f} | {len(df)} candles")

        # Start Kalshi WebSocket for real-time contract prices
        self._start_kalshi_ws()

        # Recover any open positions/orders from a previous run
        if not self.dry_run or self.demo:
            self._recover_positions()

        return True

    # ------------------------------------------------------------------
    # Mid-cycle position recovery
    # ------------------------------------------------------------------

    # Reverse map: series ticker prefix → (symbol, asset)
    _SERIES_TO_PAIR = {
        "KXBTC15M": ("BTC/USDT", "BTC"),
        "KXETH15M": ("ETH/USDT", "ETH"),
        "KXSOL15M": ("SOL/USDT", "SOL"),
        "KXXRP15M": ("XRP/USDT", "XRP"),
    }

    def _recover_positions(self):
        """Recover open positions and resting orders from Kalshi API.

        Called on startup so the daemon can manage positions placed in a
        prior run (or earlier in the same session) that are still live.
        """
        _empty_chop = {
            "bbw_15m": None, "atr_pct_15m": None,
            "bbw_1h": None, "atr_pct_1h": None,
            "bbw_15m_mkt": None, "atr_pct_15m_mkt": None,
            "bbw_1h_mkt": None, "atr_pct_1h_mkt": None,
        }
        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        now_utc = datetime.now(timezone.utc)
        recovered_positions = 0
        recovered_orders = 0

        # --- 1. Recover filled positions ---
        try:
            positions = self.kalshi_client.get_positions()
        except Exception as e:
            print(colored(f"  [RECOVERY] Failed to query positions: {e}", "red"))
            return

        # Track tickers we already have in _pending_bets
        known_tickers = {b.get("ticker") for b in self._pending_bets}

        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker or ticker in known_tickers:
                continue

            # Only recover K15 markets
            series = None
            for prefix, (symbol, asset) in self._SERIES_TO_PAIR.items():
                if ticker.startswith(prefix):
                    series = prefix
                    break
            if not series:
                continue

            # Determine position side and count
            # Kalshi positions: market_exposure > 0 = YES, < 0 = NO
            # Or: position field directly
            pos_qty = int(float(pos.get("position_fp", pos.get("position", 0))))
            if pos_qty == 0:
                continue

            side = "yes" if pos_qty > 0 else "no"
            count = abs(pos_qty)

            # Get market details for strike and close_time
            try:
                mkt_resp = self.kalshi_client.get_market(ticker)
                mkt = mkt_resp.get("market", mkt_resp)
            except Exception as e:
                print(colored(f"  [RECOVERY] Failed to get market {ticker}: {e}", "red"))
                continue

            strike = mkt.get("floor_strike")
            if strike:
                strike = float(strike)
            ct = mkt.get("close_time") or mkt.get("expiration_time", "")
            close_time_dt = None
            if ct and "T" in ct:
                close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))

            # Skip already-settled markets
            if close_time_dt and close_time_dt < now_utc:
                continue

            # Recover entry price with strict no-fallback semantics.
            def _to_cents(value):
                try:
                    v = float(value)
                except Exception:
                    return None
                if v <= 0:
                    return None
                return int(v * 100) if v < 1.5 else int(v)

            entry_price = None
            for key in ("avg_price", "average_price", "entry_price", "yes_price", "no_price"):
                v = _to_cents(pos.get(key))
                if v is not None:
                    entry_price = v
                    break

            if entry_price is None:
                try:
                    fills = self.kalshi_client.get_fills(ticker=ticker, limit=10)
                except Exception as e:
                    print(colored(f"  [RECOVERY] Failed to get fills for {ticker}: {e}", "red"))
                    fills = []
                if fills:
                    my_fills = [f for f in fills if f.get("side", "").lower() == side]
                    prices = []
                    for f in my_fills:
                        raw_price = (
                            f.get("yes_price_fp")
                            if side == "yes" else f.get("no_price_fp")
                        )
                        pv = _to_cents(raw_price)
                        if pv is not None:
                            prices.append(pv)
                    if prices:
                        entry_price = sum(prices) // len(prices)

            if entry_price is None:
                print(colored(
                    f"  [RECOVERY] Skipping {ticker}: unable to recover entry price from fills/position data",
                    "yellow",
                ))
                continue

            direction = "YES" if side == "yes" else "NO"
            bet_entry = {
                "asset": asset,
                "symbol": symbol,
                "side": side,
                "direction": direction,
                "strike": strike or 0,
                "confidence": 0,  # unknown from prior run
                "bet_time": now_utc,
                "settle_time": close_time_dt or now_utc,
                "fill_price": entry_price,
                "contract_price": entry_price,
                "count": count,
                "order_id": "",  # original order_id is unknown
                "ticker": ticker,
                "live": True,
                "needs_fill_check": False,
                "_recovered": True,
                "chop_metrics": _empty_chop,
            }
            self._pending_bets.append(bet_entry)
            self._active_kalshi_bets[ticker] = time.time()
            known_tickers.add(ticker)
            recovered_positions += 1

            # Mark asset as bet_placed so eval loop won't place a duplicate
            self._kalshi_pending_signals[asset] = {
                "direction": direction,
                "base_conf": 0,
                "last_5m_conf": 0,
                "bet_placed": True,
                "bet_attempted": True,
                "window_start": now_utc.replace(
                    minute=now_utc.minute - (now_utc.minute % 15),
                    second=0, microsecond=0,
                ),
                "strike_price": strike or 0,
                "close_time": close_time_dt,
                "probability": 0,
                "recommended_side": direction,
                "max_price_cents": 60,
                "distance_atr": 0,
                "setup_time": now_utc,
                "confirmed": True,
            }

            print(colored(
                f"  [RECOVERY] Position: {asset} {direction} x{count} @ {entry_price}c "
                f"strike={strike} ticker={ticker}",
                "cyan",
            ))

            # Subscribe WebSocket to this ticker
            if self.kalshi_ws:
                self.kalshi_ws.subscribe_ticker(ticker)

        # --- 2. Recover resting (unfilled) orders ---
        try:
            resting = self.kalshi_client.get_orders(status="resting")
        except Exception as e:
            print(colored(f"  [RECOVERY] Failed to query resting orders: {e}", "red"))
            resting = []

        known_order_ids = {o.get("order_id") for o in self._resting_orders}

        for order in resting:
            order_id = order.get("order_id", "")
            ticker = order.get("ticker", "")
            if not ticker or not order_id or order_id in known_order_ids:
                continue

            # Only recover K15 markets
            series = None
            for prefix, (symbol, asset) in self._SERIES_TO_PAIR.items():
                if ticker.startswith(prefix):
                    series = prefix
                    break
            if not series:
                continue

            side = order.get("side", "yes")
            price = int(float(order.get("yes_price_fp", order.get("no_price_fp", 50))))
            remaining = int(float(order.get("remaining_count_fp", order.get("remaining_count", 1))))

            # Get market details
            try:
                mkt_resp = self.kalshi_client.get_market(ticker)
                mkt = mkt_resp.get("market", mkt_resp)
            except Exception:
                continue

            strike = mkt.get("floor_strike")
            if strike:
                strike = float(strike)
            ct = mkt.get("close_time") or mkt.get("expiration_time", "")
            close_time_dt = None
            if ct and "T" in ct:
                close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))

            if close_time_dt and close_time_dt < now_utc:
                continue

            direction = "YES" if side == "yes" else "NO"
            resting_entry = {
                "asset": asset,
                "symbol": symbol,
                "side": side,
                "direction": direction,
                "strike": strike or 0,
                "confidence": 0,
                "bet_time": now_utc,
                "settle_time": close_time_dt or now_utc,
                "fill_price": price,
                "contract_price": price,
                "count": 0,  # unfilled
                "order_id": order_id,
                "ticker": ticker,
                "live": True,
                "needs_fill_check": True,
                "_ws_detected": False,
                "_recovered": True,
                "chop_metrics": _empty_chop,
            }
            self._resting_orders.append(resting_entry)
            self._active_kalshi_bets[ticker] = time.time()
            known_order_ids.add(order_id)
            recovered_orders += 1

            # Mark asset as bet_placed so eval loop won't place a duplicate
            if asset not in self._kalshi_pending_signals:
                self._kalshi_pending_signals[asset] = {
                    "direction": direction,
                    "base_conf": 0,
                    "last_5m_conf": 0,
                    "bet_placed": True,
                    "bet_attempted": True,
                    "window_start": now_utc.replace(
                        minute=now_utc.minute - (now_utc.minute % 15),
                        second=0, microsecond=0,
                    ),
                    "strike_price": strike or 0,
                    "close_time": close_time_dt,
                    "probability": 0,
                    "recommended_side": direction,
                    "max_price_cents": 60,
                    "distance_atr": 0,
                    "setup_time": now_utc,
                    "confirmed": True,
                }

            print(colored(
                f"  [RECOVERY] Resting order: {asset} {direction} x{remaining} @ {price}c "
                f"order_id={order_id}",
                "cyan",
            ))

            if self.kalshi_ws:
                self.kalshi_ws.subscribe_ticker(ticker)

        if recovered_positions or recovered_orders:
            print(colored(
                f"  [RECOVERY] Recovered {recovered_positions} positions, "
                f"{recovered_orders} resting orders — will manage through settlement",
                "green",
            ))

            # If we loaded after minute 10, run M10 confirmation immediately
            # on recovered positions (they missed the normal minute-10 check)
            minute_in_window = now_utc.minute % 15
            if minute_in_window >= 10 and recovered_positions > 0:
                print(colored(
                    f"  [RECOVERY] Minute {minute_in_window} — running M10 confirmation on recovered positions",
                    "yellow",
                ))
                self._m10_confirm()
        else:
            print("  [RECOVERY] No existing positions or orders found")

    def tick(self):
        """Wall-clock Kalshi evaluation trigger (every minute)."""
        # Settlements and resting checks are now inside _kalshi_eval()

        # Eval triggers:
        # Min 0-1: entry — frequent checks for market readiness + sub-60c fills
        # Min 10+: confirmation — recheck with minute-10 data
        # Other: every 50s for monitoring/settlement
        now = time.time()
        now_utc = datetime.now(timezone.utc)
        min_in_window = now_utc.minute % 15
        sec_in = now_utc.second
        time_since = now - self._last_kalshi_eval
        # Entry: retry every 2s during min 0-1 (per-asset bet_placed checked inside eval)
        # Confirm: every 5s at min 10+
        # Normal: every 50s for monitoring
        entry_trigger = (min_in_window <= 1 and time_since >= 2)
        confirm_trigger = (min_in_window >= 10 and time_since >= 5)
        normal_trigger = (min_in_window >= 2 and min_in_window < 10 and time_since >= 50)
        should_eval = entry_trigger or confirm_trigger or normal_trigger
        if should_eval:
            try:
                self._kalshi_eval()
            except Exception as e:
                print(colored(f"  [KALSHI EVAL ERR] {e}", "red"))
            self._last_kalshi_eval = now

        # Print status
        self._print_status()

    def _print_status(self):
        """Print a concise status line."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        if self.demo:
            mode_tag = colored("[DEMO]", "yellow")
        elif self.dry_run:
            mode_tag = colored("[DRY-RUN]", "magenta")
        else:
            mode_tag = colored("[LIVE]", "green")

        total = self._session_wins + self._session_losses + self._session_partial_losses
        wr = f"{self._session_wins}/{total} ({100*self._session_wins/total:.0f}%)" if total > 0 else "0/0"
        pnl = sum(b.get("pnl_dollars", 0) for b in self._completed_bets)
        pnl_str = f"${pnl:+.2f}"
        pnl_colored = colored(pnl_str, "green" if pnl >= 0 else "red")

        active = len(self._active_kalshi_bets)
        pending = len(self._pending_bets)

        print(
            f"\n{mode_tag} {now} UTC | "
            f"Bets: {self._session_bets_placed} placed | "
            f"Active: {active} | Pending settle: {pending} | "
            f"W/L: {wr} | P&L: {pnl_colored}"
        )

    def run(self, max_cycles: int = 0):
        """Main daemon loop.

        Args:
            max_cycles: stop after this many ticks (0 = run forever).
        """
        self._running = True

        def _handle_shutdown(*_):
            print(colored("\n[SHUTDOWN] Graceful shutdown requested...", "yellow"))
            self._running = False

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)
        import atexit
        atexit.register(lambda: self._cancel_all_resting_orders("process exit"))
        atexit.register(self._release_instance_lock)

        if self.startup() is False:
            self._running = False
            return

        # Run initial Kalshi eval immediately
        try:
            self._kalshi_eval()
            self._last_kalshi_eval = time.time()
        except Exception as e:
            print(colored(f"  [KALSHI EVAL ERR] startup eval: {e}", "red"))

        cycle = 0
        while self._running:
            try:
                time.sleep(TICK_INTERVAL)
                if not self._running:
                    break

                self.tick()

                cycle += 1
                if max_cycles > 0 and cycle >= max_cycles:
                    print(colored(f"\n[STOP] Reached max cycles ({max_cycles})", "yellow"))
                    break

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(colored(f"[ERROR] {e}", "red"))
                time.sleep(TICK_INTERVAL)

        # Cancel all resting orders before exit
        if not self.dry_run or self.demo:
            self._cancel_all_resting_orders("shutdown")

        # Shutdown summary
        total = self._session_wins + self._session_losses + self._session_partial_losses
        pnl = sum(b.get("pnl_dollars", 0) for b in self._completed_bets)
        print(colored(f"\n{'='*70}", "cyan"))
        print(colored("  KALSHI DAEMON STOPPED", "cyan"))
        print(f"  Bets placed: {self._session_bets_placed}")
        print(f"  W:{self._session_wins} L:{self._session_losses} PL:{self._session_partial_losses} | Total: {total}")
        if total > 0:
            print(f"  Win rate: {100*self._session_wins/total:.1f}%")
        print(f"  P&L: ${pnl:+.2f}")
        print(f"  Pending settlement: {len(self._pending_bets)}")
        print(colored(f"{'='*70}", "cyan"))
        self._release_instance_lock()


def main():
    parser = argparse.ArgumentParser(description="Kalshi 15-minute prediction daemon")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Record predictions without placing real bets",
    )
    parser.add_argument(
        "--predictor", choices=["v1", "v2", "v3"], default="v3",
        help="Kalshi predictor: v1 (mean-reversion), v2 (continuation), or v3 (strike-relative)",
    )
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Stop after N ticks (0 = run forever)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use Kalshi demo exchange (real orders, play money). Requires KalshiDemoKeys.txt",
    )
    parser.add_argument(
        "--maxbets", type=int, default=0,
        help="Max concurrent bets (0 = default). Highest confidence wins when limited.",
    )
    parser.add_argument(
        "--maxsize", type=float, default=0,
        help="Position size as %% of balance (0 = default 5%%). E.g. --maxsize=2.5",
    )
    args = parser.parse_args()

    daemon = KalshiDaemon(
        dry_run=args.dry_run or args.demo,
        predictor_version=args.predictor,
        demo=args.demo,
        max_bets=args.maxbets,
        max_size_pct=args.maxsize,
    )
    daemon.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
