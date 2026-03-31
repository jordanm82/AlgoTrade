#!/usr/bin/env python3
"""Kalshi 15-minute prediction daemon — standalone K15 UpDown bot.

Usage:
    python -m cli.kalshi_daemon --dry-run --predictor v3 --cycles 10
"""

import argparse
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from termcolor import colored

from config.production import MAX_CONCURRENT_KALSHI_BETS
from config.settings import CDP_KEY_FILE, DATA_DIR
from data.fetcher import DataFetcher
from data.indicators import add_indicators
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

    # Per-asset LR thresholds (dual trend+conviction model, walk-forward validated)
    KALSHI_THRESHOLDS = {
        "BTC/USDT": 53,
        "ETH/USDT": 53,
        "SOL/USDT": 53,
        "XRP/USDT": 53,
    }

    # Per-asset TEK (technical/probability table) thresholds
    KALSHI_TBL_THRESHOLDS = {
        "BTC/USDT": 30.0,
        "ETH/USDT": 30.0,
        "SOL/USDT": 30.0,
        "XRP/USDT": 30.0,
    }

    # Coinbase symbol mapping for live price (matches BRTI settlement source)
    COINBASE_PRICE_MAP = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "XRP/USDT": "XRP-USD",
    }

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
        self.kalshi_threshold = 30  # minimum confidence to bet
        self.kalshi_predictions: list[dict] = []  # latest predictions for dashboard
        self._active_kalshi_bets = {}  # {ticker: placement_time}
        self._kalshi_pending_signals = {}   # {asset: {direction, base_conf, last_5m_conf, ...}}
        self._last_kalshi_eval = 0          # timestamp of last eval
        self._kalshi_cached_dataframes = {}  # {symbol: DataFrame} cached 15m data
        self._btc_cached_1m: pd.DataFrame | None = None  # BTC 1m candles for confluence
        self._resting_orders: list[dict] = []  # orders waiting for price to dip

        # Bet tracking — records ALL bets (live + dry-run) and checks settlement
        self._pending_bets: list[dict] = []
        self._completed_bets: list[dict] = []
        self._session_wins = 0
        self._session_losses = 0
        self._session_bets_placed = 0

        # Dry-run simulated balance — starts from actual Kalshi balance or $100
        # Compounds with wins/losses so sizing is realistic
        self._dry_balance_cents = 10000  # updated in startup from actual balance

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
    # Data
    # ------------------------------------------------------------------

    def _fetch_all(self):
        """Fetch and cache 15m + 1h + 4h candles for all pairs. Called once at startup."""
        for symbol in self.KALSHI_PAIRS:
            try:
                df = self.fetcher.ohlcv(symbol, timeframe="15m", limit=200)
                df = add_indicators(df)
                self._dataframes[symbol] = df
                self._kalshi_cached_dataframes[symbol] = df
            except Exception as e:
                print(colored(f"  [WARN] 15m fetch failed for {symbol}: {e}", "yellow"))
            try:
                df_1h = self.fetcher.ohlcv(symbol, "1h", limit=50)
                if df_1h is not None and not df_1h.empty:
                    self._kalshi_cached_dataframes[f"{symbol}_1h"] = add_indicators(df_1h)
            except Exception:
                pass
            try:
                df_4h = self.fetcher.ohlcv(symbol, "4h", limit=50)
                if df_4h is not None and not df_4h.empty:
                    self._kalshi_cached_dataframes[f"{symbol}_4h"] = add_indicators(df_4h)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Coinbase price
    # ------------------------------------------------------------------

    def _get_kalshi_strike(self, series_ticker: str) -> tuple:
        """Get strike price and close time for current open market.

        Falls back to previous settled market's expiration_value if
        floor_strike is None (Kalshi sometimes delays populating it).

        Returns: (strike, close_time_dt, ticker) or (None, None, None)
        """
        self._init_kalshi_client()
        if not self.kalshi_client:
            return None, None, None

        try:
            markets = self.kalshi_client.get_markets(series_ticker=series_ticker, status="open")
            if not markets:
                return None, None, None

            m = markets[0]
            strike = m.get("floor_strike")
            ticker = m.get("ticker", "")
            ct = m.get("close_time") or m.get("expiration_time", "")
            close_time_dt = None
            if ct and "T" in ct:
                close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))

            # If floor_strike is None, derive from previous settled market
            if not strike:
                settled = self.kalshi_client.get_markets(
                    series_ticker=series_ticker, status="settled"
                )
                if settled:
                    # Most recent settled market's expiration_value = current strike
                    strike = settled[0].get("expiration_value")
                    if strike:
                        strike = float(strike)

            if strike:
                strike = float(strike)

            return strike, close_time_dt, ticker

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
        except Exception:
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

        # Only check bets past their settlement time (+ 1 min buffer for Kalshi to settle)
        due_bets = [b for b in self._pending_bets
                    if now >= b["settle_time"] + pd.Timedelta(minutes=1)]
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

                        entry_price = bet.get("contract_price", 0)
                        count = bet.get("count", 1)
                        settled_value = float(m.get("expiration_value", 0))

                        if self.dry_run and not self.demo:
                            # Dry-run: compute P&L from simulated entry
                            won = bet["side"] == result
                            if won:
                                pnl_cents = count * (100 - entry_price)
                            else:
                                pnl_cents = -(count * entry_price)
                            pnl_dollars = pnl_cents / 100
                        else:
                            # Live: use Kalshi settlements API for real P&L
                            try:
                                setts = self.kalshi_client.get_settlements(limit=50)
                                matched = False
                                for s in setts:
                                    if s.get("ticker") == m.get("ticker"):
                                        revenue = int(s.get("revenue", 0))
                                        y_cost = float(s.get("yes_total_cost_dollars", 0)) * 100
                                        n_cost = float(s.get("no_total_cost_dollars", 0)) * 100
                                        pnl_cents = revenue - int(y_cost + n_cost)
                                        pnl_dollars = pnl_cents / 100
                                        won = pnl_cents > 0
                                        count = int(float(s.get("yes_count_fp", 0))) or int(float(s.get("no_count_fp", 0)))
                                        if count > 0:
                                            entry_price = int((y_cost + n_cost) / count)
                                        matched = True
                                        break
                                if not matched:
                                    won = bet["side"] == result
                                    pnl_cents = count * (100 - entry_price) if won else -(count * entry_price)
                                    pnl_dollars = pnl_cents / 100
                            except Exception:
                                won = bet["side"] == result
                                pnl_cents = count * (100 - entry_price) if won else -(count * entry_price)
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

                        settled.append(bet)
                        break

            except Exception as e:
                print(colored(f"  [SETTLE ERR] {bet['asset']}: {e}", "yellow"))

        # Remove settled bets from pending
        if settled:
            self._pending_bets = [b for b in self._pending_bets if b not in settled]

    # ------------------------------------------------------------------
    # Kalshi evaluation lifecycle
    # ------------------------------------------------------------------

    def _kalshi_eval(self):
        """Kalshi evaluation — continuous entry model.

        Lifecycle within each 15m window:
        - SETUP (min 0-1): Fetch 15m candles, compute indicators, cache.
        - CONFIRMED (min 2-10): 1m updates, evaluate + execute if signal+price align.
        - DONE (min 11+): Too close to settlement.
        """
        from data.market_data import get_order_book_imbalance, get_trade_flow

        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15

        # Check resting orders — monitor during entry window, cancel at minute 10+
        if self._resting_orders and minute_in_window >= 2:
            self._check_resting_orders()

        # Settlement check moved to dashboard refresh cycle (every 5s)

        # Compute current window start (round down to :00/:15/:30/:45)
        window_minute = now_utc.minute - minute_in_window
        current_window_start = now_utc.replace(minute=window_minute, second=0, microsecond=0)
        # Pandas-compatible timestamp for DataFrame index comparisons
        current_window_start_pd = pd.Timestamp(current_window_start.replace(tzinfo=None))

        # Prune expired bets
        now_ts = time.time()
        self._active_kalshi_bets = {
            t: placed for t, placed in self._active_kalshi_bets.items()
            if now_ts - placed < 900
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

        for symbol, series_ticker in self.KALSHI_PAIRS.items():
            asset = symbol.split("/")[0]

            # Check if pending signal belongs to current window
            pending = self._kalshi_pending_signals.get(asset)
            if pending and pending.get("window_start") != current_window_start:
                # New window — clear old signal and cached data
                self._kalshi_pending_signals.pop(asset, None)
                self._kalshi_cached_dataframes.pop(symbol, None)
                pending = None

            # Skip if already bet this window — show stored scores
            if pending and pending.get("bet_placed"):
                predictions.append({
                    "symbol": symbol, "asset": asset,
                    "direction": pending.get("direction", "--"),
                    "confidence": pending.get("last_5m_conf", pending.get("base_conf", 0)),
                    "knn_score": pending.get("bet_knn_score", 0),
                    "knn_thresh": self.KALSHI_THRESHOLDS.get(symbol, 60),
                    "tbl_score": pending.get("bet_tbl_score", 0),
                    "tbl_thresh": self.KALSHI_TBL_THRESHOLDS.get(symbol, 55),
                    "btc_score": pending.get("bet_btc_score", 0),
                    "btc_thresh": 0,
                    "reason": "already bet this window",
                    "ob": 0, "flow": 0, "state": "BET_PLACED",
                })
                continue

            # Lifecycle:
            # Min 0:   SETUP — cache 15m/1h, query strike from Kalshi
            # Min 1-8: CONFIRMED — 1m candle closed, compute distance, bet immediately
            #          (83% WR at min 1 with contracts still ~50c)
            # Min 9+:  MONITORING — no new bets, cancel resting at min 10
            if minute_in_window < 1:
                state = "SETUP"
            elif minute_in_window <= 8 and not (pending and pending.get("bet_placed")):
                state = "CONFIRMED"
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
                    "tbl_score": pending.get("bet_tbl_score", 0),
                    "tbl_thresh": self.KALSHI_TBL_THRESHOLDS.get(symbol, 55),
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
                    strike, close_time_dt, market_ticker = self._get_kalshi_strike(series_ticker)

                    # Build minute-3 snapshot using shared function (same as backtest)
                    signal = None
                    if strike and close_time_dt:
                        try:
                            df_1m_setup = self.fetcher.ohlcv(symbol, "1m", limit=10)
                            snapshot = build_minute3_snapshot(df_15m, df_1m_setup, current_window_start_pd)
                            if snapshot is not None:
                                signal = self.kalshi_predictor.predict(
                                    snapshot, strike_price=float(strike),
                                    minutes_remaining=12,
                                    market_data=market_data, df_1h=df_1h,
                                    df_4h=self._kalshi_cached_dataframes.get(f"{symbol}_4h"),
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
                                "probability": signal.probability,
                                "recommended_side": signal.recommended_side,
                                "max_price_cents": signal.max_price_cents,
                                "distance_atr": signal.distance_atr,
                                "bet_placed": False,
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

                    model = "KNN" if signal and signal.adjustments.get("mode") == "knn_early_entry" else "TBL"
                    side = signal.recommended_side if signal and signal.recommended_side != "SKIP" else "--"
                    raw_pct = int(signal.probability * 100) if signal else 0
                    # Display confidence in the recommended direction
                    setup_conf = raw_pct if side == "YES" else (100 - raw_pct) if side == "NO" else raw_pct

                    # Compute KNN & TBL scores for display (preview — thresholds checked at CONFIRMED)
                    knn_score = setup_conf if signal else 0
                    knn_thresh = self.KALSHI_THRESHOLDS.get(symbol, 60)
                    tbl_score = 0
                    tbl_thresh = self.KALSHI_TBL_THRESHOLDS.get(symbol, 55.0)
                    if signal and side != "--" and strike:
                        cb_price = self._get_coinbase_price(symbol)
                        tbl_sig = self.kalshi_predictor.predict(
                            df_15m, strike_price=float(strike),
                            minutes_remaining=12,
                            current_price=cb_price,
                            force_table=True,
                        )
                        if tbl_sig:
                            tp = int(tbl_sig.probability * 100)
                            tbl_score = tp if side == "YES" else (100 - tp)

                    # BTC score preview
                    btc_preview = self.compute_btc_score(side) if side in ("YES", "NO") else 50

                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": side,
                        "confidence": setup_conf,
                        "knn_score": knn_score, "knn_thresh": knn_thresh,
                        "tbl_score": tbl_score, "tbl_thresh": tbl_thresh,
                        "btc_score": btc_preview, "btc_thresh": 0,
                        "reason": f"setup [{model}] — waiting for confirmation",
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

                # Use CACHED 15m from SETUP (already fetched — no re-fetch)
                market_data = None
                cb_price = self._get_coinbase_price(symbol)
                df_15m = self._kalshi_cached_dataframes.get(symbol)

                # If no cache (late start), fetch once
                if df_15m is None or len(df_15m) < 50:
                    try:
                        raw = self.fetcher.ohlcv(symbol, "15m", limit=200)
                        if raw is not None and not raw.empty:
                            df_15m = add_indicators(raw)
                            self._kalshi_cached_dataframes[symbol] = df_15m
                    except Exception:
                        pass

                # Build snapshot using 1m candles for early entry (contracts still cheap)
                try:
                    df_sub = self.fetcher.ohlcv(symbol, "1m", limit=10)
                    if df_15m is not None and df_sub is not None and len(df_sub) > 0:
                        snapshot = build_minute3_snapshot(df_15m, df_sub, current_window_start_pd)
                        if snapshot is not None:
                            df_15m = snapshot
                        else:
                            print(colored(f"  [ERR] {asset}: snapshot failed — skipping", "red"))
                            predictions.append({
                                "symbol": symbol, "asset": asset,
                                "direction": "--", "confidence": 0,
                                "reason": "snapshot failed", "ob": 0, "flow": 0, "state": state,
                            })
                            continue
                    else:
                        print(colored(f"  [ERR] {asset}: no 5m data — skipping", "red"))
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": "--", "confidence": 0,
                            "reason": "no 5m data", "ob": 0, "flow": 0, "state": state,
                        })
                        continue
                except Exception as e:
                    print(colored(f"  [ERR] {asset}: 5m fetch failed: {e} — skipping", "red"))
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": f"5m fetch error: {e}", "ob": 0, "flow": 0, "state": state,
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

                    if not strike:
                        # No SETUP ran — query Kalshi for strike (late start)
                        strike, close_time_dt, _ = self._get_kalshi_strike(series_ticker)

                    if strike and close_time_dt:
                        mins_left = max(0, (close_time_dt - now_utc).total_seconds() / 60)
                        cb_price = self._get_coinbase_price(symbol)
                        signal = self.kalshi_predictor.predict(
                            df_15m, strike_price=float(strike),
                            minutes_remaining=mins_left,
                            market_data=market_data, df_1h=df_1h,
                            current_price=cb_price,
                            df_4h=df_4h,
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
                        # Apply per-asset KNN confidence threshold
                        asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, 60)
                        prob_pct = int(signal.probability * 100)
                        meets_knn = prob_pct >= asset_threshold or prob_pct <= (100 - asset_threshold)

                        # TEK agreement gate — probability table confirms LR direction
                        tbl_threshold = self.KALSHI_TBL_THRESHOLDS.get(symbol, 55.0)
                        tbl_agrees = False
                        tbl_sig = None
                        if meets_knn:
                            cb_price = self._get_coinbase_price(symbol)
                            # Use actual minutes remaining + force_table to get probability
                            # table score (not another KNN prediction)
                            tek_mins = max(0, (pending.get("close_time", now_utc) - now_utc).total_seconds() / 60) if pending.get("close_time") else 10
                            tbl_sig = self.kalshi_predictor.predict(
                                df_15m, strike_price=float(pending.get("strike_price", 0)),
                                minutes_remaining=tek_mins,
                                current_price=cb_price,
                                force_table=True,
                            )
                            if tbl_sig:
                                tbl_pct = int(tbl_sig.probability * 100)
                                if signal.recommended_side == "YES":
                                    tbl_agrees = tbl_pct >= tbl_threshold
                                elif signal.recommended_side == "NO":
                                    tbl_agrees = (100 - tbl_pct) >= tbl_threshold

                        # Compute TBL display score early (needed for SIGNAL log)
                        tbl_display = 0
                        if tbl_sig:
                            tp = int(tbl_sig.probability * 100)
                            tbl_display = tp if signal.recommended_side == "YES" else (100 - tp)

                        # Strike-relative model handles direction via distance_from_strike.
                        # TEK and trend kill removed — both redundant with this model.
                        all_pass = meets_knn

                        # Store TBL score for BET_PLACED display
                        if pending:
                            pending["bet_tbl_score"] = tbl_display

                        if state == "CONFIRMED" and all_pass:
                            dir_conf = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)
                            print(colored(
                                f"  [SIGNAL] {asset} {signal.recommended_side} "
                                f"LR={dir_conf}% TEK={tbl_display}% → actionable",
                                "cyan",
                            ))
                            actionable_signals.append({
                                "symbol": symbol, "series_ticker": series_ticker,
                                "signal": signal, "market_data": market_data,
                                "state": state,
                            })
                        model = "KNN" if signal.adjustments.get("mode") == "knn_early_entry" else "TBL"
                        reason_suffix = ""
                        dir_c = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)
                        if not meets_knn:
                            reason_suffix = f" (KNN below {asset_threshold}%)"
                            if minute_in_window >= 2:
                                print(colored(
                                    f"  [SKIP] {asset} {signal.recommended_side} "
                                    f"LR={dir_c}%<{asset_threshold}% TEK={tbl_display}%",
                                    "dark_grey"))
                        elif not tbl_agrees:
                            reason_suffix = f" (TEK disagrees, need {tbl_threshold}%)"
                            if minute_in_window >= 2:
                                print(colored(
                                    f"  [SKIP] {asset} {signal.recommended_side} "
                                    f"LR={dir_c}% TEK={tbl_display}%<{tbl_threshold}%",
                                    "dark_grey"))
                        elif state == "CONFIRMED":
                            reason_suffix = f" -> BETTING (LR+TEK)"
                        elif state == "MONITORING":
                            reason_suffix = " (monitoring)"
                        else:
                            reason_suffix = " (waiting)"

                        # Display confidence in the RECOMMENDED direction, not raw UP probability
                        display_conf = prob_pct if signal.recommended_side == "YES" else (100 - prob_pct)

                        # If KNN failed and we didn't compute TBL yet, do it now for display
                        if not meets_knn and tbl_display == 0:
                            cb_price_disp = self._get_coinbase_price(symbol)
                            tek_mins_disp = max(0, (pending.get("close_time", now_utc) - now_utc).total_seconds() / 60) if pending.get("close_time") else 10
                            tbl_sig_disp = self.kalshi_predictor.predict(
                                df_15m, strike_price=float(pending.get("strike_price", 0)),
                                minutes_remaining=tek_mins_disp,
                                current_price=cb_price_disp,
                                force_table=True,
                            )
                            if tbl_sig_disp:
                                tp = int(tbl_sig_disp.probability * 100)
                                tbl_display = tp if signal.recommended_side == "YES" else (100 - tp)

                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": signal.recommended_side,
                            "confidence": display_conf,
                            "knn_score": display_conf, "knn_thresh": asset_threshold,
                            "tbl_score": tbl_display, "tbl_thresh": tbl_threshold,
                            "btc_score": 0, "btc_thresh": 0,
                            "reason": f"{state.lower()} [{model}]: prob={signal.probability:.2f} side={signal.recommended_side}" + reason_suffix,
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })
                    else:
                        # SKIP — still compute scores for dashboard visibility
                        model = "KNN" if hasattr(signal, 'adjustments') and signal.adjustments.get("mode") == "knn_early_entry" else "TBL"
                        raw_pct = int(signal.probability * 100) if hasattr(signal, 'probability') else 50
                        # Show how close to threshold (e.g., 52% → "52%")
                        skip_knn = raw_pct if raw_pct >= 50 else (100 - raw_pct)

                        # Compute TBL even for SKIP
                        skip_tbl = 0
                        if pending and pending.get("strike_price"):
                            cb_p = self._get_coinbase_price(symbol)
                            tek_mins_skip = max(0, (pending.get("close_time", now_utc) - now_utc).total_seconds() / 60) if pending.get("close_time") else 10
                            tbl_s = self.kalshi_predictor.predict(
                                df_15m, strike_price=float(pending["strike_price"]),
                                minutes_remaining=tek_mins_skip, current_price=cb_p,
                                force_table=True,
                            )
                            if tbl_s:
                                tp = int(tbl_s.probability * 100)
                                # Show the stronger side
                                skip_tbl = max(tp, 100 - tp)

                        # BTC score (neutral since no direction chosen)
                        skip_btc = self.compute_btc_score("YES")  # arbitrary side, shows magnitude

                        asset_threshold = self.KALSHI_THRESHOLDS.get(symbol, 60)
                        tbl_threshold = self.KALSHI_TBL_THRESHOLDS.get(symbol, 55.0)

                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": "--",
                            "confidence": raw_pct,
                            "knn_score": skip_knn, "knn_thresh": asset_threshold,
                            "tbl_score": skip_tbl, "tbl_thresh": tbl_threshold,
                            "btc_score": skip_btc, "btc_thresh": 0,
                            "reason": f"{state.lower()} [{model}]: SKIP (prob={signal.probability:.2f})",
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
                max_bets = 4 if self.dry_run else MAX_CONCURRENT_KALSHI_BETS
            if len(self._active_kalshi_bets) + actual_fills_this_eval >= max_bets:
                break
            result = self._kalshi_execute_bet(
                vs["symbol"], vs["series_ticker"], vs["signal"], vs["market_data"]
            )
            asset = vs["symbol"].split("/")[0]
            reason = result.get("reason", "") if result else ""
            has_fill = "filled=" in reason and "filled=0" not in reason
            is_dry_bet = "would bet" in reason
            was_skipped = "skipping" in reason or "price too high" in reason

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
                        self._kalshi_pending_signals[asset]["bet_tbl_score"] = self._kalshi_pending_signals[asset].get("bet_tbl_score", 0)
            elif was_skipped and asset in self._kalshi_pending_signals:
                # Price blocked — mark for price monitoring (5s checks until min 10)
                self._kalshi_pending_signals[asset]["price_watch"] = True
                self._kalshi_pending_signals[asset]["watch_side"] = vs["signal"].recommended_side
                self._kalshi_pending_signals[asset]["watch_series"] = vs["series_ticker"]

        self.kalshi_predictions = predictions

    # ------------------------------------------------------------------
    # Bet execution
    # ------------------------------------------------------------------

    def _kalshi_execute_bet(self, symbol: str, series_ticker: str, signal, market_data: dict | None) -> dict:
        """Execute a single Kalshi bet. Returns a pred dict with outcome details.

        Handles: balance check, market discovery, orderbook pricing,
        order placement, fill verification, active bet tracking.
        """
        asset = symbol.split("/")[0]
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

        RISK_PER_BET_PCT = 0.05

        pred["reason"] = f"would bet {side.upper()} (conf={conf_display})"

        if self.dry_run and not self.demo:
            # Record dry-run bet with ACTUAL contract price from Kalshi
            strike = 0
            settle_time = None
            contract_price = 0
            ticker = ""

            if isinstance(signal, KalshiV3Signal):
                strike = signal.strike_price
                pending = self._kalshi_pending_signals.get(asset, {})
                settle_time = pending.get("close_time")

            # Query Kalshi for actual contract price
            self._init_kalshi_client()
            if self.kalshi_client and strike:
                try:
                    series = self.KALSHI_PAIRS.get(symbol, "")
                    markets = self.kalshi_client.get_markets(series_ticker=series, status="open")
                    if markets:
                        m = markets[0]
                        ticker = m.get("ticker", "")
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
                        except Exception:
                            pass
                except Exception:
                    pass

            # If ask > max, place resting order at our max price
            if contract_price > MAX_ENTRY_CENTS:
                contract_price = MAX_ENTRY_CENTS
                print(colored(
                    f"  [KALSHI REST] {asset} {side.upper()} ask too high — resting order @ {MAX_ENTRY_CENTS}c",
                    "yellow"))
                # Mark as resting for price watch monitoring
                if asset in self._kalshi_pending_signals:
                    self._kalshi_pending_signals[asset]["price_watch"] = True
                    self._kalshi_pending_signals[asset]["watch_side"] = direction_label
                    self._kalshi_pending_signals[asset]["watch_series"] = self.KALSHI_PAIRS.get(symbol, "")

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

            # Find the next 15-min market for this series
            events = self.kalshi_client._get("/trade-api/v2/events", {
                "series_ticker": series_ticker, "status": "open",
                "limit": 3, "with_nested_markets": "true",
            })
            all_markets = []
            for evt in events.get("events", []):
                all_markets.extend(evt.get("markets", []))
            if not all_markets:
                pred["reason"] = f"no {series_ticker} markets found"
                print(colored(f"  [KALSHI] No markets for {series_ticker}", "yellow"))
                return pred

            # Sort by close_time to ensure we pick the soonest-expiring market
            all_markets.sort(key=lambda m: m.get("close_time", "9999"))
            market = all_markets[0]
            ticker = market.get("ticker", "")

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

            # Get current ask price — bid AT the ask for guaranteed fill
            if side == "yes":
                ask_raw = market.get("yes_ask_dollars") or market.get("yes_ask")
            else:
                ask_raw = market.get("no_ask_dollars") or market.get("no_ask")

            if ask_raw:
                fill_price = int(float(ask_raw) * 100) if float(ask_raw) < 1.5 else int(float(ask_raw))
            else:
                fill_price = 50  # fallback

            fill_price = max(1, min(99, fill_price))

            # If ask > max, place resting order at our max price
            if fill_price > MAX_ENTRY_CENTS:
                fill_price = MAX_ENTRY_CENTS
                print(colored(
                    f"  [KALSHI REST] {asset} {side.upper()} ask too high — limit order @ {MAX_ENTRY_CENTS}c",
                    "yellow"))

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

            # Verify fill via Kalshi (exchange is source of truth)
            if float(fill_count) < count and order_status == "resting":
                import time as _t
                _t.sleep(1)
                try:
                    positions = self.kalshi_client.get_positions()
                    for p in positions:
                        if p.get("ticker") == ticker and p.get("position", 0) > 0:
                            fill_count = str(p["position"])
                            break
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
            actual_filled = float(fill_count) if fill_count else 0
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
                        "live": True,
                    })
            elif isinstance(signal, KalshiV3Signal) and actual_filled == 0:
                # Order resting — track for fill check and minute-10 cancellation
                pending_signal = self._kalshi_pending_signals.get(asset, {})
                settle_time = pending_signal.get("close_time")
                resting_entry = {
                    "asset": asset,
                    "symbol": symbol,
                    "side": side,
                    "direction": direction_label,
                    "strike": signal.strike_price,
                    "confidence": conf_display,
                    "bet_time": datetime.now(timezone.utc),
                    "settle_time": settle_time,
                    "fill_price": fill_price,
                    "count": 0,
                    "contract_price": fill_price,
                    "order_id": order_id,
                    "ticker": ticker,
                    "live": True,
                    "needs_fill_check": True,
                }
                if settle_time:
                    self._pending_bets.append(resting_entry)
                self._resting_orders.append(resting_entry)

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
                   if p.get("price_watch") and not p.get("bet_placed")]
        watches.sort(key=lambda x: x[1].get("probability", 0), reverse=True)

        for asset, pending in watches:

            series = pending.get("watch_series", "")
            side = pending.get("watch_side", "")
            if not series or not side:
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
                        max_bets = 4 if self.dry_run else MAX_CONCURRENT_KALSHI_BETS
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

    def _check_resting_orders(self):
        """Check resting orders for fills, cancel at minute 10."""
        if not self._resting_orders:
            return

        now_utc = datetime.now(timezone.utc)
        minute_in_window = now_utc.minute % 15

        self._init_kalshi_client()
        if not self.kalshi_client:
            return

        still_resting = []
        for order in self._resting_orders:
            order_id = order.get("order_id", "")
            asset = order.get("asset", "?")
            if not order_id or order_id == "?":
                continue

            try:
                status = self.kalshi_client.get_order_status(order_id)
                filled = float(status.get("fill_count_fp", 0))
                order_status = status.get("status", "")

                # CANCEL FIRST if past deadline — don't accept late fills
                order_expired = False
                settle_time = order.get("settle_time")
                if settle_time and now_utc > settle_time:
                    order_expired = True

                should_cancel = minute_in_window >= 10 or order_expired
                if should_cancel and order_status != "executed":
                    reason = "previous window" if order_expired else f"min {minute_in_window}"
                    try:
                        self.kalshi_client.cancel_order_safe(order_id)
                        print(colored(
                            f"  [RESTING CANCEL] {asset} {order['side'].upper()} "
                            f"@ {order['fill_price']}c — cancelled ({reason}, was {order_status})",
                            "yellow",
                        ))
                        self._pending_bets = [
                            b for b in self._pending_bets
                            if b.get("order_id") != order_id
                        ]
                    except Exception as e:
                        print(colored(f"  [CANCEL ERR] {asset}: {e}", "red"))
                    continue  # don't add back

                if filled > 0:
                    # Filled (within the allowed window)
                    order["count"] = int(filled)
                    order["needs_fill_check"] = False
                    if asset in self._kalshi_pending_signals:
                        self._kalshi_pending_signals[asset]["bet_placed"] = True
                    print(colored(
                        f"  [RESTING FILL] {asset} {order['side'].upper()} "
                        f"x{int(filled)} @ {order['fill_price']}c — filled!",
                        "green",
                    ))
                    continue  # don't add back to resting list

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
        """Check if LogReg model is older than 7 days. Retrain if stale."""
        from pathlib import Path
        import os

        model_path = Path("models/knn_kalshi.pkl")
        max_age_days = 7

        if not model_path.exists():
            print(colored("  [MODEL] LogReg model not found — training...", "yellow"))
            self._retrain_model()
            return

        age_seconds = time.time() - os.path.getmtime(model_path)
        age_days = age_seconds / 86400

        if age_days > max_age_days:
            print(colored(
                f"  [MODEL] LogReg model is {age_days:.1f} days old (max {max_age_days}) — retraining...",
                "yellow",
            ))
            self._retrain_model()
        else:
            print(colored(
                f"  [MODEL] LogReg model is {age_days:.1f} days old — OK",
                "green",
            ))

    def _retrain_model(self):
        """Retrain strike-relative model with walk-forward validation.

        Uses scripts/retrain_strike_relative.py which:
        - Predicts the actual Kalshi question: 'will price close above strike?'
        - Key feature: distance_from_strike (price at min5 vs strike, in ATR)
        - Walk-forward: train on 120 days oldest, test on 59 days newest
        - Fetches from Coinbase (matches BRTI settlement source)
        """
        import subprocess

        print(colored("  [MODEL] Starting strike-relative retrain (~10 min)...", "yellow"))
        print(colored("  [MODEL] Predicts: 'Will price close above strike?'", "yellow"))

        try:
            result = subprocess.run(
                ["./venv/bin/python", "scripts/retrain_strike_relative.py",
                 "--days", "179", "--output", "models/knn_kalshi.pkl"],
                capture_output=True, text=True, timeout=900,
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

    def tick(self):
        """Wall-clock Kalshi evaluation trigger (every minute)."""
        # Settlements and resting checks are now inside _kalshi_eval()

        # Eval every minute during SETUP (0-1) and ENTRY_WINDOW (2-10)
        now = time.time()
        current_minute = datetime.now(timezone.utc).minute
        min_in_window = current_minute % 15
        # Eval at minute 1 (SETUP), 5 (CONFIRMED — 5m candle closed), 6+ (MONITORING)
        should_eval = (min_in_window >= 1 and now - self._last_kalshi_eval >= 50)
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

        total = self._session_wins + self._session_losses
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

        self.startup()

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

        # Shutdown summary
        total = self._session_wins + self._session_losses
        pnl = sum(b.get("pnl_dollars", 0) for b in self._completed_bets)
        print(colored(f"\n{'='*70}", "cyan"))
        print(colored("  KALSHI DAEMON STOPPED", "cyan"))
        print(f"  Bets placed: {self._session_bets_placed}")
        print(f"  Wins: {self._session_wins} | Losses: {self._session_losses} | Total: {total}")
        if total > 0:
            print(f"  Win rate: {100*self._session_wins/total:.1f}%")
        print(f"  P&L: ${pnl:+.2f}")
        print(f"  Pending settlement: {len(self._pending_bets)}")
        print(colored(f"{'='*70}", "cyan"))


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
