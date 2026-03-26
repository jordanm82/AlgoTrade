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
from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3, KalshiV3Signal

# Intervals in seconds
TICK_INTERVAL = 60

# Kalshi multi-timeframe evaluation
KALSHI_CUTOFF_MINUTES = 13     # no new bets after this minute in the 15m window
KALSHI_LASTLOOK_MINUTE = 12    # elevated threshold window
KALSHI_THRESHOLD_BOOST = 10    # added to per-asset threshold for last-look


class KalshiDaemon:
    """Standalone Kalshi 15-minute prediction bot."""

    # Kalshi series tickers for each asset
    KALSHI_PAIRS = {
        "BTC/USDT": "KXBTC15M",
        "ETH/USDT": "KXETH15M",
        "SOL/USDT": "KXSOL15M",
        "XRP/USDT": "KXXRP15M",
        "BNB/USDT": "KXBNB15M",
    }

    # Per-asset minimum confidence thresholds (from 3-month backtest optimization)
    # Class A (threshold 30): BTC — strong at lower confidence
    # Class A (threshold 35): ETH, SOL, BNB — best WR at moderate confidence
    # Class B (threshold 30): XRP — degrades at higher thresholds
    KALSHI_THRESHOLDS = {
        "BTC/USDT": 30,
        "ETH/USDT": 35,
        "SOL/USDT": 35,
        "XRP/USDT": 30,
        "BNB/USDT": 35,
    }

    # Coinbase symbol mapping for V3 live price (closer to CF Benchmarks BRTI)
    COINBASE_PRICE_MAP = {
        "BTC/USDT": "BTC-USD",
        "ETH/USDT": "ETH-USD",
        "SOL/USDT": "SOL-USD",
        "XRP/USDT": "XRP-USD",
        "BNB/USDT": "BNB-USD",
    }

    def __init__(self, dry_run: bool = True, predictor_version: str = "v3"):
        self.dry_run = dry_run
        self.kalshi_predictor_version = predictor_version
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

        # Bet tracking — records ALL bets (live + dry-run) and checks settlement
        self._pending_bets: list[dict] = []
        self._completed_bets: list[dict] = []
        self._session_wins = 0
        self._session_losses = 0
        self._session_bets_placed = 0

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
        """Fetch 15m candles for Kalshi pairs only."""
        for symbol in self.KALSHI_PAIRS:
            try:
                df = self.fetcher.ohlcv(symbol, timeframe="15m", limit=200)
                df = add_indicators(df)
                self._dataframes[symbol] = df
            except Exception as e:
                print(colored(f"  [WARN] Fetch failed for {symbol}: {e}", "yellow"))

    # ------------------------------------------------------------------
    # Coinbase price
    # ------------------------------------------------------------------

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
    # Kalshi client
    # ------------------------------------------------------------------

    def _init_kalshi_client(self):
        """Lazy-initialize the Kalshi client.
        Always uses production API — V3 needs real strike prices even in dry-run.
        Dry-run only skips order placement, not market data queries."""
        if self.kalshi_client is not None:
            return
        try:
            from exchange.kalshi import KalshiClient
            from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
            self.kalshi_client = KalshiClient(
                api_key_id=KALSHI_API_KEY_ID,
                private_key_path=str(KALSHI_KEY_FILE),
                demo=False,  # always production — need real strikes/prices
            )
        except Exception as e:
            print(colored(f"  [WARN] Kalshi client init failed: {e}", "yellow"))

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
                # Find the settled market by series ticker
                asset = bet["asset"]
                series = self.KALSHI_PAIRS.get(bet["symbol"], "")
                if not series:
                    continue

                settled_markets = self.kalshi_client.get_markets(
                    series_ticker=series, status="settled"
                )

                # Find the market that matches our bet's settlement time
                for m in settled_markets:
                    ct = m.get("close_time", "")
                    if not ct:
                        continue
                    market_close = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    # Match within 1 minute of our expected settle time
                    if abs((market_close - bet["settle_time"]).total_seconds()) < 60:
                        result = m.get("result", "")  # "yes" or "no"
                        expiration_value = m.get("expiration_value", 0)

                        if not result:
                            continue

                        # Our bet wins if our side matches the result
                        won = bet["side"] == result
                        result_str = "WIN" if won else "LOSS"
                        color = "green" if won else "red"

                        # Calculate P&L
                        entry_price = bet.get("contract_price", 0)
                        count = bet.get("count", 1)
                        if won:
                            pnl_cents = count * (100 - entry_price)
                        else:
                            pnl_cents = -(count * entry_price)
                        pnl_dollars = pnl_cents / 100

                        print(colored(
                            f"  [SETTLED] {asset} {bet['direction']} "
                            f"@ {entry_price}c x{count} "
                            f"strike=${bet['strike']:,.2f} "
                            f"settled=${float(expiration_value):,.2f} "
                            f"-> {result_str} ${pnl_dollars:+.2f}",
                            color,
                        ))

                        bet["result"] = result_str
                        bet["settle_price"] = float(expiration_value)
                        bet["kalshi_result"] = result
                        bet["pnl_cents"] = pnl_cents
                        bet["pnl_dollars"] = pnl_dollars
                        self._completed_bets.append(bet)
                        if won:
                            self._session_wins += 1
                        else:
                            self._session_losses += 1
                        settled.append(bet)
                        break

            except Exception as e:
                print(colored(f"  [SETTLE ERR] {bet['asset']}: {e}", "yellow"))

        for bet in settled:
            self._pending_bets.remove(bet)

    # ------------------------------------------------------------------
    # Kalshi evaluation lifecycle
    # ------------------------------------------------------------------

    def _kalshi_eval(self):
        """Kalshi evaluation cycle with 15m scoring + fresh leading indicators at each eval.

        Lifecycle within each 15m window:
        - SETUP (min 0-4): Score on 15m candles + leading indicators. Cache 15m data. No betting.
        - OBSERVING (min 5-9): Re-fetch OB + trade flow. Re-score on cached 15m data.
        - CONFIRMED (min 10-11): Same as OBSERVING — refresh leading indicators, re-score. Place bets.
        - LAST_LOOK (min 12): 1m momentum check with elevated threshold.
        - EXPIRED (min 13+): No new bets.
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
            # Bet early — contracts are cheapest before minute 5
            # SETUP at first eval, CONFIRMED from second eval onward
            if minute_in_window <= 2:
                state = "SETUP"          # first eval — score + cache, no bet
            elif minute_in_window <= 11:
                state = "CONFIRMED"      # eligible to bet from minute 3+
            elif minute_in_window == KALSHI_LASTLOOK_MINUTE:
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

                # V3: re-run predict() at LAST_LOOK with updated time remaining
                if self.kalshi_predictor_version == "v3":
                    strike = pending.get("strike_price")
                    close_time_dt = pending.get("close_time")
                    if strike and close_time_dt:
                        mins_left = max(0, (close_time_dt - now_utc).total_seconds() / 60)
                        df_15m = self._kalshi_cached_dataframes.get(symbol)
                        if df_15m is not None and len(df_15m) >= 50:
                            cb_price = self._get_coinbase_price(symbol)
                            signal = self.kalshi_predictor.predict(
                                df_15m, strike_price=float(strike),
                                minutes_remaining=mins_left,
                                current_price=cb_price,
                            )
                            if signal and signal.recommended_side != "SKIP":
                                actionable_signals.append({
                                    "symbol": symbol, "series_ticker": series_ticker,
                                    "signal": signal, "market_data": None,
                                    "state": state,
                                })
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"],
                        "confidence": last_conf,
                        "reason": "last-look: re-evaluated with updated time",
                        "ob": 0, "flow": 0, "state": state,
                    })
                else:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"],
                        "confidence": last_conf,
                        "reason": "last-look: no action",
                        "ob": 0, "flow": 0, "state": state,
                    })
                continue

            # --- SETUP: 15m base scoring ---
            if state == "SETUP":
                # Fetch 15m candles for base scoring
                df_15m = None
                try:
                    df_15m = self.fetcher.ohlcv(symbol, "15m", limit=200)
                    if df_15m is not None and not df_15m.empty:
                        df_15m = add_indicators(df_15m)
                except Exception:
                    df_15m = None

                if df_15m is None or len(df_15m) < 50:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "no 15m data", "ob": 0, "flow": 0, "state": state,
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

                ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
                net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

                # Cache 15m data with indicators for CONFIRMED re-scoring
                self._kalshi_cached_dataframes[symbol] = df_15m

                if self.kalshi_predictor_version == "v3":
                    # V3: query Kalshi for strike price + time remaining
                    strike = None
                    close_time_dt = None
                    self._init_kalshi_client()
                    if self.kalshi_client:
                        try:
                            v3_markets = self.kalshi_client.get_markets(series_ticker=series_ticker, status="open")
                            if v3_markets:
                                v3_market = v3_markets[0]
                                strike = v3_market.get("floor_strike")
                                ct = v3_market.get("close_time") or v3_market.get("expiration_time", "")
                                if ct and "T" in ct:
                                    close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                        except Exception as e:
                            print(colored(f"  [V3] Kalshi API error for {asset}: {e}", "yellow"))

                    if strike and close_time_dt:
                        mins_left = max(0, (close_time_dt - now_utc).total_seconds() / 60)
                        # Use Coinbase price (closer to BRTI) instead of BinanceUS
                        cb_price = self._get_coinbase_price(symbol)
                        signal = self.kalshi_predictor.predict(
                            df_15m, strike_price=float(strike),
                            minutes_remaining=mins_left,
                            market_data=market_data, df_1h=df_1h,
                            current_price=cb_price,
                        )
                        # Store strike info in pending signal for V3
                        if signal and asset not in self._kalshi_pending_signals:
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
                                "base_conf": int(signal.probability * 100),
                                "last_5m_conf": int(signal.probability * 100),
                                "setup_time": now_utc,
                                "confirmed": False,
                            }
                    else:
                        signal = None

                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": (signal.recommended_side if signal and signal.recommended_side != "SKIP" else "--"),
                        "confidence": int(signal.probability * 100) if signal else 0,
                        "reason": "setup -- waiting for confirmation (V3 strike-relative)",
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

            # --- CONFIRMED: score and bet if signal is strong ---
            elif state == "CONFIRMED":
                if not pending and self.kalshi_predictor_version != "v3":
                    # No SETUP signal — nothing to re-score (V1/V2 only)
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": f"{state.lower()}: no SETUP signal",
                        "ob": 0, "flow": 0, "state": state,
                    })
                    continue

                # Re-fetch leading indicators (order book + trade flow shift every minute)
                market_data = None
                try:
                    ob = get_order_book_imbalance(symbol)
                    tf = get_trade_flow(symbol, limit=100)
                    market_data = {"order_book": ob, "trade_flow": tf}
                except Exception:
                    pass

                # Use cached 15m data, or fetch fresh if no cache (late start)
                df_15m = self._kalshi_cached_dataframes.get(symbol)
                if df_15m is None or len(df_15m) < 50:
                    try:
                        df_15m = self.fetcher.ohlcv(symbol, "15m", limit=200)
                        if df_15m is not None and not df_15m.empty:
                            df_15m = add_indicators(df_15m)
                            self._kalshi_cached_dataframes[symbol] = df_15m
                    except Exception:
                        pass
                if df_15m is None or len(df_15m) < 50:
                    predictions.append({
                        "symbol": symbol, "asset": asset,
                        "direction": pending["direction"] if pending else "--",
                        "confidence": 0,
                        "reason": f"{state.lower()}: no cached 15m data",
                        "ob": 0, "flow": 0, "state": state,
                    })
                    continue

                # Fetch 1h MTF data
                df_1h = None
                try:
                    df_1h = self.fetcher.ohlcv(symbol, "1h", limit=50)
                    if df_1h is not None and not df_1h.empty:
                        df_1h = add_indicators(df_1h)
                except Exception:
                    pass

                ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
                net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

                if self.kalshi_predictor_version == "v3":
                    # V3: get strike from pending or query Kalshi fresh
                    strike = pending.get("strike_price") if pending else None
                    close_time_dt = pending.get("close_time") if pending else None

                    if not strike:
                        # No SETUP ran — query Kalshi for strike (late start)
                        self._init_kalshi_client()
                        if self.kalshi_client:
                            try:
                                v3_markets = self.kalshi_client.get_markets(series_ticker=series_ticker, status="open")
                                if v3_markets:
                                    v3_market = v3_markets[0]
                                    strike = float(v3_market.get("floor_strike", 0))
                                    ct = v3_market.get("close_time") or v3_market.get("expiration_time", "")
                                    if ct and "T" in ct:
                                        close_time_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                            except Exception:
                                pass

                    if strike and close_time_dt:
                        mins_left = max(0, (close_time_dt - now_utc).total_seconds() / 60)
                        cb_price = self._get_coinbase_price(symbol)
                        signal = self.kalshi_predictor.predict(
                            df_15m, strike_price=float(strike),
                            minutes_remaining=mins_left,
                            market_data=market_data, df_1h=df_1h,
                            current_price=cb_price,
                        )
                        if signal and pending:
                            pending["probability"] = signal.probability
                            pending["recommended_side"] = signal.recommended_side
                            pending["max_price_cents"] = signal.max_price_cents
                            pending["last_5m_conf"] = int(signal.probability * 100)
                            pending["confirmed"] = True
                        elif signal and not pending:
                            # Create pending signal on the fly (late start)
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
                                "base_conf": int(signal.probability * 100),
                                "last_5m_conf": int(signal.probability * 100),
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
                        # Only bet at CONFIRMED (min 10+) or LAST_LOOK, not OBSERVING
                        if state in ("CONFIRMED", "LAST_LOOK"):
                            actionable_signals.append({
                                "symbol": symbol, "series_ticker": series_ticker,
                                "signal": signal, "market_data": market_data,
                                "state": state,
                            })
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": signal.recommended_side,
                            "confidence": int(signal.probability * 100),
                            "reason": f"{state.lower()}: V3 prob={signal.probability:.2f} side={signal.recommended_side}"
                                      + (" -> BETTING" if state in ("CONFIRMED", "LAST_LOOK") else " (observing)"),
                            "ob": ob_imb, "flow": net_flow, "state": state,
                        })
                    else:
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": "--",
                            "confidence": int(signal.probability * 100) if hasattr(signal, 'probability') else 0,
                            "reason": f"{state.lower()}: V3 SKIP (prob={signal.probability:.2f})",
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

        # Execute top actionable signals by confidence (V3 uses probability, V1/V2 use confidence)
        def _signal_score(x):
            sig = x["signal"]
            if hasattr(sig, 'probability'):
                return sig.probability
            return getattr(sig, 'confidence', 0) / 100
        actionable_signals.sort(key=_signal_score, reverse=True)
        for vs in actionable_signals:
            max_bets = 4 if self.dry_run else MAX_CONCURRENT_KALSHI_BETS
            if len(self._active_kalshi_bets) >= max_bets:
                break
            self._kalshi_execute_bet(
                vs["symbol"], vs["series_ticker"], vs["signal"], vs["market_data"]
            )
            # Mark bet as placed in pending signals
            asset = vs["symbol"].split("/")[0]
            if asset in self._kalshi_pending_signals:
                self._kalshi_pending_signals[asset]["bet_placed"] = True

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
            conf_display = int(signal.probability * 100)
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

        if self.dry_run:
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

            # Calculate position size using Kelly criterion
            # Get balance from Kalshi (or simulate)
            dry_balance_cents = 10000  # default $100 simulated balance
            try:
                if self.kalshi_client:
                    bal = self.kalshi_client.get_balance()
                    dry_balance_cents = bal.get("balance", 10000)
            except Exception:
                pass

            if isinstance(signal, KalshiV3Signal) and contract_price > 0:
                count = KalshiPredictorV3.kelly_size(
                    probability=signal.probability,
                    contract_price_cents=contract_price,
                    balance_cents=dry_balance_cents,
                    fraction=0.5,  # half-Kelly for safety
                )
                count = max(1, count)  # minimum 1 contract
            else:
                risk_budget = int(dry_balance_cents * 0.05)
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
            risk_budget_cents = max(500, int(balance_cents * RISK_PER_BET_PCT))

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

            # Pick the soonest-expiring open market
            market = all_markets[0]
            ticker = market.get("ticker", "")

            # Optimal entry: query orderbook for bid-ask spread
            best_bid = None
            best_ask = None
            try:
                book = self.kalshi_client.get_orderbook(ticker)
                ob = book.get("orderbook", {})
                our_side = "yes" if side == "yes" else "no"
                asks = ob.get(our_side, [])
                if asks and len(asks) > 0:
                    best_ask = asks[0][0] if isinstance(asks[0], list) else asks[0]
                # Get the bid side too for spread calculation
                other_side = "no" if side == "yes" else "yes"
                bids = ob.get(other_side, [])
                if bids and len(bids) > 0:
                    other_best = bids[0][0] if isinstance(bids[0], list) else bids[0]
                    best_bid = 100 - other_best  # complement
            except Exception:
                pass

            # Fall back to market data if orderbook failed
            if best_ask is None:
                if side == "yes":
                    best_ask = market.get("yes_ask") or 50
                else:
                    best_ask = market.get("no_ask") or 50

            best_ask = int(best_ask) if best_ask else 50

            # Calculate optimal bid
            if best_bid is not None:
                best_bid = int(best_bid)
                spread = best_ask - best_bid
                if spread <= 3:
                    # Tight spread — bid at ask for instant fill
                    fill_price = best_ask
                elif spread <= 10:
                    # Moderate spread — bid at midpoint + 1c
                    fill_price = (best_bid + best_ask) // 2 + 1
                else:
                    # Wide spread — bid 30% above mid (aggressive but not at ask)
                    mid = (best_bid + best_ask) // 2
                    fill_price = mid + max(1, spread // 3)
            else:
                # No bid info — bid at ask
                fill_price = best_ask

            fill_price = max(1, min(99, int(fill_price)))

            # NEVER pay above max entry cap
            if fill_price > MAX_ENTRY_CENTS:
                pred["reason"] = f"price {fill_price}c > max {MAX_ENTRY_CENTS}c -- R:R unfavorable"
                print(colored(
                    f"  [KALSHI SKIP] {asset} {side.upper()} @ {fill_price}c "
                    f"-- above {MAX_ENTRY_CENTS}c cap (R:R < 1:1)",
                    "yellow"))
                return pred

            # Position size: Kelly criterion for V3, flat % for V1/V2
            if isinstance(signal, KalshiV3Signal) and fill_price > 0:
                count = KalshiPredictorV3.kelly_size(
                    probability=signal.probability,
                    contract_price_cents=fill_price,
                    balance_cents=balance_cents,
                    fraction=0.5,  # half-Kelly
                )
                count = max(1, count)
            else:
                count = max(1, risk_budget_cents // fill_price)
            potential_profit = count * (100 - fill_price)
            potential_loss = count * fill_price
            rr_ratio = potential_profit / potential_loss if potential_loss > 0 else 0

            # Verify against actual Kalshi balance (exchange is source of truth)
            cost_cents = count * fill_price
            if cost_cents > balance_cents:
                count = max(1, balance_cents // fill_price)
                if count < 1:
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

            # Track for settlement (both live and dry-run)
            if isinstance(signal, KalshiV3Signal):
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
                        "count": count,
                        "live": True,
                    })

        except Exception as e:
            pred["reason"] = f"order failed: {e}"
            print(colored(f"  [KALSHI ERR] {asset}: {e}", "red"))

        return pred

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def startup(self):
        """Fetch initial data and run first eval."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
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
        print(colored(f"{'='*70}", "cyan"))

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
        # Check settlements
        self._check_dryrun_settlements()

        # Wall-clock aligned Kalshi eval at :01, :06, :11 + LAST_LOOK at :12/:27/:42/:57
        now = time.time()
        current_minute = datetime.now(timezone.utc).minute
        should_eval = (current_minute % 5 == 1 and now - self._last_kalshi_eval >= 240) \
                   or (current_minute % 15 == 12 and now - self._last_kalshi_eval >= 50) \
                   or (current_minute % 15 == 1 and now - self._last_kalshi_eval >= 50)
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
        mode_tag = colored("[DRY-RUN]", "magenta") if self.dry_run else colored("[LIVE]", "green")

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
    args = parser.parse_args()

    daemon = KalshiDaemon(dry_run=args.dry_run, predictor_version=args.predictor)
    daemon.run(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
