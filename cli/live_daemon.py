#!/usr/bin/env python3
"""Production trading daemon — runs validated strategies on 15m timeframe.

Usage:
    python -m cli.live_daemon              # live trading
    python -m cli.live_daemon --dry-run    # print signals only
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from termcolor import colored

from config.pair_config import (
    ALL_PAIRS,
    COINBASE_MAP,
    PAIR_CONFIG,
    get_pair_config,
)
from config.production import (
    MAX_CONCURRENT_POSITIONS,
    MAX_CONCURRENT_KALSHI_BETS,
    POSITION_SIZE_PCT,
    STOP_LOSS_PCT,
)

# Derive leverage pairs from per-pair config (any pair with leverage > 1)
LEVERAGE_PAIRS = [sym for sym, cfg in PAIR_CONFIG.items() if cfg["leverage"] > 1]
from config.settings import CDP_KEY_FILE, DATA_DIR
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from data.store import DataStore
from exchange.coinbase import CoinbaseExecutor
from exchange.positions import PositionTracker
from risk.manager import RiskManager

# Pairs eligible for funding rate arbitrage (Coinbase perps)
FUNDING_ARB_PERPS = {
    "BTC-PERP-INTX": "BTC/USDT",
    "ETH-PERP-INTX": "ETH/USDT",
    "SOL-PERP-INTX": "SOL/USDT",
}
# Funding rate thresholds
FUNDING_HIGH_THRESHOLD = 0.0003   # 0.03% per hour = ~26% annualized
FUNDING_LOW_THRESHOLD = -0.0001   # -0.01% per hour

# Intervals in seconds
TICK_INTERVAL = 60       # price update + stop enforcement
SIGNAL_INTERVAL = 900    # 15 minutes — matches candle timeframe
SNAPSHOT_INTERVAL = 900  # save snapshot every 15 minutes

# Kalshi multi-timeframe evaluation
KALSHI_CUTOFF_MINUTES = 13     # no new bets after this minute in the 15m window
KALSHI_LASTLOOK_MINUTE = 12    # elevated threshold window
KALSHI_THRESHOLD_BOOST = 10    # added to per-asset threshold for last-look


class LiveDaemon:
    """Production daemon that runs BB Grid + RSI MR on validated pairs."""

    def __init__(self, dry_run: bool = False, kalshi_only: bool = False, predictor_version: str = "v1"):
        self.dry_run = dry_run
        self.kalshi_only = kalshi_only
        self.kalshi_predictor_version = predictor_version
        self.fetcher = DataFetcher()
        self.store = DataStore(DATA_DIR)
        self.tracker = PositionTracker(max_concurrent=MAX_CONCURRENT_POSITIONS)
        self.tracker.load_state()  # restore positions from previous session
        self._running = False

        # Dataframes keyed by binance symbol (e.g. "ATOM/USDT")
        self._dataframes: dict[str, pd.DataFrame] = {}

        # Track equity for compounding position sizing
        self._equity: float = 0.0

        if dry_run:
            self.executor = None
            self.risk = RiskManager(portfolio_value=10_000.0)
            self._equity = 10_000.0
            self.risk.record_daily_start(self._equity)
        else:
            self.executor = CoinbaseExecutor(str(CDP_KEY_FILE))
            balances = self.executor.get_balances()
            self._equity = sum(balances.values())
            self.risk = RiskManager(portfolio_value=self._equity)
            self.risk.record_daily_start(self._equity)

        # Trade counters for status line
        self._trades_today = 0
        self._signals_today = 0
        self._pnl_today = 0.0

        # Kalshi predictor + client
        if predictor_version == "v3":
            from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3
            self.kalshi_predictor = KalshiPredictorV3()
        elif predictor_version == "v2":
            from strategy.strategies.kalshi_predictor_v2 import KalshiPredictorV2
            self.kalshi_predictor = KalshiPredictorV2()
        else:
            from strategy.strategies.kalshi_predictor import KalshiPredictor
            self.kalshi_predictor = KalshiPredictor()
        self.kalshi_client = None  # lazy init
        self.kalshi_threshold = 30  # minimum confidence to bet (lowered from 40 — conf 25-35 with flow confirmation is the sweet spot)
        self.kalshi_predictions: list[dict] = []  # latest predictions for dashboard
        self._active_kalshi_bets = {}  # {ticker: placement_time}
        self._kalshi_pending_signals = {}   # {asset: {direction, base_conf, last_5m_conf, ...}}
        self._last_kalshi_eval = 0          # timestamp of last eval
        # Dry-run bet tracking — records bets and checks settlement
        self._dryrun_bets: list[dict] = []  # pending bets awaiting settlement
        self._dryrun_results: list[dict] = []  # completed bets with W/L
        self._dryrun_wins = 0
        self._dryrun_losses = 0
        self._kalshi_cached_dataframes = {}  # {symbol: DataFrame} cached 15m data with indicators for CONFIRMED re-scoring

    # ------------------------------------------------------------------
    # Position sync from exchanges
    # ------------------------------------------------------------------

    def _sync_positions_from_exchange(self):
        """On startup, check Coinbase for any token balances we're not tracking
        and Kalshi for open positions.  This catches positions from previous
        sessions that weren't saved."""
        if self.dry_run or not self.executor:
            return

        # --- Coinbase spot balances ---
        try:
            balances = self.executor.get_balances()
            for currency, token_amount in balances.items():
                if currency in ("USD", "USDC", "USDT"):
                    continue
                # Get current price to compute USD value
                binance_sym = f"{currency}/USDT"
                try:
                    ticker = self.fetcher.ticker(binance_sym)
                    price = ticker.get("last", 0)
                except Exception:
                    price = 0
                if price <= 0:
                    continue
                usd_value = token_amount * price
                if usd_value < 1.0:  # skip dust (< $1)
                    continue
                # Check if we have a tracked position for this currency
                coinbase_sym = f"{currency}-USD"
                tracked = any(coinbase_sym in p["symbol"] for p in self.tracker.open_positions())
                if not tracked:
                    print(colored(
                        f"  [SYNC] Found untracked {currency}: {token_amount:.6f} "
                        f"(${usd_value:.2f} @ ${price:.2f}) — adding to tracker",
                        "yellow"))
                    pos_key = f"{coinbase_sym}:synced:long"
                    self.tracker.open(pos_key, "BUY", usd_value, price, price * 0.97, price * 1.10)
        except Exception as e:
            print(colored(f"  [WARN] Coinbase sync failed: {e}", "yellow"))

        # --- Kalshi open positions ---
        try:
            self._init_kalshi_client()
            if self.kalshi_client is not None:
                positions = self.kalshi_client.get_positions()
                if positions:
                    print(colored(f"  [SYNC] Found {len(positions)} open Kalshi positions", "yellow"))
                    for pos in positions:
                        ticker = pos.get("ticker", pos.get("market_ticker", "unknown"))
                        side = "yes" if pos.get("total_traded", 0) > 0 or pos.get("position", 0) > 0 else "no"
                        contracts = abs(pos.get("total_traded", 0) or pos.get("position", 0))
                        cost_cents = abs(pos.get("realized_pnl", 0) + pos.get("market_exposure", 0))
                        print(colored(
                            f"  [SYNC] Kalshi: {ticker} | {side} x{contracts} | cost={cost_cents}c",
                            "yellow",
                        ))
        except Exception as e:
            print(colored(f"  [WARN] Kalshi sync failed: {e}", "yellow"))

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_pair(self, symbol: str) -> pd.DataFrame | None:
        """Fetch 15m OHLCV for a single pair and add indicators."""
        try:
            df = self.fetcher.ohlcv(symbol, timeframe="15m", limit=200)
            df = add_indicators(df)
            return df
        except Exception as e:
            print(colored(f"  [WARN] Fetch failed for {symbol}: {e}", "yellow"))
            return None

    def _fetch_all(self):
        """Fetch 15m data for all monitored pairs."""
        for symbol in ALL_PAIRS:
            df = self._fetch_pair(symbol)
            if df is not None:
                self._dataframes[symbol] = df

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _bb_grid_signals(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Check BB Grid Long+Short signals on a single pair."""
        cfg = get_pair_config(symbol)
        if "bb_grid" not in cfg.get("enabled_strategies", []):
            return []

        if len(df) < 25:
            return []

        last = df.iloc[-1]
        close = float(last["close"])
        rsi = last.get("rsi")
        bb_lower = last.get("bb_lower")
        bb_mid = last.get("bb_middle")
        bb_upper = last.get("bb_upper")
        atr = last.get("atr")

        if any(pd.isna(v) for v in [rsi, bb_lower, bb_mid, bb_upper, atr]):
            return []

        rsi = float(rsi)
        bb_lower = float(bb_lower)
        bb_mid = float(bb_mid)
        bb_upper = float(bb_upper)
        atr = float(atr)

        signals = []
        buy_thresh = cfg["bb_rsi_buy"]
        short_thresh = cfg["bb_rsi_short"]
        leverage = cfg["leverage"]

        coinbase_sym = COINBASE_MAP[symbol]

        # BUY signal: close < BB lower AND RSI < threshold
        if close < bb_lower and rsi < buy_thresh:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "BUY",
                "strategy": f"bb_grid_{leverage}x",
                "leverage": leverage,
                "price": close,
                "rsi": rsi,
                "bb_lower": bb_lower,
                "bb_mid": bb_mid,
                "stop": close * (1 - STOP_LOSS_PCT),
                "take_profit": bb_mid,
                "atr": atr,
            })

        # SELL (exit long): close > BB mid — only if we have a long position
        if close > bb_mid:
            pos_key = f"{coinbase_sym}:bb_grid:long"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_LONG",
                    "strategy": f"bb_grid_{leverage}x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                    "bb_mid": bb_mid,
                })

        # SHORT signal: close > BB upper AND RSI > threshold
        if close > bb_upper and rsi > short_thresh:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "SHORT",
                "strategy": f"bb_grid_{leverage}x",
                "leverage": leverage,
                "price": close,
                "rsi": rsi,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "stop": close * (1 + STOP_LOSS_PCT),
                "take_profit": bb_mid,
                "atr": atr,
            })

        # COVER (exit short): close < BB mid — only if we have a short position
        if close < bb_mid:
            pos_key = f"{coinbase_sym}:bb_grid:short"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_SHORT",
                    "strategy": f"bb_grid_{leverage}x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                    "bb_mid": bb_mid,
                })

        return signals

    def _rsi_mr_signals(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Check RSI Mean Reversion signals using per-pair thresholds."""
        cfg = get_pair_config(symbol)
        if "rsi_mr" not in cfg.get("enabled_strategies", []):
            return []
        if len(df) < 25:
            return []

        last = df.iloc[-1]
        close = float(last["close"])
        rsi = last.get("rsi")
        atr = last.get("atr")

        if any(pd.isna(v) for v in [rsi, atr]):
            return []

        rsi = float(rsi)
        atr = float(atr)

        signals = []
        coinbase_sym = COINBASE_MAP[symbol]
        leverage = cfg["leverage"]

        # BUY: RSI < oversold
        if rsi < cfg["rsi_mr_oversold"]:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "BUY",
                "strategy": f"rsi_mr_{leverage}x",
                "leverage": leverage,
                "price": close,
                "rsi": rsi,
                "stop": close * (1 - STOP_LOSS_PCT),
                "take_profit": close + atr * 3,
                "atr": atr,
            })

        # SELL (exit long): RSI > exit_long
        if rsi > cfg["rsi_mr_exit_long"]:
            pos_key = f"{coinbase_sym}:rsi_mr:long"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_LONG",
                    "strategy": f"rsi_mr_{leverage}x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                })

        # SHORT: RSI > overbought
        if rsi > cfg["rsi_mr_overbought"]:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "SHORT",
                "strategy": f"rsi_mr_{leverage}x",
                "leverage": leverage,
                "price": close,
                "rsi": rsi,
                "stop": close * (1 + STOP_LOSS_PCT),
                "take_profit": close - atr * 3,
                "atr": atr,
            })

        # COVER (exit short): RSI < exit_short
        if rsi < cfg["rsi_mr_exit_short"]:
            pos_key = f"{coinbase_sym}:rsi_mr:short"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_SHORT",
                    "strategy": f"rsi_mr_{leverage}x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                })

        return signals

    def _funding_arb_signals(self) -> list[dict]:
        """Check funding rates on Coinbase perps for arb opportunities."""
        signals = []
        try:
            from coinbase.rest import RESTClient
            client = RESTClient()  # no auth needed for public product data
            for perp_sym, spot_sym in FUNDING_ARB_PERPS.items():
                try:
                    product = client.get_public_product(perp_sym)
                    details = product.get("future_product_details") or {}
                    perp_details = details.get("perpetual_details") or {}
                    rate_str = perp_details.get("funding_rate", "0")
                    rate = float(rate_str)
                    # Coinbase uses 1-hour funding intervals
                    annualized = rate * 24 * 365 * 100

                    coinbase_spot = spot_sym.replace("/USDT", "-USD")
                    price = self._live_prices.get(spot_sym, 0) if hasattr(self, '_live_prices') else 0
                    if price == 0:
                        df = self._dataframes.get(spot_sym)
                        if df is not None and len(df) > 0:
                            price = float(df.iloc[-1]["close"])

                    if rate > FUNDING_HIGH_THRESHOLD:
                        # Longs paying shorts — long spot, short perp to collect
                        signals.append({
                            "symbol": spot_sym,
                            "coinbase_symbol": coinbase_spot,
                            "action": "BUY",
                            "strategy": f"funding_arb ({annualized:.0f}% ann)",
                            "leverage": 1,
                            "price": price,
                            "rsi": 50,  # neutral — not RSI-driven
                            "stop": price * 0.97,
                            "take_profit": price * 1.03,
                            "atr": 0,
                            "funding_rate": rate,
                            "annualized_pct": annualized,
                        })
                    elif rate < FUNDING_LOW_THRESHOLD:
                        # Shorts paying longs — long perp, short spot
                        signals.append({
                            "symbol": spot_sym,
                            "coinbase_symbol": coinbase_spot,
                            "action": "SHORT",
                            "strategy": f"funding_arb ({annualized:.0f}% ann)",
                            "leverage": 1,
                            "price": price,
                            "rsi": 50,
                            "stop": price * 1.03,
                            "take_profit": price * 0.97,
                            "atr": 0,
                            "funding_rate": rate,
                            "annualized_pct": annualized,
                        })
                except Exception as e:
                    print(colored(f"  [WARN] Funding rate fetch failed for {perp_sym}: {e}", "yellow"))
        except ImportError:
            pass
        return signals

    def _signal_confidence(self, sig: dict) -> float:
        """Score a signal's confidence for priority ranking.
        Higher = more confident. Used to decide which signals get the limited slots."""
        score = 0.0
        rsi = sig.get("rsi", 50)
        action = sig.get("action", "")

        # RSI extremity — further from 50 = more confident
        if action in ("BUY", "CLOSE_SHORT"):
            score += max(0, (35 - rsi) * 2)  # RSI 20 scores 30, RSI 30 scores 10
        elif action in ("SHORT", "CLOSE_LONG"):
            score += max(0, (rsi - 65) * 2)  # RSI 80 scores 30, RSI 70 scores 10

        # Leverage pairs get a bonus (better backtested)
        if sig.get("symbol", "") in LEVERAGE_PAIRS:
            score += 10

        # Funding arb gets a high score (delta-neutral, lowest risk)
        if "funding_arb" in sig.get("strategy", ""):
            score += 25

        # Close signals always take priority (protect capital)
        if "CLOSE" in action:
            score += 100

        # Confluence bonus — if both strategies fire on same pair, caller handles this
        sig["_confidence"] = round(score, 1)
        return score

    def _collect_all_signals(self) -> list[dict]:
        """Collect all signals, score by confidence, sort highest first.
        Only iterates Coinbase trading pairs (ALL_PAIRS), not Kalshi-only pairs."""
        all_signals = []
        for symbol in ALL_PAIRS:
            df = self._dataframes.get(symbol)
            if df is None or len(df) < 20:
                continue
            all_signals.extend(self._bb_grid_signals(symbol, df))
            all_signals.extend(self._rsi_mr_signals(symbol, df))
        # Add funding rate arb signals
        all_signals.extend(self._funding_arb_signals())

        # Score and sort: closes first, then highest confidence
        for sig in all_signals:
            self._signal_confidence(sig)
        all_signals.sort(key=lambda s: s.get("_confidence", 0), reverse=True)

        return all_signals

    # ------------------------------------------------------------------
    # Leading indicator gate
    # ------------------------------------------------------------------

    def _check_leading_indicators(self, symbol: str, direction: str) -> tuple[bool, str]:
        """Check order book + trade flow before executing a trade.
        Returns (should_trade, reason).

        Rules:
        - If order book imbalance strongly contradicts direction (>0.3 against), SKIP
        - If trade flow strongly contradicts (net_flow >0.2 against AND buy_ratio against), SKIP
        - If both order book AND trade flow confirm direction, log as HIGH CONFLUENCE
        """
        try:
            from data.market_data import get_order_book_imbalance, get_trade_flow
            ob = get_order_book_imbalance(symbol)
            tf = get_trade_flow(symbol, limit=100)

            imbalance = ob["imbalance"]
            net_flow = tf["net_flow"]

            if direction == "BUY":
                # Skip if strong sell pressure
                if imbalance < -0.3 and net_flow < -0.15:
                    return False, f"BLOCKED: sell pressure (OB={imbalance:+.2f}, flow={net_flow:+.2f})"
                if net_flow < -0.3:
                    return False, f"BLOCKED: heavy selling (flow={net_flow:+.2f})"
                if imbalance > 0.15 and net_flow > 0.1:
                    return True, f"CONFIRMED: buy pressure (OB={imbalance:+.2f}, flow={net_flow:+.2f})"
            elif direction == "SHORT" or direction == "SELL":
                # Skip if strong buy pressure
                if imbalance > 0.3 and net_flow > 0.15:
                    return False, f"BLOCKED: buy pressure (OB={imbalance:+.2f}, flow={net_flow:+.2f})"
                if net_flow > 0.3:
                    return False, f"BLOCKED: heavy buying (flow={net_flow:+.2f})"
                if imbalance < -0.15 and net_flow < -0.1:
                    return True, f"CONFIRMED: sell pressure (OB={imbalance:+.2f}, flow={net_flow:+.2f})"

            return True, "NEUTRAL"  # no strong signal either way, proceed
        except Exception as e:
            return True, f"SKIP_CHECK: {e}"  # if data fetch fails, proceed anyway

    # ------------------------------------------------------------------
    # Kalshi prediction cycle
    # ------------------------------------------------------------------

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

    def _get_coinbase_price(self, symbol: str) -> float | None:
        """Get live price from Coinbase (closer to BRTI than BinanceUS)."""
        try:
            from coinbase.rest import RESTClient
            from config.settings import CDP_KEY_FILE
            cb_symbol = self.COINBASE_PRICE_MAP.get(symbol)
            if not cb_symbol:
                return None
            client = RESTClient(key_file=str(CDP_KEY_FILE))
            product = client.get_product(cb_symbol)
            return float(product["price"])
        except Exception:
            return None

    def _check_dryrun_settlements(self):
        """Check if any dry-run bets have settled and record W/L."""
        if not self._dryrun_bets:
            return
        now = datetime.now(timezone.utc)
        settled = []
        for bet in self._dryrun_bets:
            if now >= bet["settle_time"]:
                # Get actual price from Coinbase at settlement
                cb_price = self._get_coinbase_price(bet["symbol"])
                if cb_price is None:
                    continue  # can't check yet, try next tick

                above_strike = cb_price >= bet["strike"]
                # YES wins if price >= strike, NO wins if price < strike
                won = (bet["side"] == "yes" and above_strike) or \
                      (bet["side"] == "no" and not above_strike)

                result_str = "WIN" if won else "LOSS"
                color = "green" if won else "red"
                print(colored(
                    f"  [KALSHI DRY SETTLED] {bet['asset']} {bet['direction']} "
                    f"strike=${bet['strike']:,.2f} actual=${cb_price:,.2f} → {result_str}",
                    color,
                ))

                bet["result"] = result_str
                bet["settle_price"] = cb_price
                self._dryrun_results.append(bet)
                if won:
                    self._dryrun_wins += 1
                else:
                    self._dryrun_losses += 1
                settled.append(bet)

        for bet in settled:
            self._dryrun_bets.remove(bet)

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

    def _kalshi_eval(self):
        """Kalshi evaluation cycle with 15m scoring + fresh leading indicators at each eval.

        Lifecycle within each 15m window:
        - SETUP (min 0-4): Score on 15m candles + leading indicators. Cache 15m data. No betting.
        - CONFIRMED (min 5-9): Re-fetch OB + trade flow. Re-score on cached 15m data.
        - DOUBLE_CONFIRMED (min 10-11): Same as CONFIRMED — refresh leading indicators, re-score.
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
                        "reason": "bet placed — awaiting settlement",
                        "ob": 0, "flow": 0, "state": "SETTLING",
                    })
                else:
                    expired_preds.append({
                        "symbol": sym, "asset": asset,
                        "direction": "--", "confidence": 0,
                        "reason": "expired — too close to settlement",
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
            if minute_in_window <= 4:
                state = "SETUP"
            elif minute_in_window <= 9:
                state = "CONFIRMED"
            elif minute_in_window <= 11:
                state = "DOUBLE_CONFIRMED"
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

                # Fetch 1m candles for momentum check
                try:
                    df_1m = self.fetcher.ohlcv(symbol, "1m", limit=5)
                except Exception:
                    df_1m = None

                if df_1m is not None and self.kalshi_predictor.check_1m_momentum(df_1m, pending["direction"]):
                    # Momentum confirmed — actionable
                    from strategy.strategies.kalshi_predictor import KalshiSignal
                    # Use last known price from cached 15m data
                    cached_15m = self._kalshi_cached_dataframes.get(symbol)
                    last_price = float(cached_15m.iloc[-1]["close"]) if cached_15m is not None and len(cached_15m) > 0 else 0
                    fake_signal = KalshiSignal(
                        asset=asset, direction=pending["direction"],
                        confidence=last_conf, components={},
                        price=last_price, rsi=0,
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
                        "reason": "last-look: 1m momentum confirmed, taking bet",
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
                        "reason": "setup — waiting for confirmation (V3 strike-relative)",
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
                        "reason": "setup — waiting for confirmation (leading indicators will refresh)",
                        "ob": ob_imb, "flow": net_flow, "state": state,
                    })

            # --- CONFIRMED / DOUBLE_CONFIRMED: re-score 15m with fresh leading indicators ---
            elif state in ("CONFIRMED", "DOUBLE_CONFIRMED"):
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
                    from strategy.strategies.kalshi_predictor_v3 import KalshiV3Signal
                    if isinstance(signal, KalshiV3Signal) and signal.recommended_side != "SKIP":
                        pending["confirmed"] = True
                        actionable_signals.append({
                            "symbol": symbol, "series_ticker": series_ticker,
                            "signal": signal, "market_data": market_data,
                            "state": state,
                        })
                        predictions.append({
                            "symbol": symbol, "asset": asset,
                            "direction": signal.recommended_side,
                            "confidence": int(signal.probability * 100),
                            "reason": f"{state.lower()}: V3 prob={signal.probability:.2f} side={signal.recommended_side} (fresh OB/flow)",
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

    def _kalshi_execute_bet(self, symbol: str, series_ticker: str, signal, market_data: dict | None) -> dict:
        """Execute a single Kalshi bet. Returns a pred dict with outcome details.

        Handles: balance check, market discovery, orderbook pricing,
        order placement, fill verification, active bet tracking.
        """
        asset = symbol.split("/")[0]
        ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
        net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

        # Detect V3 signal type and extract side/confidence accordingly
        from strategy.strategies.kalshi_predictor_v3 import KalshiV3Signal
        if isinstance(signal, KalshiV3Signal):
            direction_label = signal.recommended_side  # "YES" or "NO"
            side = "yes" if signal.recommended_side == "YES" else "no"
            conf_display = int(signal.probability * 100)
            MAX_ENTRY_CENTS = min(50, signal.max_price_cents) if signal.max_price_cents > 0 else 50
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
            # Record dry-run bet for settlement tracking
            from strategy.strategies.kalshi_predictor_v3 import KalshiV3Signal
            strike = 0
            settle_time = None
            if isinstance(signal, KalshiV3Signal):
                strike = signal.strike_price
                pending = self._kalshi_pending_signals.get(asset, {})
                settle_time = pending.get("close_time")

            if strike and settle_time:
                self._dryrun_bets.append({
                    "asset": asset,
                    "symbol": symbol,
                    "side": side,
                    "direction": direction_label,
                    "strike": strike,
                    "confidence": conf_display,
                    "bet_time": datetime.now(timezone.utc),
                    "settle_time": settle_time,
                })

            print(colored(
                f"  [KALSHI DRY] {asset} {direction_label} "
                f"conf={conf_display} | "
                f"OB={ob_imb:+.2f} flow={net_flow:+.2f} | "
                f"bet {side.upper()} strike=${strike:,.2f} (hold to settlement)",
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
            # Tight spread → bid at ask (instant fill)
            # Wide spread → bid at midpoint+1c (better price, likely fills)
            best_bid = None
            best_ask = None
            try:
                book = self.kalshi_client.get_orderbook(ticker)
                ob = book.get("orderbook", {})
                our_side = "yes" if side == "yes" else "no"
                asks = ob.get(our_side, [])
                # Orderbook format: [[price, qty], ...]
                # For YES side: asks are offers to sell YES contracts
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

            # NEVER pay above 50c — above that the R:R is against us
            if fill_price > MAX_ENTRY_CENTS:
                pred["reason"] = f"price {fill_price}c > max {MAX_ENTRY_CENTS}c — R:R unfavorable"
                print(colored(
                    f"  [KALSHI SKIP] {asset} {side.upper()} @ {fill_price}c "
                    f"— above {MAX_ENTRY_CENTS}c cap (R:R < 1:1)",
                    "yellow"))
                return pred

            # Position size from risk budget: count = max_loss / entry_price
            # entry_price IS the max loss per contract (we hold to settlement)
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
                order_type="market",
            )
            order = result.get("order", {})
            order_id = order.get("order_id", "?")
            fill_count = order.get("fill_count_fp", "0")
            order_status = order.get("status", "?")

            # Verify fill via Kalshi (exchange is source of truth)
            if float(fill_count) < count and order_status == "resting":
                # Partially filled or resting — check positions
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
        except Exception as e:
            pred["reason"] = f"order failed: {e}"
            print(colored(f"  [KALSHI ERR] {asset}: {e}", "red"))

        return pred

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _position_size(self, leverage: int = 1) -> float:
        """Calculate position size (margin): 10% of current equity.
        Leverage amplifies P&L but the margin (capital at risk) stays at 10%."""
        return self._equity * POSITION_SIZE_PCT

    def _has_position(self, pos_key: str) -> bool:
        """Check if we already have a position with this key."""
        return pos_key in [p["symbol"] for p in self.tracker.open_positions()]

    def _execute_signal(self, sig: dict):
        """Execute a single signal (or log it in dry-run mode)."""
        action = sig["action"]
        symbol = sig["symbol"]
        coinbase_sym = sig["coinbase_symbol"]
        strategy = sig["strategy"]
        price = sig["price"]

        if action == "BUY":
            side_key = "long"
            strat_key = "bb_grid" if "bb_grid" in strategy else "rsi_mr"
            pos_key = f"{coinbase_sym}:{strat_key}:{side_key}"

            if self._has_position(pos_key):
                return  # already have a position for this pair+strategy+side

            if not self.tracker.can_open():
                print(colored(f"  [SKIP] {symbol} BUY — max positions reached", "yellow"))
                return

            leverage = sig.get("leverage", 1)
            size_usd = self._position_size(leverage)
            stop = sig["stop"]
            take_profit = sig["take_profit"]

            ok, reason = self.risk.check_entry(
                size_usd=size_usd, leverage=leverage,
                current_positions=len(self.tracker.open_positions()),
            )
            if not ok:
                print(colored(f"  [RISK] {symbol} BUY blocked: {reason}", "yellow"))
                return

            # Leading indicator gate
            should_trade, li_reason = self._check_leading_indicators(symbol, "BUY")
            if "CONFIRMED" in li_reason:
                print(colored(f"  [LEAD] {symbol} BUY {li_reason}", "green"))
            elif "BLOCKED" in li_reason:
                print(colored(f"  [LEAD] {symbol} BUY {li_reason}", "red"))
            else:
                print(colored(f"  [LEAD] {symbol} BUY {li_reason}", "yellow"))
            time.sleep(0.2)  # rate limit: 2 extra API calls
            if not should_trade:
                return

            self._signals_today += 1

            if self.dry_run:
                print(colored(
                    f"  [DRY-RUN BUY] {symbol} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy} | "
                    f"size=${size_usd:.0f} stop=${stop:.4f}",
                    "green",
                ))
                # Still track the position for dry-run simulation
                self.tracker.open(pos_key, "BUY", size_usd, price, stop, take_profit)
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": coinbase_sym, "side": "BUY", "strategy": strategy,
                    "size_usd": size_usd, "price": price, "stop": stop,
                    "leverage": leverage, "source": "live_daemon_dry",
                })
                return  # skip exchange execution only

            # Execute: spot buy via Coinbase
            result = self.executor.market_buy(coinbase_sym, size_usd)
            if result.get("success") or result.get("order_id") or result.get("success_response"):
                order_id = (result.get("success_response") or {}).get("order_id", "")

                # Verify the order actually filled by checking Coinbase balance
                time.sleep(1)  # brief wait for settlement
                currency = coinbase_sym.replace("-USD", "")
                actual_balance = self.executor.get_token_balance(currency)
                if actual_balance > 0:
                    # Use actual balance to compute USD value
                    actual_usd = actual_balance * price  # approximate
                    self.tracker.open(pos_key, "BUY", actual_usd, price, stop, take_profit)
                    self._trades_today += 1
                    self.store.append_trade({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": coinbase_sym, "side": "BUY", "strategy": strategy,
                        "size_usd": actual_usd, "price": price, "stop": stop,
                        "leverage": leverage, "source": "live_daemon",
                        "order_id": order_id,
                    })
                    print(colored(
                        f"  [BUY] {coinbase_sym} @ ${price:.4f} | "
                        f"RSI={sig.get('rsi', 0):.1f} | {strategy} | "
                        f"size=${actual_usd:.0f} (requested ${size_usd:.0f})",
                        "green",
                    ))
                else:
                    print(colored(
                        f"  [WARN] BUY {coinbase_sym} order {order_id} — no tokens received",
                        "yellow",
                    ))
            else:
                print(colored(f"  [FAIL] BUY {coinbase_sym}: {result}", "red"))

        elif action == "SHORT":
            side_key = "short"
            strat_key = "bb_grid" if "bb_grid" in strategy else "rsi_mr"
            pos_key = f"{coinbase_sym}:{strat_key}:{side_key}"

            if self._has_position(pos_key):
                return

            if not self.tracker.can_open():
                print(colored(f"  [SKIP] {symbol} SHORT — max positions reached", "yellow"))
                return

            leverage = sig.get("leverage", 1)
            size_usd = self._position_size(leverage)
            stop = sig["stop"]
            take_profit = sig["take_profit"]

            ok, reason = self.risk.check_entry(
                size_usd=size_usd, leverage=leverage,
                current_positions=len(self.tracker.open_positions()),
            )
            if not ok:
                print(colored(f"  [RISK] {symbol} SHORT blocked: {reason}", "yellow"))
                return

            # Leading indicator gate
            should_trade, li_reason = self._check_leading_indicators(symbol, "SHORT")
            if "CONFIRMED" in li_reason:
                print(colored(f"  [LEAD] {symbol} SHORT {li_reason}", "green"))
            elif "BLOCKED" in li_reason:
                print(colored(f"  [LEAD] {symbol} SHORT {li_reason}", "red"))
            else:
                print(colored(f"  [LEAD] {symbol} SHORT {li_reason}", "yellow"))
            time.sleep(0.2)  # rate limit: 2 extra API calls
            if not should_trade:
                return

            self._signals_today += 1

            if self.dry_run:
                print(colored(
                    f"  [DRY-RUN SHORT] {symbol} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy} | "
                    f"size=${size_usd:.0f} stop=${stop:.4f}",
                    "red",
                ))
                # Still track the position for dry-run simulation
                self.tracker.open(pos_key, "SELL", size_usd, price, stop, take_profit)
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": coinbase_sym, "side": "SHORT", "strategy": strategy,
                    "size_usd": size_usd, "price": price, "stop": stop,
                    "leverage": leverage, "source": "live_daemon_dry",
                })
                return  # skip exchange execution only

            # Execute: perp short via Coinbase
            perp_sym = coinbase_sym.replace("-USD", "-PERP-INTX")
            base_size = size_usd / price if price > 0 else 0
            try:
                result = self.executor.open_perp_short(perp_sym, base_size, leverage)
            except Exception as e:
                print(colored(f"  [FAIL] SHORT {perp_sym}: {e}", "red"))
                return
            if result.get("success") or result.get("order_id") or result.get("success_response"):
                self.tracker.open(pos_key, "SELL", size_usd, price, stop, take_profit)
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": perp_sym, "side": "SHORT", "strategy": strategy,
                    "size_usd": size_usd, "price": price, "stop": stop,
                    "leverage": leverage, "source": "live_daemon",
                })
                print(colored(
                    f"  [SHORT] {perp_sym} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy} | "
                    f"size=${size_usd:.0f}",
                    "red",
                ))
            else:
                print(colored(f"  [FAIL] SHORT {perp_sym}: {result}", "red"))

        elif action == "CLOSE_LONG":
            pos_key = sig["pos_key"]
            if not self._has_position(pos_key):
                return

            self._signals_today += 1

            if self.dry_run:
                print(colored(
                    f"  [DRY-RUN CLOSE LONG] {symbol} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy}",
                    "cyan",
                ))
                # Still close in tracker for dry-run bookkeeping
                closed = self.tracker.close(pos_key, price)
                pnl = closed.get("pnl_usd", 0)
                self._pnl_today += pnl
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": coinbase_sym, "side": "CLOSE_LONG", "strategy": strategy,
                    "size_usd": closed.get("size_usd", 0), "price": price,
                    "pnl": pnl, "source": "live_daemon_dry",
                })
                pnl_color = "green" if pnl >= 0 else "red"
                print(colored(
                    f"  [DRY-RUN P&L] {coinbase_sym} ${pnl:+.2f}",
                    pnl_color,
                ))
                return  # skip exchange execution only

            # Get position details before closing
            positions = self.tracker.open_positions()
            pos = next((p for p in positions if p["symbol"] == pos_key), None)
            if not pos:
                return

            # Query actual Coinbase balance to sell ALL tokens (not calculated from tracker)
            currency = coinbase_sym.replace("-USD", "")
            try:
                balances = self.executor.get_balances()
                base_size = balances.get(currency, 0)
            except Exception:
                base_size = pos["size_usd"] / price if price > 0 else 0
            if base_size <= 0:
                base_size = pos["size_usd"] / price if price > 0 else 0
            try:
                result = self.executor.market_sell(coinbase_sym, base_size)
            except Exception as e:
                print(colored(f"  [FAIL] CLOSE LONG {coinbase_sym}: {e}", "red"))
                return
            closed = self.tracker.close(pos_key, price)
            pnl = closed.get("pnl_usd", 0)
            self._pnl_today += pnl
            self._trades_today += 1
            self.store.append_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": coinbase_sym, "side": "CLOSE_LONG", "strategy": strategy,
                "size_usd": pos["size_usd"], "price": price,
                "pnl": pnl, "source": "live_daemon",
            })
            pnl_color = "green" if pnl >= 0 else "red"
            print(colored(
                f"  [CLOSE LONG] {coinbase_sym} @ ${price:.4f} | "
                f"P&L: ${pnl:+.2f} | {strategy}",
                pnl_color,
            ))

        elif action == "CLOSE_SHORT":
            pos_key = sig["pos_key"]
            if not self._has_position(pos_key):
                return

            self._signals_today += 1

            if self.dry_run:
                print(colored(
                    f"  [DRY-RUN COVER] {symbol} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy}",
                    "cyan",
                ))
                # Still close in tracker for dry-run bookkeeping
                closed = self.tracker.close(pos_key, price)
                pnl = closed.get("pnl_usd", 0)
                self._pnl_today += pnl
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": coinbase_sym, "side": "COVER", "strategy": strategy,
                    "size_usd": closed.get("size_usd", 0), "price": price,
                    "pnl": pnl, "source": "live_daemon_dry",
                })
                pnl_color = "green" if pnl >= 0 else "red"
                print(colored(
                    f"  [DRY-RUN P&L] {coinbase_sym} ${pnl:+.2f}",
                    pnl_color,
                ))
                return  # skip exchange execution only

            positions = self.tracker.open_positions()
            pos = next((p for p in positions if p["symbol"] == pos_key), None)
            if not pos:
                return

            perp_sym = coinbase_sym.replace("-USD", "-PERP-INTX")
            result = self.executor.close_perp(perp_sym)
            closed = self.tracker.close(pos_key, price)
            pnl = closed.get("pnl_usd", 0)
            self._pnl_today += pnl
            self._trades_today += 1
            self.store.append_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": perp_sym, "side": "COVER", "strategy": strategy,
                "size_usd": pos["size_usd"], "price": price,
                "pnl": pnl, "source": "live_daemon",
            })
            pnl_color = "green" if pnl >= 0 else "red"
            print(colored(
                f"  [COVER] {perp_sym} @ ${price:.4f} | "
                f"P&L: ${pnl:+.2f} | {strategy}",
                pnl_color,
            ))

    # ------------------------------------------------------------------
    # Price updates & stop enforcement
    # ------------------------------------------------------------------

    def _update_prices(self):
        """Update current prices for all open positions."""
        for pos in self.tracker.open_positions():
            pos_key = pos["symbol"]
            # Extract coinbase symbol from pos_key (e.g. "ATOM-USD:bb_grid:long")
            coinbase_sym = pos_key.split(":")[0]
            binance_sym = coinbase_sym.replace("-USD", "/USDT")
            try:
                ticker = self.fetcher.ticker(binance_sym)
                self.tracker.update_price(pos_key, ticker["last"])
            except Exception as e:
                print(colored(f"  [WARN] Price update failed for {pos_key}: {e}", "yellow"))

    def _check_profit_taking(self):
        """Check profit-taking levels on all open positions."""
        for pos_key, pos in list(self.tracker._positions.items()):
            actions = pos.check_profit_taking()
            for action in actions:
                if action["action"] == "partial_sell":
                    pct = action["pct"]
                    reason = action["reason"]
                    usd_to_sell = pos.reduce_size(pct)

                    # In live mode, execute partial sell — exchange is source of truth
                    coinbase_sym = pos_key.split(":")[0]
                    if not self.dry_run and self.executor:
                        currency = coinbase_sym.replace("-USD", "")
                        if pos.side == "BUY":
                            try:
                                actual_balance = self.executor.get_token_balance(currency)
                                base_to_sell = actual_balance * (pct / 100)
                                if base_to_sell > 0:
                                    self.executor.market_sell(coinbase_sym, base_to_sell)
                            except Exception as e:
                                print(colored(f"  [FAIL] TP sell {coinbase_sym}: {e}", "red"))

                    self._pnl_today += (pos.current_price - pos.entry_price) / pos.entry_price * usd_to_sell
                    print(colored(f"  [TP] {pos_key} {reason} — sold {pct}% (${usd_to_sell:.2f})", "green"))
                    self.store.append_trade({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": coinbase_sym, "side": "PARTIAL_SELL",
                        "size_usd": usd_to_sell, "price": pos.current_price,
                        "pnl": (pos.current_price - pos.entry_price) / pos.entry_price * usd_to_sell,
                        "source": "profit_taking",
                    })
                    self.tracker.save_state()  # persist after reduce_size

                elif action["action"] == "trailing_stop":
                    reason = action["reason"]
                    print(colored(f"  [TRAIL] {pos_key} {reason}", "yellow"))
                    # Close remaining position
                    closed = self.tracker.close(pos_key, pos.current_price)
                    # Execute sell — exchange is source of truth for actual balance
                    coinbase_sym = pos_key.split(":")[0]
                    if not self.dry_run and self.executor:
                        currency = coinbase_sym.replace("-USD", "")
                        if pos.side == "BUY":
                            try:
                                base_size = self.executor.get_token_balance(currency)
                                if base_size > 0:
                                    self.executor.market_sell(coinbase_sym, base_size)
                            except Exception as e:
                                print(colored(f"  [FAIL] TRAIL sell {coinbase_sym}: {e}", "red"))
                    self._pnl_today += closed.get("pnl_usd", 0)
                    self.store.append_trade({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": coinbase_sym, "side": "TRAILING_STOP",
                        "size_usd": pos.size_usd, "price": pos.current_price,
                        "pnl": closed.get("pnl_usd", 0),
                        "source": "trailing_stop",
                    })
                    break  # position is closed, stop iterating actions

    def _enforce_stops(self):
        """Check and execute stop-losses."""
        stopped = self.tracker.check_stops()
        for pos in stopped:
            pos_key = pos["symbol"]
            coinbase_sym = pos_key.split(":")[0]
            price = pos["current_price"]

            if self.dry_run:
                print(colored(
                    f"  [DRY-RUN STOP] {pos_key} hit stop at ${pos['stop_price']:.4f} "
                    f"(current: ${price:.4f})",
                    "yellow",
                ))
                # Still close in tracker for dry-run bookkeeping
                closed = self.tracker.close(pos_key, price)
                self._pnl_today += closed.get("pnl_usd", 0)
                continue

            is_short = pos["side"] == "SELL"
            if is_short:
                perp_sym = coinbase_sym.replace("-USD", "-PERP-INTX")
                try:
                    self.executor.close_perp(perp_sym)
                except Exception as e:
                    print(colored(f"  [FAIL] STOP close perp {perp_sym}: {e}", "red"))
                    continue
            else:
                # Exchange is source of truth — query actual balance
                currency = coinbase_sym.replace("-USD", "")
                try:
                    base_size = self.executor.get_token_balance(currency)
                except Exception:
                    base_size = pos["size_usd"] / price if price > 0 else 0
                if base_size <= 0:
                    continue
                try:
                    self.executor.market_sell(coinbase_sym, base_size)
                except Exception as e:
                    print(colored(f"  [FAIL] STOP sell {coinbase_sym}: {e}", "red"))
                    continue

            closed = self.tracker.close(pos_key, price)
            pnl = closed.get("pnl_usd", 0)
            self._pnl_today += pnl
            self._trades_today += 1

            self.store.append_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": pos_key, "side": "STOP_CLOSE",
                "size_usd": pos["size_usd"], "price": price,
                "stop": pos["stop_price"], "pnl": pnl,
                "source": "live_daemon",
            })
            print(colored(
                f"  [STOP] {pos_key} @ ${price:.4f} | "
                f"stop=${pos['stop_price']:.4f} | P&L: ${pnl:+.2f}",
                "yellow",
            ))

    # ------------------------------------------------------------------
    # Equity update
    # ------------------------------------------------------------------

    def _update_equity(self):
        """Refresh equity from balances + unrealized P&L."""
        if self.dry_run:
            # In dry-run, equity = starting + realized P&L + unrealized
            exposure = self.tracker.total_exposure()
            self._equity = 10_000.0 + self._pnl_today + exposure
        else:
            balances = self.executor.get_balances()
            self._equity = sum(balances.values()) + self.tracker.total_exposure()
        self.risk.update_portfolio_value(self._equity)

    def _reconcile_positions(self):
        """Compare tracker vs exchange. Log mismatches. Run every signal cycle."""
        if self.dry_run or not self.executor:
            return
        try:
            balances = self.executor.get_balances()
            tracked_currencies = set()
            for pos in self.tracker.open_positions():
                currency = pos["symbol"].split("-")[0]
                tracked_currencies.add(currency)

            # Check for tokens on Coinbase not in tracker
            for currency, amount in balances.items():
                if currency in ("USD", "USDC", "USDT"):
                    continue
                if amount > 0.001 and currency not in tracked_currencies:
                    print(colored(
                        f"  [RECONCILE] Untracked {currency}: {amount:.6f} on Coinbase — not in tracker",
                        "yellow"))

            # Check for tracked positions with no tokens on Coinbase
            # Coinbase is source of truth — auto-close phantom positions
            for currency in tracked_currencies:
                if currency not in balances or balances.get(currency, 0) < 0.001:
                    print(colored(
                        f"  [RECONCILE] Tracked {currency} has no tokens on Coinbase — removing phantom position",
                        "yellow"))
                    # Find and close the phantom position in tracker
                    # tracker.close() expects the position symbol key (e.g. "SOL-USD:synced:long")
                    for pos in self.tracker.open_positions():
                        pos_currency = pos["symbol"].split("-")[0]
                        if pos_currency == currency:
                            try:
                                self.tracker.close(pos["symbol"], float(pos.get("entry_price", 0)))
                                print(colored(f"  [RECONCILE] Closed phantom: {pos['symbol']}", "yellow"))
                            except Exception as e:
                                print(colored(f"  [RECONCILE] Failed to close {pos['symbol']}: {e}", "yellow"))
        except Exception as e:
            print(colored(f"  [RECONCILE] Failed: {e}", "yellow"))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _save_snapshot(self, signals: list[dict]):
        """Save a snapshot for Claude review."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "dry_run" if self.dry_run else "live",
            "equity": self._equity,
            "positions": self.tracker.open_positions(),
            "exposure": self.tracker.total_exposure(),
            "signals": signals,
            "trades_today": self._trades_today,
            "signals_today": self._signals_today,
            "pnl_today": self._pnl_today,
            "halted": self.risk.is_halted(self._equity),
        }
        self.store.save_snapshot(snapshot)

    # ------------------------------------------------------------------
    # Status line
    # ------------------------------------------------------------------

    def _print_status(self, signals: list[dict]):
        """Print a clear status line showing equity, positions, signals, P&L."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")
        n_pos = len(self.tracker.open_positions())
        exposure = self.tracker.total_exposure()
        halted = self.risk.is_halted(self._equity)

        mode_tag = colored("[DRY-RUN]", "magenta") if self.dry_run else colored("[LIVE]", "green")
        halt_tag = colored(" HALTED", "red") if halted else ""

        pnl_str = f"${self._pnl_today:+.2f}"
        pnl_colored = colored(pnl_str, "green" if self._pnl_today >= 0 else "red")

        print(
            f"\n{mode_tag} {now} UTC | "
            f"Equity: ${self._equity:,.2f} | "
            f"Positions: {n_pos}/{MAX_CONCURRENT_POSITIONS} | "
            f"Exposure: ${exposure:,.2f} | "
            f"Signals: {len(signals)} | "
            f"Trades: {self._trades_today} | "
            f"P&L: {pnl_colored}{halt_tag}"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def startup(self):
        """Fetch initial data on startup."""
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        kalshi_tag = " (KALSHI-ONLY)" if self.kalshi_only else ""
        print(colored(f"{'='*70}", "cyan"))
        print(colored(f"  Production Trading Daemon — {mode} MODE{kalshi_tag}", "cyan"))
        print(colored(f"  Strategies: {'Kalshi 15m Predictions only' if self.kalshi_only else 'BB Grid + RSI MR (per-pair config)'}", "cyan"))
        print(colored(f"  Pairs: {', '.join(ALL_PAIRS)}", "cyan"))
        print(colored(f"  Timeframe: 15m | Equity: ${self._equity:,.2f}", "cyan"))
        print(colored(f"  Stop-loss: {STOP_LOSS_PCT:.0%} | Position size: {POSITION_SIZE_PCT:.0%}", "cyan"))
        if self.kalshi_predictor_version == 'v3':
            label = 'V3 Strike-Relative'
        elif self.kalshi_predictor_version == 'v2':
            label = 'V2 Continuation'
        else:
            label = 'V1 Mean-Reversion'
        print(colored(f"  Predictor: {label}", "cyan"))
        print(colored(f"{'='*70}", "cyan"))

        print("\n[STARTUP] Fetching initial 15m data for all pairs...")
        self._fetch_all()
        print(f"[STARTUP] Loaded data for {len(self._dataframes)}/{len(ALL_PAIRS)} pairs")
        for sym, df in self._dataframes.items():
            last = df.iloc[-1]
            rsi = last.get("rsi", 0)
            close = float(last["close"])
            print(f"  {sym}: ${close:.4f} | RSI={rsi:.1f} | {len(df)} candles")

        # Sync positions from exchanges (belt + suspenders)
        self._sync_positions_from_exchange()

    def signal_cycle(self):
        """Full signal cycle: fetch data, generate signals, execute."""
        print(f"\n[CYCLE] {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC — fetching 15m candles...")
        self._fetch_all()

        # Check if trading is halted (use full equity, not just exposure)
        if self.risk.is_halted(self._equity):
            print(colored("[HALT] Daily drawdown limit reached — skipping signal generation", "red"))
            return []

        signals = []

        # Spot trading (Coinbase) — skip when kalshi_only mode
        if not self.kalshi_only:
            signals = self._collect_all_signals()

            if signals:
                print(f"[CYCLE] {len(signals)} signals detected (sorted by confidence):")
                for sig in signals:
                    conf = sig.get("_confidence", 0)
                    action = sig.get("action", "?")
                    sym = sig.get("symbol", "?")
                    # Skip new entries if at max positions (closes always execute)
                    if action in ("BUY", "SHORT") and not self.tracker.can_open():
                        print(colored(f"  [SKIP] {action} {sym} (conf={conf:.0f}) — max {MAX_CONCURRENT_POSITIONS} positions reached", "yellow"))
                        continue
                    self._execute_signal(sig)
            else:
                print("[CYCLE] No signals")
        else:
            print("[CYCLE] Kalshi-only mode — skipping spot signals")

        self._update_equity()
        self._reconcile_positions()
        self._save_snapshot(signals)
        self._print_status(signals)
        return signals

    def tick(self):
        """Minute tick: update prices, enforce stops, check profit taking."""
        self._update_prices()
        if not self.kalshi_only:
            self._enforce_stops()
            self._check_profit_taking()
        self._update_equity()

    def run(self):
        """Main daemon loop."""
        self._running = True

        def _handle_shutdown(*_):
            print(colored("\n[SHUTDOWN] Graceful shutdown requested...", "yellow"))
            self._running = False

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        self.startup()

        # Run initial signal cycle immediately
        self.signal_cycle()

        # Run initial Kalshi eval immediately
        try:
            self._kalshi_eval()
            self._last_kalshi_eval = time.time()
        except Exception as e:
            print(colored(f"  [KALSHI EVAL ERR] startup eval: {e}", "red"))

        last_signal_time = time.time()

        while self._running:
            try:
                time.sleep(TICK_INTERVAL)
                if not self._running:
                    break

                self.tick()

                now = time.time()
                if now - last_signal_time >= SIGNAL_INTERVAL:
                    self.signal_cycle()
                    last_signal_time = now

                # Wall-clock aligned Kalshi eval at :01, :06, :11 + LAST_LOOK at :12/:27/:42/:57
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

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(colored(f"[ERROR] {e}", "red"))
                time.sleep(TICK_INTERVAL)

        # Shutdown summary
        print(colored(f"\n{'='*70}", "cyan"))
        print(colored("  DAEMON STOPPED", "cyan"))
        print(f"  Trades: {self._trades_today} | Signals: {self._signals_today} | P&L: ${self._pnl_today:+.2f}")
        print(f"  Open positions: {len(self.tracker.open_positions())}")
        print(f"  Final equity: ${self._equity:,.2f}")
        print(colored(f"{'='*70}", "cyan"))


def main():
    parser = argparse.ArgumentParser(description="Production trading daemon")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print signals without executing trades",
    )
    parser.add_argument(
        "--kalshi-only", action="store_true",
        help="Run only Kalshi 15m predictions, skip Coinbase spot trading",
    )
    parser.add_argument("--predictor", choices=["v1", "v2", "v3"], default="v1",
                        help="Kalshi predictor: v1 (mean-reversion), v2 (continuation), or v3 (strike-relative)")
    args = parser.parse_args()

    daemon = LiveDaemon(dry_run=args.dry_run, kalshi_only=args.kalshi_only, predictor_version=args.predictor)
    daemon.run()


if __name__ == "__main__":
    main()
