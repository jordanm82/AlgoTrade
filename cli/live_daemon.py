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


class LiveDaemon:
    """Production daemon that runs BB Grid + RSI MR on validated pairs."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
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
        from strategy.strategies.kalshi_predictor import KalshiPredictor
        self.kalshi_predictor = KalshiPredictor()
        self.kalshi_client = None  # lazy init
        self.kalshi_threshold = 30  # minimum confidence to bet (lowered from 40 — conf 25-35 with flow confirmation is the sweet spot)
        self.kalshi_predictions: list[dict] = []  # latest predictions for dashboard

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
    }

    def _init_kalshi_client(self):
        """Lazy-initialize the Kalshi client."""
        if self.kalshi_client is not None:
            return
        try:
            from exchange.kalshi import KalshiClient
            from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
            demo = self.dry_run  # demo=True for dry-run, demo=False for live
            self.kalshi_client = KalshiClient(
                api_key_id=KALSHI_API_KEY_ID,
                private_key_path=str(KALSHI_KEY_FILE),
                demo=demo,
            )
        except Exception as e:
            print(colored(f"  [WARN] Kalshi client init failed: {e}", "yellow"))

    def _kalshi_cycle(self):
        """Run Kalshi predictions for BTC/ETH/SOL/XRP and optionally place bets."""
        predictions = []
        for symbol, series_ticker in self.KALSHI_PAIRS.items():
            # Kalshi pairs may not be in our Coinbase trading set — fetch independently
            df = self._dataframes.get(symbol)
            if df is None or len(df) < 20:
                try:
                    df = self._fetch_pair(symbol)
                    if df is not None:
                        self._dataframes[symbol] = df
                except Exception:
                    pass
            if df is None or len(df) < 20:
                predictions.append({
                    "symbol": symbol, "asset": symbol.split("/")[0],
                    "direction": "--", "confidence": 0,
                    "reason": "no data", "ob": 0, "flow": 0,
                })
                continue

            # Get leading indicator data for enhanced prediction
            market_data = None
            try:
                from data.market_data import get_order_book_imbalance, get_trade_flow
                ob = get_order_book_imbalance(symbol)
                tf = get_trade_flow(symbol, limit=100)
                market_data = {"order_book": ob, "trade_flow": tf}
            except Exception:
                pass

            signal = self.kalshi_predictor.score(df, market_data=market_data)
            if signal is None:
                predictions.append({
                    "symbol": symbol, "asset": symbol.split("/")[0],
                    "direction": "--", "confidence": 0,
                    "reason": "neutral", "ob": 0, "flow": 0,
                })
                continue

            ob_imb = (market_data or {}).get("order_book", {}).get("imbalance", 0)
            net_flow = (market_data or {}).get("trade_flow", {}).get("net_flow", 0)

            pred = {
                "symbol": symbol,
                "asset": symbol.split("/")[0],
                "direction": signal.direction,
                "confidence": signal.confidence,
                "ob": ob_imb,
                "flow": net_flow,
                "reason": "",
            }

            if signal.confidence < self.kalshi_threshold:
                pred["reason"] = f"below threshold ({self.kalshi_threshold})"
                predictions.append(pred)
                continue

            # Determine bet side and approximate price
            side = "yes" if signal.direction == "UP" else "no"
            approx_cents = max(5, min(95, int(50 + (signal.confidence - 50) * 0.5)))
            pred["reason"] = f"would bet {side.upper()} @ ~{approx_cents}c"

            if self.dry_run:
                print(colored(
                    f"  [KALSHI DRY] {pred['asset']} {signal.direction} "
                    f"conf={signal.confidence} | "
                    f"OB={ob_imb:+.2f} flow={net_flow:+.2f} | "
                    f"bet {side.upper()} @ ~{approx_cents}c",
                    "magenta",
                ))
            else:
                # Live: place the bet
                self._init_kalshi_client()
                if self.kalshi_client is None:
                    pred["reason"] = "client init failed"
                    predictions.append(pred)
                    continue
                try:
                    # Get balance and compute bet size (5% of balance, min $5)
                    balance_resp = self.kalshi_client.get_balance()
                    balance_cents = balance_resp.get("balance", 0)
                    bet_cents = max(500, int(balance_cents * 0.05))
                    count = max(1, bet_cents // approx_cents)

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
                        predictions.append(pred)
                        continue

                    # Pick the soonest-expiring open market
                    ticker = all_markets[0].get("ticker", "")
                    # Kalshi requires a price — use aggressive limit (99c YES / 99c NO) for instant fill
                    fill_price = 99 if side == "yes" else 99
                    result = self.kalshi_client.place_order(
                        ticker=ticker,
                        side=side,
                        count=count,
                        price_cents=fill_price,
                        order_type="limit",
                    )
                    order_id = result.get("order", {}).get("order_id", "?")
                    fill_count = result.get("order", {}).get("fill_count_fp", "0")
                    pred["reason"] = f"placed {side.upper()} x{count} filled={fill_count} (#{order_id})"
                    print(colored(
                        f"  [KALSHI BET] {pred['asset']} {signal.direction} "
                        f"conf={signal.confidence} | {side.upper()} x{count} @ {approx_cents}c",
                        "magenta",
                    ))
                except Exception as e:
                    pred["reason"] = f"order failed: {e}"
                    print(colored(f"  [KALSHI ERR] {pred['asset']}: {e}", "red"))

            predictions.append(pred)

        self.kalshi_predictions = predictions
        return predictions

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
                self.tracker.open(pos_key, "BUY", size_usd, price, stop, take_profit)
                self._trades_today += 1
                self.store.append_trade({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": coinbase_sym, "side": "BUY", "strategy": strategy,
                    "size_usd": size_usd, "price": price, "stop": stop,
                    "leverage": leverage, "source": "live_daemon",
                })
                print(colored(
                    f"  [BUY] {coinbase_sym} @ ${price:.4f} | "
                    f"RSI={sig.get('rsi', 0):.1f} | {strategy} | "
                    f"size=${size_usd:.0f}",
                    "green",
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

            base_size = pos["size_usd"] / price if price > 0 else 0
            result = self.executor.market_sell(coinbase_sym, base_size)
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

                    # In live mode, execute partial sell
                    coinbase_sym = pos_key.split(":")[0]
                    if not self.dry_run and self.executor:
                        base_to_sell = usd_to_sell / pos.current_price if pos.current_price > 0 else 0
                        if pos.side == "BUY":
                            self.executor.market_sell(coinbase_sym, base_to_sell)
                        # For shorts, partial close is more complex -- skip for now

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
                    # Execute sell
                    coinbase_sym = pos_key.split(":")[0]
                    if not self.dry_run and self.executor:
                        base_size = pos.size_usd / pos.current_price if pos.current_price > 0 else 0
                        if pos.side == "BUY":
                            self.executor.market_sell(coinbase_sym, base_size)
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
                self.executor.close_perp(perp_sym)
            else:
                base_size = pos["size_usd"] / price if price > 0 else 0
                self.executor.market_sell(coinbase_sym, base_size)

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
        print(colored(f"{'='*70}", "cyan"))
        print(colored(f"  Production Trading Daemon — {mode} MODE", "cyan"))
        print(colored(f"  Strategies: BB Grid + RSI MR (per-pair config)", "cyan"))
        print(colored(f"  Pairs: {', '.join(ALL_PAIRS)}", "cyan"))
        print(colored(f"  Timeframe: 15m | Equity: ${self._equity:,.2f}", "cyan"))
        print(colored(f"  Stop-loss: {STOP_LOSS_PCT:.0%} | Position size: {POSITION_SIZE_PCT:.0%}", "cyan"))
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

        # Kalshi prediction cycle
        try:
            self._kalshi_cycle()
        except Exception as e:
            print(colored(f"  [KALSHI ERR] cycle failed: {e}", "yellow"))

        self._update_equity()
        self._save_snapshot(signals)
        self._print_status(signals)
        return signals

    def tick(self):
        """Minute tick: update prices, enforce stops, check profit taking."""
        self._update_prices()
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
    args = parser.parse_args()

    daemon = LiveDaemon(dry_run=args.dry_run)
    daemon.run()


if __name__ == "__main__":
    main()
