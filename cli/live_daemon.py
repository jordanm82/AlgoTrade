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

from config.production import (
    BB_GRID_CONFIG,
    LEVERAGE_PAIRS,
    MAX_CONCURRENT_POSITIONS,
    MONITORED_PAIRS_15M,
    PAIR_TO_COINBASE,
    POSITION_SIZE_PCT,
    RSI_MR_CONFIG,
    STOP_LOSS_PCT,
)
from config.settings import CDP_KEY_FILE, DATA_DIR
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from data.store import DataStore
from exchange.coinbase import CoinbaseExecutor
from exchange.positions import PositionTracker
from risk.manager import RiskManager

# Pairs eligible for RSI Mean Reversion strategy
RSI_MR_PAIRS = ["ATOM/USDT", "FIL/USDT"]

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
        for symbol in MONITORED_PAIRS_15M:
            df = self._fetch_pair(symbol)
            if df is not None:
                self._dataframes[symbol] = df

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _bb_grid_signals(self, symbol: str, df: pd.DataFrame) -> list[dict]:
        """Check BB Grid Long+Short signals on a single pair."""
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
        buy_thresh = BB_GRID_CONFIG["rsi_buy_threshold"]
        short_thresh = BB_GRID_CONFIG["rsi_short_threshold"]
        leverage = 2 if symbol in LEVERAGE_PAIRS else 1

        coinbase_sym = PAIR_TO_COINBASE[symbol]

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
        """Check RSI Mean Reversion signals for ATOM and FIL."""
        if symbol not in RSI_MR_PAIRS:
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
        coinbase_sym = PAIR_TO_COINBASE[symbol]

        # BUY: RSI < oversold (30)
        if rsi < RSI_MR_CONFIG["oversold"]:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "BUY",
                "strategy": "rsi_mr_1x",
                "leverage": 1,
                "price": close,
                "rsi": rsi,
                "stop": close * (1 - STOP_LOSS_PCT),
                "take_profit": close + atr * 3,
                "atr": atr,
            })

        # SELL (exit long): RSI > exit_long (65)
        if rsi > RSI_MR_CONFIG["exit_long"]:
            pos_key = f"{coinbase_sym}:rsi_mr:long"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_LONG",
                    "strategy": "rsi_mr_1x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                })

        # SHORT: RSI > overbought (70)
        if rsi > RSI_MR_CONFIG["overbought"]:
            signals.append({
                "symbol": symbol,
                "coinbase_symbol": coinbase_sym,
                "action": "SHORT",
                "strategy": "rsi_mr_1x",
                "leverage": 1,
                "price": close,
                "rsi": rsi,
                "stop": close * (1 + STOP_LOSS_PCT),
                "take_profit": close - atr * 3,
                "atr": atr,
            })

        # COVER (exit short): RSI < exit_short (35)
        if rsi < RSI_MR_CONFIG["exit_short"]:
            pos_key = f"{coinbase_sym}:rsi_mr:short"
            if pos_key in [p["symbol"] for p in self.tracker.open_positions()]:
                signals.append({
                    "symbol": symbol,
                    "coinbase_symbol": coinbase_sym,
                    "action": "CLOSE_SHORT",
                    "strategy": "rsi_mr_1x",
                    "pos_key": pos_key,
                    "price": close,
                    "rsi": rsi,
                })

        return signals

    def _collect_all_signals(self) -> list[dict]:
        """Collect all signals across all pairs and strategies."""
        all_signals = []
        for symbol, df in self._dataframes.items():
            all_signals.extend(self._bb_grid_signals(symbol, df))
            all_signals.extend(self._rsi_mr_signals(symbol, df))
        return all_signals

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _position_size(self, leverage: int = 1) -> float:
        """Calculate position size: 10% of current equity * leverage."""
        return self._equity * POSITION_SIZE_PCT * leverage

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
            result = self.executor.open_perp_short(perp_sym, base_size, leverage)
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
        print(colored(f"  Strategies: BB Grid (2x on ATOM/FIL/DOT, 1x others) + RSI MR", "cyan"))
        print(colored(f"  Pairs: {', '.join(MONITORED_PAIRS_15M)}", "cyan"))
        print(colored(f"  Timeframe: 15m | Equity: ${self._equity:,.2f}", "cyan"))
        print(colored(f"  Stop-loss: {STOP_LOSS_PCT:.0%} | Position size: {POSITION_SIZE_PCT:.0%}", "cyan"))
        print(colored(f"{'='*70}", "cyan"))

        print("\n[STARTUP] Fetching initial 15m data for all pairs...")
        self._fetch_all()
        print(f"[STARTUP] Loaded data for {len(self._dataframes)}/{len(MONITORED_PAIRS_15M)} pairs")
        for sym, df in self._dataframes.items():
            last = df.iloc[-1]
            rsi = last.get("rsi", 0)
            close = float(last["close"])
            print(f"  {sym}: ${close:.4f} | RSI={rsi:.1f} | {len(df)} candles")

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
            print(f"[CYCLE] {len(signals)} signals detected:")
            for sig in signals:
                self._execute_signal(sig)
        else:
            print("[CYCLE] No signals")

        self._update_equity()
        self._save_snapshot(signals)
        self._print_status(signals)
        return signals

    def tick(self):
        """Minute tick: update prices, enforce stops."""
        self._update_prices()
        self._enforce_stops()
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
