#!/usr/bin/env python3
"""Live ASCII trading dashboard — runs signal cycles with 1-minute full redraws.

Writes to both stdout AND data/store/dashboard.log so Claude can read asynchronously.
Fetches fresh ticker prices every 1-minute tick so prices update live.

Usage:
    python dashboard.py --dry-run     # paper trading (default)
    python dashboard.py --live        # live trading
    python dashboard.py --cycles 15   # number of 15-min signal cycles
"""
import sys
sys.path.insert(0, '.')
import argparse
import io
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from termcolor import colored

from config.production import (
    BB_GRID_CONFIG, LEVERAGE_PAIRS, MAX_CONCURRENT_POSITIONS,
    MONITORED_PAIRS_15M, PAIR_TO_COINBASE, POSITION_SIZE_PCT,
    RSI_MR_CONFIG, STOP_LOSS_PCT, MAX_LEVERAGE,
)
from cli.live_daemon import LiveDaemon

LOG_FILE = Path("data/store/dashboard.log")


class Dashboard:
    def __init__(self, dry_run: bool = True, max_cycles: int = 15):
        self.daemon = LiveDaemon(dry_run=dry_run)
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self._cycle_count = 0
        self._tick_count = 0
        self._start_time = datetime.now(timezone.utc)
        self._start_equity = self.daemon._equity
        self._last_signals: list[dict] = []
        # Live ticker prices updated every minute
        self._live_prices: dict[str, float] = {}

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _out(self, text: str):
        """Write to both stdout and log file."""
        print(text)
        with open(LOG_FILE, "a") as f:
            f.write(text + "\n")

    def _write_log(self, lines: list[str]):
        """Write full dashboard frame to stdout + overwrite latest snapshot in log."""
        output = "\n".join(lines)
        print(output)
        sys.stdout.flush()
        # Overwrite log with latest frame (not append — keeps it readable)
        with open(LOG_FILE, "w") as f:
            f.write(output + "\n")

    def _fetch_live_prices(self):
        """Fetch fresh ticker prices for all pairs (fast, ~1 API call each)."""
        for sym in MONITORED_PAIRS_15M:
            try:
                ticker = self.daemon.fetcher.ticker(sym)
                self._live_prices[sym] = float(ticker.get("last", 0))
            except Exception:
                pass  # keep last known price

    def _get_price(self, sym: str) -> float:
        """Get best available price: live ticker > cached dataframe."""
        if sym in self._live_prices and self._live_prices[sym] > 0:
            return self._live_prices[sym]
        df = self.daemon._dataframes.get(sym)
        if df is not None and len(df) > 0:
            return float(df.iloc[-1]["close"])
        return 0.0

    def _draw_opening(self):
        """Opening status screen."""
        now = datetime.now(timezone.utc)
        mode = "DRY-RUN" if self.dry_run else "LIVE"
        lines = [
            "",
            "=" * 78,
            "",
            "     ___    __    _____ ____  ______ ____  ___    ____  ______",
            "    /   |  / /   / ___// __ \\/_  __// __ \\/   |  / __ \\/ ____/",
            "   / /| | / /   / __ \\/ / / / / /  / /_/ / /| | / / / / __/",
            "  / ___ |/ /___/ /_/ / /_/ / / /  / _, _/ ___ |/ /_/ / /___",
            " /_/  |_/_____/\\____/\\____/ /_/  /_/ |_/_/  |_/_____/_____/",
            "",
            "=" * 78,
            f"  MODE: {mode}          {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"  EQUITY: ${self._start_equity:,.2f}",
            f"  MAX CYCLES: {self.max_cycles}  (~{self.max_cycles * 15} minutes)",
            "=" * 78,
            "",
            "  STRATEGIES:",
            "    1. BB Grid Long+Short (buy<BB_lower+RSI<35, sell>BB_mid)",
            "       - 2x leverage: ATOM, FIL, DOT",
            "       - 1x leverage: UNI, LTC, SHIB",
            "",
            "    2. RSI Mean Reversion Long+Short (buy RSI<30, sell RSI>65)",
            "       - 1x leverage: ATOM, FIL",
            "",
            "  RISK CONTROLS:",
            f"    - Position size: {POSITION_SIZE_PCT:.0%} of equity (compounding)",
            f"    - Max leverage: {MAX_LEVERAGE}x",
            f"    - Stop-loss: {STOP_LOSS_PCT:.0%} hard stop",
            f"    - Max positions: {MAX_CONCURRENT_POSITIONS}",
            "    - Daily drawdown halt: 5%",
            "",
            "  MONITORED PAIRS:",
            f"    {'  '.join(MONITORED_PAIRS_15M)}",
            "",
            "  BACKTEST PERFORMANCE (6 months, Sep 2025 - Mar 2026):",
            "    ATOM BB Grid 2x:  87.7% WR | +437% return | 487 trades",
            "    FIL  BB Grid 2x:  71.2% WR | +309% return | 472 trades",
            "    ATOM RSI MR:      84.5% WR |  +70% return | 290 trades",
            "",
            "=" * 78,
        ]
        self._write_log(lines)
        time.sleep(2)

    def _draw_dashboard(self, is_signal_cycle: bool, signals: list[dict] | None):
        """Draw full dashboard — used for both signal cycles and minute ticks."""
        now = datetime.now(timezone.utc)
        uptime = now - self._start_time
        h = int(uptime.total_seconds() // 3600)
        m = int((uptime.total_seconds() % 3600) // 60)

        mode = "DRY" if self.dry_run else "LIVE"
        equity = self.daemon._equity
        pnl = self.daemon._pnl_today
        pnl_pct = (equity - self._start_equity) / self._start_equity * 100 if self._start_equity > 0 else 0
        positions = self.daemon.tracker.open_positions()
        exposure = self.daemon.tracker.total_exposure()
        closed = self.daemon.tracker.closed_trades()
        wins = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
        losses = sum(1 for t in closed if t.get("pnl_usd", 0) <= 0)
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0

        label = f"Cycle {self._cycle_count}/{self.max_cycles}" if is_signal_cycle else f"Tick {self._tick_count}"

        lines = [
            "",
            "=" * 78,
            f"  ALGOTRADE [{mode}]  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {label}  |  Up {h}h{m}m",
            "=" * 78,
            "",
            f"  EQUITY: ${equity:,.2f}  |  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            f"  POSITIONS: {len(positions)}/{MAX_CONCURRENT_POSITIONS}  |  EXPOSURE: ${exposure:,.2f}",
            f"  TRADES: {total}  |  W:{wins} L:{losses}  |  WIN RATE: {wr:.1f}%",
            "",
        ]

        # Pairs table with LIVE prices
        lines.append(f"  {'PAIR':<12} {'PRICE':>10} {'RSI':>6} {'BB_LOW':>10} {'BB_MID':>10} {'BB_UP':>10} {'SIGNAL':<16}")
        lines.append(f"  {'-'*76}")
        for sym in MONITORED_PAIRS_15M:
            df = self.daemon._dataframes.get(sym)
            if df is None or len(df) == 0:
                lines.append(f"  {sym:<12} {'--':>10}")
                continue

            last = df.iloc[-1]
            # Use live ticker price if available, fall back to candle close
            price = self._live_prices.get(sym, float(last["close"]))
            rsi = float(last.get("rsi", 0)) if pd.notna(last.get("rsi")) else 0
            bb_l = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
            bb_m = float(last.get("bb_middle", 0)) if pd.notna(last.get("bb_middle")) else 0
            bb_u = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0

            sig_str = ""
            if price < bb_l and rsi < 35:
                sig_str = "** BB+RSI BUY **"
            elif price > bb_u and rsi > 65:
                sig_str = "** BB+RSI SHORT **"
            elif rsi < 30:
                sig_str = "! RSI OVERSOLD"
            elif rsi > 70:
                sig_str = "! RSI OVERBOUGHT"

            lev_tag = " [2x]" if sym in LEVERAGE_PAIRS else ""
            lines.append(f"  {sym:<12} ${price:>9.4f} {rsi:5.1f} ${bb_l:>9.4f} ${bb_m:>9.4f} ${bb_u:>9.4f} {sig_str}{lev_tag}")

        # Open positions with live P&L
        if positions:
            lines.append("")
            lines.append("  OPEN POSITIONS:")
            lines.append(f"  {'KEY':<32} {'SIDE':<5} {'ENTRY':>10} {'NOW':>10} {'P&L':>10} {'STOP':>10}")
            lines.append(f"  {'-'*80}")
            for p in positions:
                key = p["symbol"][:32]
                side = p["side"]
                upnl = p["unrealized_pnl"]
                pnl_sign = "+" if upnl >= 0 else ""
                lines.append(
                    f"  {key:<32} {side:<5} "
                    f"${p['entry_price']:>9.4f} ${p['current_price']:>9.4f} "
                    f"${pnl_sign}{upnl:.2f}{'':>4} ${p['stop_price']:>9.4f}"
                )

        # Signals
        sigs = signals if is_signal_cycle else self._last_signals
        if sigs:
            lines.append("")
            lines.append(f"  SIGNALS ({len(sigs)}):")
            for s in sigs[-8:]:
                action = s.get("action", "?")
                sym = s.get("symbol", "?")
                price = s.get("price", 0)
                rsi_v = s.get("rsi", 0)
                strat = s.get("strategy", "?")
                lines.append(f"    >> {action} {sym} @ ${price:.4f} RSI={rsi_v:.1f} [{strat}]")
        else:
            lines.append("")
            lines.append("  No signals this cycle")

        # Closed trades
        if closed:
            lines.append("")
            lines.append(f"  CLOSED TRADES (last 5):")
            for t in closed[-5:]:
                sym = t.get("symbol", "?")[:25]
                side = t.get("side", "?")
                pnl_t = t.get("pnl_usd", 0)
                sign = "+" if pnl_t >= 0 else ""
                lines.append(f"    {sym:<25} {side:<6} ${sign}{pnl_t:.2f}")

        # Footer
        remaining = self.max_cycles - self._cycle_count
        if is_signal_cycle:
            lines.append(f"\n  Next signal cycle in ~15 min | {remaining} cycles remaining")
        else:
            mins_to_cycle = 15 - (self._tick_count % 15)
            lines.append(f"\n  Next signal cycle in ~{mins_to_cycle} min | {remaining} cycles remaining")
        lines.append("=" * 78)

        self._write_log(lines)

    def _draw_final(self):
        """Final summary screen."""
        now = datetime.now(timezone.utc)
        uptime = now - self._start_time
        equity = self.daemon._equity
        pnl = self.daemon._pnl_today
        pnl_pct = (equity - self._start_equity) / self._start_equity * 100 if self._start_equity > 0 else 0
        closed = self.daemon.tracker.closed_trades()
        wins = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
        losses = sum(1 for t in closed if t.get("pnl_usd", 0) <= 0)
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        lines = [
            "",
            "=" * 78,
            f"  SESSION COMPLETE [{mode}]",
            "=" * 78,
            "",
            f"  Duration:     {uptime}",
            f"  Cycles:       {self._cycle_count}",
            f"  Ticks:        {self._tick_count}",
            "",
            f"  Start Equity: ${self._start_equity:,.2f}",
            f"  End Equity:   ${equity:,.2f}",
            f"  P&L:          ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            "",
            f"  Total Trades: {total}",
            f"  Wins:         {wins}",
            f"  Losses:       {losses}",
            f"  Win Rate:     {wr:.1f}%",
            "",
            f"  Open Positions: {len(self.daemon.tracker.open_positions())}",
            "",
        ]
        if closed:
            lines.append("  ALL TRADES:")
            for i, t in enumerate(closed):
                sym = t.get("symbol", "?")[:25]
                pnl_t = t.get("pnl_usd", 0)
                sign = "+" if pnl_t >= 0 else ""
                lines.append(f"    {i+1}. {sym:<25} ${sign}{pnl_t:.2f}")

        lines.append("")
        lines.append("=" * 78)
        self._write_log(lines)

    def run(self):
        """Main loop: signal cycles with 1-min live-price ticks between."""
        self.daemon._running = True

        def _shutdown(*_):
            self.daemon._running = False
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self._draw_opening()
        self.daemon.startup()

        # Fetch initial live prices
        self._fetch_live_prices()

        while self.daemon._running and self._cycle_count < self.max_cycles:
            # Signal cycle (fetches fresh 15m candles + indicators)
            self._cycle_count += 1
            signals = self.daemon.signal_cycle()
            self._last_signals = signals or []
            self._fetch_live_prices()
            self._draw_dashboard(is_signal_cycle=True, signals=self._last_signals)

            if self._cycle_count >= self.max_cycles:
                break

            # 15 x 1-minute ticks with live price updates
            for tick in range(15):
                if not self.daemon._running:
                    break
                time.sleep(60)
                self._tick_count += 1
                # Fetch fresh prices from exchange
                self._fetch_live_prices()
                # Update position prices + enforce stops
                for sym in MONITORED_PAIRS_15M:
                    if sym in self._live_prices:
                        coinbase_sym = PAIR_TO_COINBASE.get(sym, "")
                        # Update any positions that match this pair
                        for pos in self.daemon.tracker.open_positions():
                            if coinbase_sym in pos["symbol"]:
                                self.daemon.tracker.update_price(pos["symbol"], self._live_prices[sym])
                self.daemon._enforce_stops()
                self.daemon._update_equity()
                self._draw_dashboard(is_signal_cycle=False, signals=None)

        self._draw_final()


def main():
    parser = argparse.ArgumentParser(description="AlgoTrade Dashboard")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Paper trading (default)")
    parser.add_argument("--live", action="store_true", help="Live trading")
    parser.add_argument("--cycles", type=int, default=15, help="Signal cycles to run")
    args = parser.parse_args()

    dry_run = not args.live
    Dashboard(dry_run=dry_run, max_cycles=args.cycles).run()


if __name__ == "__main__":
    main()
