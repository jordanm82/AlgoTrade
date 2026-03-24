#!/usr/bin/env python3
"""Live ASCII trading dashboard — runs 15 signal cycles with 1-minute status updates.

Usage:
    python dashboard.py --dry-run     # paper trading (default)
    python dashboard.py               # live trading

Outputs status every minute so Claude can read and manage actively.
Runs 15 full signal cycles (~3.75 hours) then exits with summary.
"""
import sys
sys.path.insert(0, '.')
import argparse
import signal
import time
from datetime import datetime, timezone

import pandas as pd
from termcolor import colored

from config.production import (
    BB_GRID_CONFIG, LEVERAGE_PAIRS, MAX_CONCURRENT_POSITIONS,
    MONITORED_PAIRS_15M, PAIR_TO_COINBASE, POSITION_SIZE_PCT,
    RSI_MR_CONFIG, STOP_LOSS_PCT, MAX_LEVERAGE,
)
from cli.live_daemon import LiveDaemon


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

    def _draw_opening(self):
        """Opening status screen with full system info."""
        now = datetime.now(timezone.utc)
        mode = colored("DRY-RUN", "magenta") if self.dry_run else colored("LIVE", "green", attrs=["bold"])

        print(f"""
{'='*78}

     ___    __    _____ ____  ______ ____  ___    ____  ______
    /   |  / /   / ___// __ \\/_  __// __ \\/   |  / __ \\/ ____/
   / /| | / /   / __ \\/ / / / / /  / /_/ / /| | / / / / __/
  / ___ |/ /___/ /_/ / /_/ / / /  / _, _/ ___ |/ /_/ / /___
 /_/  |_/_____/\\____/\\____/ /_/  /_/ |_/_/  |_/_____/_____/

{'='*78}
  MODE: {mode}          {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
  EQUITY: ${self._start_equity:,.2f}
  MAX CYCLES: {self.max_cycles}  (~{self.max_cycles * 15} minutes)
{'='*78}

  STRATEGIES:
    1. BB Grid Long+Short (buy<BB_lower+RSI<35, sell>BB_mid)
       - 2x leverage: ATOM, FIL, DOT
       - 1x leverage: UNI, LTC, SHIB

    2. RSI Mean Reversion Long+Short (buy RSI<30, sell RSI>65)
       - 1x leverage: ATOM, FIL

  RISK CONTROLS:
    - Position size: {POSITION_SIZE_PCT:.0%} of equity (compounding)
    - Max leverage: {MAX_LEVERAGE}x
    - Stop-loss: {STOP_LOSS_PCT:.0%} hard stop
    - Max positions: {MAX_CONCURRENT_POSITIONS}
    - Daily drawdown halt: 5%

  MONITORED PAIRS:
    {'  '.join(MONITORED_PAIRS_15M)}

  BACKTEST PERFORMANCE (6 months, Sep 2025 - Mar 2026):
    ATOM BB Grid 2x:  87.7% WR | +437% return | 487 trades
    FIL  BB Grid 2x:  71.2% WR | +309% return | 472 trades
    ATOM RSI MR:      84.5% WR |  +70% return | 290 trades

{'='*78}
  Starting in 3 seconds...
{'='*78}""")
        sys.stdout.flush()
        time.sleep(3)

    def _draw_status(self, is_signal_cycle: bool = False, signals: list[dict] | None = None):
        """Draw compact 1-minute status update or full dashboard on signal cycles."""
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

        # Signal cycles show "SIGNAL CYCLE" in header, tick updates show "TICK"
        cycle_label = f"Cycle {self._cycle_count}/{self.max_cycles}" if is_signal_cycle else f"Tick {self._tick_count}"

        # Full dashboard (every tick and every signal cycle)
        print(f"""
{'='*78}
  ALGOTRADE [{mode}]  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {cycle_label}  |  Up {h}h{m}m
{'='*78}

  EQUITY: ${equity:,.2f}  |  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)
  POSITIONS: {len(positions)}/{MAX_CONCURRENT_POSITIONS}  |  EXPOSURE: ${exposure:,.2f}
  TRADES: {total}  |  W:{wins} L:{losses}  |  WIN RATE: {wr:.1f}%
""")
        # Pairs table
        print(f"  {'PAIR':<12} {'PRICE':>10} {'RSI':>6} {'BB_LOW':>10} {'BB_MID':>10} {'BB_UP':>10} {'SIGNAL':<16}")
        print(f"  {'-'*76}")
        for sym in MONITORED_PAIRS_15M:
            df = self.daemon._dataframes.get(sym)
            if df is None or len(df) == 0:
                print(f"  {sym:<12} {'--':>10}")
                continue
            last = df.iloc[-1]
            price = float(last["close"])
            rsi = float(last.get("rsi", 0)) if pd.notna(last.get("rsi")) else 0
            bb_l = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
            bb_m = float(last.get("bb_middle", 0)) if pd.notna(last.get("bb_middle")) else 0
            bb_u = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0

            sig_str = ""
            if price < bb_l and rsi < 35:
                sig_str = colored("BB+RSI BUY", "green")
            elif price > bb_u and rsi > 65:
                sig_str = colored("BB+RSI SHORT", "red")
            elif rsi < 30:
                sig_str = colored("RSI OVERSOLD", "yellow")
            elif rsi > 70:
                sig_str = colored("RSI OVERBOUGHT", "red")

            lev = colored("2x", "cyan") if sym in LEVERAGE_PAIRS else "1x"
            print(f"  {sym:<12} ${price:>9.4f} {rsi:5.1f} ${bb_l:>9.4f} ${bb_m:>9.4f} ${bb_u:>9.4f} {sig_str}")

        # Open positions
        if positions:
            print(f"\n  OPEN POSITIONS:")
            print(f"  {'KEY':<32} {'SIDE':<5} {'ENTRY':>10} {'NOW':>10} {'P&L':>10} {'STOP':>10}")
            print(f"  {'-'*80}")
            for p in positions:
                key = p["symbol"][:32]
                side = colored(p["side"], "green" if p["side"] == "BUY" else "red")
                upnl = p["unrealized_pnl"]
                pnl_c = colored(f"${upnl:+.2f}", "green" if upnl >= 0 else "red")
                print(f"  {key:<32} {side:<14} ${p['entry_price']:>9.4f} ${p['current_price']:>9.4f} {pnl_c:>19} ${p['stop_price']:>9.4f}")

        # Signals (show current cycle signals, or last known signals on ticks)
        sigs = signals if is_signal_cycle else self._last_signals
        if sigs:
            print(f"\n  SIGNALS THIS CYCLE ({len(sigs)}):")
            for s in sigs[-8:]:
                action = s.get("action", "?")
                sym = s.get("symbol", "?")
                price = s.get("price", 0)
                rsi_v = s.get("rsi", 0)
                strat = s.get("strategy", "?")
                if "BUY" in action:
                    print(colored(f"    >> {action} {sym} @ ${price:.4f} RSI={rsi_v:.1f} [{strat}]", "green"))
                elif "SHORT" in action:
                    print(colored(f"    >> {action} {sym} @ ${price:.4f} RSI={rsi_v:.1f} [{strat}]", "red"))
                elif "CLOSE" in action or "COVER" in action:
                    print(colored(f"    >> {action} {sym} @ ${price:.4f} [{strat}]", "cyan"))
        else:
            print(f"\n  No signals this cycle")

        # Recent closed trades
        if closed:
            print(f"\n  CLOSED TRADES (last 5):")
            for t in closed[-5:]:
                sym = t.get("symbol", "?")[:25]
                side = t.get("side", "?")
                pnl_t = t.get("pnl_usd", 0)
                pnl_c = colored(f"${pnl_t:+.2f}", "green" if pnl_t >= 0 else "red")
                print(f"    {sym:<25} {side:<5} {pnl_c}")

        remaining = self.max_cycles - self._cycle_count
        if is_signal_cycle:
            print(f"\n  Next signal cycle in ~15 min | {remaining} cycles remaining")
        else:
            mins_to_cycle = 15 - (self._tick_count % 15)
            print(f"\n  Next signal cycle in ~{mins_to_cycle} min | {remaining} cycles remaining")
        print(f"{'='*78}")
        sys.stdout.flush()

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
        print(f"""
{'='*78}
  SESSION COMPLETE [{mode}]
{'='*78}

  Duration:     {uptime}
  Cycles:       {self._cycle_count}
  Ticks:        {self._tick_count}

  Start Equity: ${self._start_equity:,.2f}
  End Equity:   ${equity:,.2f}
  P&L:          ${pnl:+,.2f} ({pnl_pct:+.2f}%)

  Total Trades: {total}
  Wins:         {wins}
  Losses:       {losses}
  Win Rate:     {wr:.1f}%

  Open Positions: {len(self.daemon.tracker.open_positions())}
""")
        if closed:
            print("  ALL TRADES:")
            for i, t in enumerate(closed):
                sym = t.get("symbol", "?")[:25]
                pnl_t = t.get("pnl_usd", 0)
                pnl_c = colored(f"${pnl_t:+.2f}", "green" if pnl_t >= 0 else "red")
                print(f"    {i+1}. {sym:<25} {pnl_c}")

        print(f"\n{'='*78}")
        sys.stdout.flush()

    def run(self):
        """Run max_cycles signal cycles with 1-minute status ticks between."""
        self.daemon._running = True

        def _shutdown(*_):
            self.daemon._running = False
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Opening screen
        self._draw_opening()

        # Startup data fetch
        self.daemon.startup()

        # Main loop: run cycles with minute ticks between
        while self.daemon._running and self._cycle_count < self.max_cycles:
            # Signal cycle
            self._cycle_count += 1
            signals = self.daemon.signal_cycle()
            self._last_signals = signals or []
            self._draw_status(is_signal_cycle=True, signals=self._last_signals)

            if self._cycle_count >= self.max_cycles:
                break

            # Wait ~15 minutes with 1-minute full dashboard redraws
            for tick in range(15):
                if not self.daemon._running:
                    break
                time.sleep(60)
                self._tick_count += 1
                self.daemon.tick()
                self._draw_status(is_signal_cycle=False, signals=None)

        # Final summary
        self._draw_final()


def main():
    parser = argparse.ArgumentParser(description="AlgoTrade Dashboard")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--cycles", type=int, default=15, help="Number of signal cycles")
    args = parser.parse_args()

    dry_run = not args.live
    dashboard = Dashboard(dry_run=dry_run, max_cycles=args.cycles)
    dashboard.run()


if __name__ == "__main__":
    main()
