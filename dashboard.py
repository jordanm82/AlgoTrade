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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from termcolor import colored

from config.pair_config import (
    ALL_PAIRS, COINBASE_MAP, PAIR_CONFIG, get_pair_config,
)
from config.production import (
    MAX_CONCURRENT_POSITIONS, POSITION_SIZE_PCT, STOP_LOSS_PCT, MAX_LEVERAGE,
)

# Derive leverage pairs from per-pair config
LEVERAGE_PAIRS = [sym for sym, cfg in PAIR_CONFIG.items() if cfg["leverage"] > 1]
from cli.live_daemon import LiveDaemon
from cli.kalshi_daemon import KalshiDaemon

LOG_FILE = Path("data/store/dashboard.log")


class Dashboard:
    def __init__(self, dry_run: bool = True, max_cycles: int = 15, kalshi_only: bool = False, predictor_version: str = "v1"):
        if kalshi_only:
            self.daemon = KalshiDaemon(dry_run=dry_run, predictor_version=predictor_version)
            self.daemon.kalshi_only = True  # for dashboard rendering checks
        else:
            self.daemon = LiveDaemon(dry_run=dry_run, kalshi_only=kalshi_only, predictor_version=predictor_version)
        self.dry_run = dry_run
        self.max_cycles = max_cycles
        self._cycle_count = 0
        self._tick_count = 0
        self._start_time = datetime.now(timezone.utc)
        self._start_equity = self.daemon._equity
        self._last_signals: list[dict] = []
        # Live ticker prices updated every minute
        self._live_prices: dict[str, float] = {}
        # Timestamp of last signal cycle (for indicator freshness label)
        self._last_signal_time: datetime | None = None
        # Live account balances
        self._coinbase_balance: float = 0.0
        self._kalshi_balance: float = 0.0
        self._start_coinbase: float = 0.0
        self._start_kalshi: float = 0.0
        self._fetch_account_balances()
        self._start_coinbase = self._coinbase_balance
        self._start_kalshi = self._kalshi_balance

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _fetch_account_balances(self):
        """Fetch live balances from both Coinbase and Kalshi."""
        # Coinbase
        try:
            if self.daemon.executor:
                balances = self.daemon.executor.get_balances()
                self._coinbase_balance = sum(balances.values())
            elif self.dry_run:
                self._coinbase_balance = self.daemon._equity
        except Exception:
            pass  # keep last known

        # Kalshi
        try:
            from exchange.kalshi import KalshiClient
            from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
            client = KalshiClient(
                api_key_id=KALSHI_API_KEY_ID,
                private_key_path=str(KALSHI_KEY_FILE),
                demo=self.dry_run,
            )
            resp = client.get_balance()
            self._kalshi_balance = resp.get("balance", 0) / 100  # cents to dollars
        except Exception:
            pass  # keep last known

    def _fmt_price(self, price: float) -> str:
        """Format price with appropriate precision for micro-priced tokens."""
        if price == 0:
            return "    $0.00"
        elif price < 0.001:
            return f"${price:>9.8f}"
        elif price < 1:
            return f"${price:>9.6f}"
        elif price < 100:
            return f"${price:>9.4f}"
        else:
            return f"${price:>9.2f}"

    def _market_regime(self, df: pd.DataFrame) -> str:
        """Compute market regime from last 96 candles (24h on 15m)."""
        if df is None or len(df) < 96:
            return "N/A"
        closes = df["close"].iloc[-96:]
        sma_96 = float(closes.mean())
        current = float(df.iloc[-1]["close"])
        if sma_96 == 0:
            return "N/A"
        pct_diff = (current - sma_96) / sma_96
        if pct_diff > 0.03:
            return "TRENDING UP"
        elif pct_diff < -0.03:
            return "TRENDING DN"
        else:
            return "RANGING"

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
        for sym in ALL_PAIRS:
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
            f"  COINBASE: ${self._start_coinbase:,.2f}  |  KALSHI: ${self._start_kalshi:,.2f}  |  TOTAL: ${self._start_coinbase + self._start_kalshi:,.2f}",
            f"  MAX CYCLES: {self.max_cycles}  (~{self.max_cycles * 15} minutes)",
            "=" * 78,
            "",
            "  STRATEGIES (per-pair optimized thresholds):",
            "    1. BB Grid Long+Short — per-pair RSI buy/short thresholds",
            "    2. RSI Mean Reversion Long+Short — per-pair oversold/overbought",
            "",
            "  RISK CONTROLS:",
            f"    - Position size: {POSITION_SIZE_PCT:.0%} of equity (compounding)",
            f"    - Max leverage: {MAX_LEVERAGE}x",
            f"    - Stop-loss: {STOP_LOSS_PCT:.0%} hard stop",
            f"    - Max positions: {MAX_CONCURRENT_POSITIONS}",
            "    - Daily drawdown halt: 5%",
            "",
            "  MONITORED PAIRS:",
            f"    {'  '.join(ALL_PAIRS)}",
            "",
            "  BACKTEST PERFORMANCE (6 months, Sep 2025 - Mar 2026):",
            "    ATOM BB Grid 2x:  88.2% WR | +592% return | 551 trades",
            "    FIL  BB Grid 2x:  71.5% WR | +325% return | 492 trades",
            "    DOT  BB Grid 2x:  76.2% WR | +101% return | 564 trades",
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

        # Refresh account balances every tick
        self._fetch_account_balances()

        equity = self.daemon._equity
        if self.daemon.tracker is not None:
            positions = self.daemon.tracker.open_positions()
            exposure = self.daemon.tracker.total_exposure()
            closed = self.daemon.tracker.closed_trades()
        else:
            positions = []
            exposure = 0
            closed = []
        wins = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
        losses = sum(1 for t in closed if t.get("pnl_usd", 0) <= 0)
        total = wins + losses
        wr = (wins / total * 100) if total > 0 else 0

        # Daily P&L: realized (closed trades) + unrealized (open positions)
        realized_pnl = sum(t.get("pnl_usd", 0) for t in closed)
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)
        total_pnl = realized_pnl + unrealized_pnl

        # Coinbase P&L from account balance change
        cb_pnl = self._coinbase_balance - self._start_coinbase
        # Kalshi P&L from account balance change
        kl_pnl = self._kalshi_balance - self._start_kalshi
        # Proper P&L: account balance + position value (not just cash)
        position_value = sum(p.get("size_usd", 0) + p.get("unrealized_pnl", 0) for p in positions)
        combined_start = self._start_coinbase + self._start_kalshi
        total_value = self._coinbase_balance + self._kalshi_balance + position_value
        combined_pnl = total_value - combined_start
        combined_pct = (combined_pnl / combined_start * 100) if combined_start > 0 else 0

        label = f"Cycle {self._cycle_count}/{self.max_cycles}" if is_signal_cycle else f"Tick {self._tick_count}"

        if self.daemon.kalshi_only:
            # Kalshi-only mode — show Kalshi stats, not Coinbase
            # Get real Kalshi balance even in dry-run
            kalshi_bal = self._kalshi_balance
            if kalshi_bal == 0 and not hasattr(self, '_kalshi_balance_fetched'):
                try:
                    self.daemon._init_kalshi_client()
                    if self.daemon.kalshi_client:
                        bal_resp = self.daemon.kalshi_client.get_balance()
                        kalshi_bal = bal_resp.get("balance", 0) / 100
                        self._kalshi_balance = kalshi_bal
                        self._kalshi_balance_fetched = True
                except Exception:
                    pass

            dw = self.daemon._session_wins
            dl = self.daemon._session_losses
            dt = dw + dl
            dwr = dw / dt * 100 if dt > 0 else 0
            pending = len(self.daemon._pending_bets)
            from config.production import MAX_CONCURRENT_KALSHI_BETS

            lines = [
                "",
                "=" * 78,
                f"  K15 UPDOWN [{mode}] V3  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  Tick {self._tick_count}  |  Up {h}h{m}m",
                "=" * 78,
                "",
                f"  KALSHI BALANCE: ${kalshi_bal:,.2f}  |  BETS: {pending} pending settlement",
            ]
            session_pnl = sum(b.get("pnl_dollars", 0) for b in self.daemon._completed_bets)
            pnl_sign = "+" if session_pnl >= 0 else ""
            lines.append(
                f"  SESSION: W:{dw} L:{dl} WR:{dwr:.0f}%  |  P&L: ${pnl_sign}{session_pnl:.2f}  |  Max bets: {MAX_CONCURRENT_KALSHI_BETS}"
            )
            lines.extend([
                "",
            ])
        else:
            lines = [
                "",
                "=" * 78,
                f"  ALGOTRADE [{mode}]  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC  |  {label}  |  Up {h}h{m}m",
                "=" * 78,
                "",
                f"  COINBASE: ${self._coinbase_balance:,.2f} + positions ${position_value:,.2f}  |  KALSHI: ${self._kalshi_balance:,.2f}  |  TOTAL: ${total_value:,.2f}",
                f"  DAILY P&L: ${combined_pnl:+,.2f} ({combined_pct:+.2f}%)  |  Realized: ${realized_pnl:+,.2f}  |  Unrealized: ${unrealized_pnl:+,.2f}",
                f"  POSITIONS: {len(positions)}/{MAX_CONCURRENT_POSITIONS}  |  EXPOSURE: ${exposure:,.2f}  |  TRADES: {total} (W:{wins} L:{losses} WR:{wr:.0f}%)",
                "",
            ]

        # Indicator freshness label
        if self._last_signal_time:
            delta = now - self._last_signal_time
            mins_ago = int(delta.total_seconds() // 60)
            mins_to_next = max(0, 15 - mins_ago)
            lines.append(f"  PRICES: live | INDICATORS: {mins_ago} min ago (next refresh in ~{mins_to_next} min)")
        elif not self.daemon.kalshi_only:
            lines.append(f"  PRICES: live | INDICATORS: pending first cycle")
        lines.append("")

        # Pairs table with LIVE prices (skip in kalshi-only mode)
        if not self.daemon.kalshi_only:
            lines.append(f"  {'PAIR':<12} {'PRICE':>10} {'RSI':>6} {'BB_LOW':>10} {'BB_MID':>10} {'BB_UP':>10} {'REGIME':<13} {'SIGNAL':<16}")
            lines.append(f"  {'-'*91}")
        for sym in (ALL_PAIRS if not self.daemon.kalshi_only else []):
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
            regime = self._market_regime(df)

            pcfg = get_pair_config(sym)
            sig_str = ""
            if price < bb_l and rsi < pcfg["bb_rsi_buy"]:
                sig_str = "** BB+RSI BUY **"
            elif price > bb_u and rsi > pcfg["bb_rsi_short"]:
                sig_str = "** BB+RSI SHORT **"
            elif rsi < pcfg["rsi_mr_oversold"]:
                sig_str = "! RSI OVERSOLD"
            elif rsi > pcfg["rsi_mr_overbought"]:
                sig_str = "! RSI OVERBOUGHT"

            lev_tag = f" [{pcfg['leverage']}x]" if pcfg["leverage"] > 1 else ""
            price_str = self._fmt_price(price)
            bb_l_str = self._fmt_price(bb_l)
            bb_m_str = self._fmt_price(bb_m)
            bb_u_str = self._fmt_price(bb_u)
            lines.append(f"  {sym:<12} {price_str} {rsi:5.1f} {bb_l_str} {bb_m_str} {bb_u_str} {regime:<13}{sig_str}{lev_tag}")

        # Open positions with live P&L (skip in kalshi-only)
        if positions and not self.daemon.kalshi_only:
            lines.append("")
            lines.append("  OPEN POSITIONS:")
            lines.append(f"  {'KEY':<32} {'SIDE':<5} {'ENTRY':>10} {'NOW':>10} {'P&L':>10} {'STOP':>10}")
            lines.append(f"  {'-'*80}")
            for p in positions:
                key = p["symbol"][:32]
                side = p["side"]
                upnl = p["unrealized_pnl"]
                pnl_sign = "+" if upnl >= 0 else ""
                entry_str = self._fmt_price(p['entry_price'])
                now_str = self._fmt_price(p['current_price'])
                stop_str = self._fmt_price(p['stop_price'])
                lines.append(
                    f"  {key:<32} {side:<5} "
                    f"{entry_str} {now_str} "
                    f"${pnl_sign}{upnl:.2f}{'':>4} {stop_str}"
                )

        # Signals (with confluence detection) — skip in kalshi-only
        sigs = signals if is_signal_cycle else self._last_signals
        if sigs and not self.daemon.kalshi_only:
            lines.append("")
            # Group signals by (symbol, action) for confluence detection
            grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for s in sigs:
                key = (s.get("symbol", "?"), s.get("action", "?"))
                grouped[key].append(s)

            confluence_lines = []
            for (sym, action), group in grouped.items():
                first = group[0]
                price_val = first.get("price", 0)
                rsi_v = first.get("rsi", 0)
                price_str = self._fmt_price(price_val)
                if len(group) > 1:
                    strats = " + ".join(s.get("strategy", "?") for s in group)
                    confluence_lines.append(
                        f"    >> {action} {sym} @ {price_str} RSI={rsi_v:.1f} "
                        f"[{len(group)}x CONFLUENCE: {strats}]"
                    )
                else:
                    strat = first.get("strategy", "?")
                    confluence_lines.append(
                        f"    >> {action} {sym} @ {price_str} RSI={rsi_v:.1f} [{strat}]"
                    )

            lines.append(f"  SIGNALS ({len(sigs)}):")
            for cl in confluence_lines[-8:]:
                lines.append(cl)
        elif not self.daemon.kalshi_only:
            lines.append("")
            lines.append("  No signals this cycle")

        # Closed trades (skip in kalshi-only)
        if closed and not self.daemon.kalshi_only:
            lines.append("")
            lines.append(f"  CLOSED TRADES (last 5):")
            for t in closed[-5:]:
                sym = t.get("symbol", "?")[:25]
                side = t.get("side", "?")
                pnl_t = t.get("pnl_usd", 0)
                sign = "+" if pnl_t >= 0 else ""
                lines.append(f"    {sym:<25} {side:<6} ${sign}{pnl_t:.2f}")

        # Kalshi predictions
        kalshi_preds = getattr(self.daemon, "kalshi_predictions", [])
        if kalshi_preds:
            lines.append("")
            lines.append("  KALSHI PREDICTIONS:")
            lines.append(f"    {'ASSET':<5} {'SIDE':<5} {'PROB':>5}  {'PRICE':>12} {'TARGET':>12} {'DIST':>8}  {'MDL':<4}{'STATE':<18}")
            lines.append(f"    {'─'*70}")
            for pred in kalshi_preds:
                asset = pred.get("asset", "?")
                direction = pred.get("direction", "--")
                conf = pred.get("confidence", 0)
                state = pred.get("state", "")

                # Get current price + target from pending signals
                current_str = ""
                target_str = ""
                dist_str = ""
                if hasattr(self.daemon, '_kalshi_pending_signals'):
                    pending = self.daemon._kalshi_pending_signals.get(asset, {})
                    target = pending.get("strike_price", 0)
                    if target:
                        target_str = f"${target:,.2f}"
                        # Get current price from Coinbase (closer to BRTI settlement)
                        symbol = f"{asset}/USDT"
                        cb_price = self.daemon._get_coinbase_price(symbol) if hasattr(self.daemon, '_get_coinbase_price') else None
                        if cb_price:
                            current = cb_price
                        else:
                            # Fallback to cached candle data
                            df = self.daemon._dataframes.get(symbol) or self.daemon._kalshi_cached_dataframes.get(symbol)
                            current = float(df.iloc[-1]["close"]) if df is not None and len(df) > 0 else 0
                        if current:
                            current_str = f"${current:,.2f}"
                            diff = current - target
                            sign = "+" if diff >= 0 else ""
                            dist_str = f"{sign}{diff:.2f}"

                side_display = direction if direction != "--" else "--"
                prob_display = f"{conf}%" if conf > 0 else "--"

                # Show which model produced the prediction
                reason = pred.get("reason", "")
                model_tag = "KNN" if "knn" in reason.lower() else "TBL"

                lines.append(
                    f"    {asset:<5} {side_display:<5} {prob_display:>5}  "
                    f"{current_str:>12} {target_str:>12} {dist_str:>8}  "
                    f"{model_tag:<4}{state:<18}"
                )

        # Footer — show time to next Kalshi window
        now_utc = datetime.now(timezone.utc)
        min_in_window = now_utc.minute % 15
        mins_to_next_window = 15 - min_in_window
        window_start = now_utc.minute - min_in_window
        window_end = (window_start + 15) % 60
        next_eval_mins = 5 - (min_in_window % 5)
        if next_eval_mins == 0:
            next_eval_mins = 5
        lines.append(f"\n  Window: :{window_start:02d}-:{window_end:02d} (min {min_in_window}/15)  |  "
                     f"Next eval: ~{next_eval_mins} min  |  Next window: ~{mins_to_next_window} min")
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

        if self.daemon.kalshi_only:
            dw = self.daemon._session_wins
            dl = self.daemon._session_losses
            dt = dw + dl
            dwr = dw / dt * 100 if dt > 0 else 0
            pending = len(self.daemon._pending_bets)
            total_placed = self.daemon._session_bets_placed

            # Before showing summary, do a final settlement check
            self.daemon._check_dryrun_settlements()
            # Re-count after final check
            dw = self.daemon._session_wins
            dl = self.daemon._session_losses
            dt = dw + dl
            dwr = dw / dt * 100 if dt > 0 else 0
            pending = len(self.daemon._pending_bets)

            # Calculate total P&L
            total_pnl = sum(b.get("pnl_dollars", 0) for b in self.daemon._completed_bets)
            total_risked = sum(b.get("cost_cents", 0) for b in self.daemon._completed_bets) / 100
            roi = (total_pnl / total_risked * 100) if total_risked > 0 else 0

            lines = [
                "",
                "=" * 78,
                f"  K15 SESSION COMPLETE [{mode}]",
                "=" * 78,
                "",
                f"  Duration:       {uptime}",
                f"  Ticks:          {self._tick_count}",
                "",
                f"  Bets Placed:    {total_placed}",
                f"  Settled:        {dt}",
                f"  Wins:           {dw}",
                f"  Losses:         {dl}",
                f"  Win Rate:       {dwr:.1f}%",
                "",
                f"  Total Risked:   ${total_risked:,.2f}",
                f"  Net P&L:        ${total_pnl:+,.2f}",
                f"  ROI:            {roi:+.1f}%",
            ]
            if pending > 0:
                lines.append(f"\n  Still Pending:  {pending} (not yet settled)")
            lines.append("")

            if self.daemon._completed_bets:
                lines.append("  SETTLED BETS:")
                lines.append(f"    {'':>4} {'ASSET':<5} {'SIDE':<4} {'ENTRY':>6} {'QTY':>4} {'STRIKE':>12} {'SETTLED':>12} {'RESULT':>6} {'P&L':>8}")
                lines.append(f"    {'─'*65}")
                for i, r in enumerate(self.daemon._completed_bets, 1):
                    result_tag = "WIN" if r["result"] == "WIN" else "LOSS"
                    entry = r.get("contract_price", 0)
                    qty = r.get("count", 1)
                    pnl = r.get("pnl_dollars", 0)
                    pnl_sign = "+" if pnl >= 0 else ""
                    lines.append(
                        f"    {i:>3}. {r['asset']:<5} {r['direction']:<4} "
                        f"{entry:>4}c x{qty:<3} "
                        f"${r['strike']:>10,.2f} ${r.get('settle_price',0):>10,.2f} "
                        f" {result_tag:<4} ${pnl_sign}{pnl:.2f}"
                    )
                lines.append("")

            if pending > 0:
                lines.append("  PENDING BETS (not yet settled):")
                for b in self.daemon._pending_bets:
                    lines.append(
                        f"    {b['asset']:<5} {b['direction']:<4} "
                        f"strike=${b['strike']:,.2f} settles={b['settle_time'].strftime('%H:%M:%S')} UTC"
                    )
                lines.append("")
        else:
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

        def _handle_sigint(*_):
            print("\n\n  [SHUTDOWN] Ctrl+C received — stopping...")
            self.daemon._running = False
        signal.signal(signal.SIGINT, _handle_sigint)

        if self.daemon.kalshi_only:
            # ── Kalshi-only mode: simple 1-minute tick loop ──
            # Fetch 15m data (needed for scoring before first signal cycle)
            self.daemon._fetch_all()

            # Run initial eval immediately
            try:
                self.daemon._kalshi_eval()
                self.daemon._last_kalshi_eval = time.time()
            except Exception as e:
                print(f"  [KALSHI EVAL ERR] {e}")
            self._draw_dashboard(is_signal_cycle=True, signals=None)

            total_ticks = self.max_cycles * 15  # convert cycles to minutes
            tick = 0
            while self.daemon._running and tick < total_ticks:
                # Sleep 1 minute in 1-second chunks
                for _ in range(60):
                    if not self.daemon._running:
                        break
                    time.sleep(1)
                if not self.daemon._running:
                    break
                tick += 1
                self._tick_count += 1

                # Kalshi eval every tick — let the eval method decide if it's time
                current_minute = datetime.now(timezone.utc).minute
                now_ts = time.time()
                should_eval = (current_minute % 5 == 1 and now_ts - self.daemon._last_kalshi_eval >= 240) \
                           or (current_minute % 15 == 12 and now_ts - self.daemon._last_kalshi_eval >= 50) \
                           or (current_minute % 15 == 1 and now_ts - self.daemon._last_kalshi_eval >= 50)
                if should_eval:
                    try:
                        self.daemon._kalshi_eval()
                    except Exception as e:
                        print(f"  [KALSHI EVAL ERR] {e}")
                    self.daemon._last_kalshi_eval = now_ts

                self.daemon._update_equity()
                # Check bet settlements (both live and dry-run)
                self.daemon._check_dryrun_settlements()
                self._draw_dashboard(is_signal_cycle=False, signals=None)

        else:
            # ── Full mode: 15-minute signal cycles with ticks ──
            while self.daemon._running and self._cycle_count < self.max_cycles:
                self._cycle_count += 1
                signals = self.daemon.signal_cycle()
                self._last_signals = signals or []
                self._last_signal_time = datetime.now(timezone.utc)
                self._fetch_live_prices()
                try:
                    self.daemon._kalshi_eval()
                    self.daemon._last_kalshi_eval = time.time()
                except Exception as e:
                    print(f"  [KALSHI EVAL ERR] {e}")
                self._draw_dashboard(is_signal_cycle=True, signals=self._last_signals)

                if self._cycle_count >= self.max_cycles:
                    break

                for tick in range(15):
                    if not self.daemon._running:
                        break
                    for _ in range(60):
                        if not self.daemon._running:
                            break
                        time.sleep(1)
                    self._tick_count += 1
                    self._fetch_live_prices()
                    for sym in ALL_PAIRS:
                        if sym in self._live_prices:
                            coinbase_sym = COINBASE_MAP.get(sym, "")
                            for pos in self.daemon.tracker.open_positions():
                                if coinbase_sym in pos["symbol"]:
                                    self.daemon.tracker.update_price(pos["symbol"], self._live_prices[sym])
                    self.daemon._enforce_stops()
                    self.daemon._update_equity()
                    current_minute = datetime.now(timezone.utc).minute
                    now_ts = time.time()
                    should_eval = (current_minute % 5 == 1 and now_ts - self.daemon._last_kalshi_eval >= 240) \
                               or (current_minute % 15 == 12 and now_ts - self.daemon._last_kalshi_eval >= 50) \
                               or (current_minute % 15 == 1 and now_ts - self.daemon._last_kalshi_eval >= 50)
                    if should_eval:
                        try:
                            self.daemon._kalshi_eval()
                        except Exception as e:
                            print(f"  [KALSHI EVAL ERR] {e}")
                        self.daemon._last_kalshi_eval = now_ts
                    self._draw_dashboard(is_signal_cycle=False, signals=None)

        self._draw_final()


def main():
    parser = argparse.ArgumentParser(description="AlgoTrade Dashboard")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Paper trading (default)")
    parser.add_argument("--live", action="store_true", help="Live trading")
    parser.add_argument("--cycles", type=int, default=15, help="Signal cycles to run")
    parser.add_argument("--k15", action="store_true", default=True, help="Run K15 Kalshi 15m prediction bot (default)")
    parser.add_argument("--spot", action="store_true", help="Run Coinbase spot trading bot (BB Grid + RSI MR)")
    parser.add_argument("--simple", action="store_true", help="Force plain text output (for logs/pipes)")
    args = parser.parse_args()

    # --spot overrides the default --k15
    kalshi_only = not args.spot

    dry_run = not args.live
    predictor_version = "v3" if kalshi_only else "v1"
    Dashboard(dry_run=dry_run, max_cycles=args.cycles, kalshi_only=kalshi_only, predictor_version=predictor_version).run()


if __name__ == "__main__":
    main()
