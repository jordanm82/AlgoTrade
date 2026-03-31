#!/usr/bin/env python3
"""Rich-based K15 dashboard — predictions, orders, live positions, activity log.

Usage:
    ./venv/bin/python dashboard_rich.py --dry-run
    ./venv/bin/python dashboard_rich.py --live
    ./venv/bin/python dashboard_rich.py --arb --dry-run
"""
import argparse
import signal as sig
import sys
import time
from collections import deque
from datetime import datetime, timezone

sys.path.insert(0, ".")

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.kalshi_daemon import KalshiDaemon

MAX_LOG_LINES = 50
POSITION_REFRESH_INTERVAL = 15  # seconds — matches resting order monitoring


class RichDashboard:
    def __init__(self, dry_run: bool = True, arb_mode: bool = False, max_cycles: int = 96,
                 demo: bool = False, max_bets: int = 0, max_size_pct: float = 0):
        self.arb_mode = arb_mode
        self.dry_run = dry_run
        self.demo = demo
        self.max_cycles = max_cycles

        if arb_mode:
            from cli.kalshi_arb import KalshiArbDaemon
            self.daemon = KalshiArbDaemon(dry_run=dry_run)
        else:
            self.daemon = KalshiDaemon(
                dry_run=dry_run or demo, predictor_version="v3", demo=demo,
                max_bets=max_bets, max_size_pct=max_size_pct,
            )
        self.daemon.kalshi_only = True

        self._start_time = datetime.now(timezone.utc)
        self._tick_count = 0
        self._log_lines: deque[Text] = deque(maxlen=MAX_LOG_LINES)
        self._kalshi_balance: float = 0.0
        self._console = Console()

        # Live position tracking (refreshed every 5s)
        self._live_positions: list[dict] = []  # current open positions with live prices
        self._last_position_refresh: float = 0

        # Order tracking
        self._order_summary: list[dict] = []  # resting + filled orders this session

        self._install_log_capture()

    def _install_log_capture(self):
        """Redirect daemon print() calls to our activity log."""
        import builtins

        def captured_print(*args, **kwargs):
            text = " ".join(str(a) for a in args)
            if not text.strip():
                return
            import re
            clean = re.sub(r'\x1b\[[0-9;]*m', '', text).strip()
            if not clean:
                return

            style = "white"
            if "[SIGNAL]" in clean or "[RESTING FILL]" in clean:
                style = "cyan bold"
            elif "[KALSHI BET]" in clean or "FILLED" in clean:
                style = "green bold"
            elif "[KALSHI WAIT]" in clean:
                style = "yellow"
            elif "[SETTLED]" in clean and "WIN" in clean:
                style = "green"
            elif "[SETTLED]" in clean and "LOSS" in clean:
                style = "red"
            elif "[ARB]" in clean and "BOTH" in clean:
                style = "green bold"
            elif "[ARB]" in clean and "SELL" in clean:
                style = "yellow"
            elif "ERR" in clean or "ERROR" in clean:
                style = "red bold"
            elif "[STARTUP]" in clean or "[MODEL]" in clean:
                style = "cyan"
            elif "====" in clean or "----" in clean:
                return

            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._log_lines.append(Text(f"{ts} {clean}", style=style))

        builtins.print = captured_print

    def _fetch_balance(self):
        try:
            if self.daemon.dry_run and not self.demo:
                # Dry-run: use compounding simulated balance
                self._kalshi_balance = self.daemon._dry_balance_cents / 100
            else:
                # Live + demo: query actual exchange balance
                self.daemon._init_kalshi_client()
                if self.daemon.kalshi_client:
                    bal = self.daemon.kalshi_client.get_balance()
                    self._kalshi_balance = bal.get("balance", 0) / 100
        except Exception as e:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._log_lines.append(Text(f"{ts} [BAL ERR] {e}", style="red"))
            pass

    def _refresh_positions(self):
        """Refresh positions — from Kalshi API (live/demo) or pending bets (dry-run)."""
        now = time.time()
        if now - self._last_position_refresh < POSITION_REFRESH_INTERVAL:
            return
        self._last_position_refresh = now

        self.daemon._init_kalshi_client()
        if not self.daemon.kalshi_client:
            return

        positions = []

        try:
            # Get ACTUAL open positions from Kalshi — source of truth
            kalshi_positions = self.daemon.kalshi_client.get_positions()

            for kp in kalshi_positions:
                ticker = kp.get("ticker", "")
                count = int(float(kp.get("position_fp", 0)))
                cost = float(kp.get("total_traded_dollars", 0))
                if count <= 0:
                    continue

                # Parse asset from ticker (e.g., KXETH15M-26MAR271400-00 → ETH)
                asset = ""
                for a in ["ETH", "SOL", "XRP", "BNB", "BTC"]:
                    if f"KX{a}" in ticker:
                        asset = a
                        break
                if not asset:
                    continue

                # Determine side from our pending bets or default to YES
                side = "YES"
                entry = int(cost / count * 100) if count > 0 else 0
                for bet in getattr(self.daemon, '_pending_bets', []):
                    if bet.get("asset") == asset and bet.get("count", 0) > 0:
                        side = bet.get("side", "yes").upper()
                        entry = bet.get("contract_price", bet.get("fill_price", entry))
                        break

                # Get live market prices
                symbol = f"{asset}/USDT"
                series_map = getattr(self.daemon, 'KALSHI_PAIRS', {})
                series = series_map.get(symbol, "")
                yes_bid, no_bid = 0, 0
                strike = 0
                winning = None

                if series:
                    try:
                        mkts = self.daemon.kalshi_client.get_markets(
                            series_ticker=series, status="open"
                        )
                        if mkts:
                            m = max(mkts, key=lambda x: x.get("close_time", ""))
                            yb = m.get("yes_bid_dollars")
                            nb = m.get("no_bid_dollars")
                            yes_bid = int(float(yb) * 100) if yb else 0
                            no_bid = int(float(nb) * 100) if nb else 0
                            # Get strike for winning determination
                            strike_val, _, _ = self.daemon._get_kalshi_strike(series)
                            if strike_val:
                                strike = float(strike_val)
                    except Exception:
                        pass

                # Determine who's winning from Coinbase price vs strike
                if strike:
                    cb_price = self.daemon._get_coinbase_price(symbol)
                    if cb_price:
                        winning = "YES" if cb_price >= strike else "NO"

                positions.append({
                    "asset": asset,
                    "side": side,
                    "entry": entry,
                    "count": count,
                    "status": "OPEN",
                    "strike": strike,
                    "ticker": ticker,
                    "current_yes": yes_bid,
                    "current_no": no_bid,
                    "winning": winning,
                })

            # Also check resting orders from exchange — only for current/future markets
            now_utc = datetime.now(timezone.utc)
            resting = self.daemon.kalshi_client.get_orders(status="resting")
            for o in resting:
                ticker = o.get("ticker", "")

                # Skip orders for markets that have already closed
                expiration = o.get("expiration_time") or o.get("close_time", "")
                if expiration:
                    try:
                        exp_dt = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
                        if exp_dt < now_utc:
                            # Stale order from settled market — try to cancel it
                            try:
                                order_id = o.get("order_id")
                                if order_id:
                                    self.daemon.kalshi_client.cancel_order_safe(order_id)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                asset = ""
                for a in ["ETH", "SOL", "XRP", "BNB", "BTC"]:
                    if f"KX{a}" in ticker:
                        asset = a
                        break
                if not asset:
                    continue

                side = o.get("side", "yes").upper()
                if side == "YES":
                    price_raw = o.get("yes_price") or o.get("yes_price_dollars") or 0
                else:
                    price_raw = o.get("no_price") or o.get("no_price_dollars") or 0
                price = float(price_raw) * 100 if float(price_raw) < 1.5 else float(price_raw)
                count_raw = o.get("remaining_count_fp") or o.get("remaining_count") or o.get("count") or 0
                count = int(float(count_raw))
                filled_raw = o.get("fill_count_fp") or o.get("fill_count") or 0
                filled = int(float(filled_raw))

                status = "PARTIAL" if filled > 0 and count > 0 else "RESTING"

                positions.append({
                    "asset": asset,
                    "side": side,
                    "entry": int(price),
                    "count": count + filled,
                    "status": status,
                    "strike": 0,
                    "ticker": ticker,
                    "current_yes": 0,
                    "current_no": 0,
                    "winning": None,
                })

        except Exception as e:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._log_lines.append(Text(f"{ts} [POS ERR] {e}", style="red"))

        # In dry-run (non-demo): show pending bets as simulated positions
        if self.daemon.dry_run and not self.demo:
            for bet in getattr(self.daemon, '_pending_bets', []):
                if bet.get("result"):  # already settled
                    continue
                count = bet.get("count", 0)
                if count <= 0:
                    continue
                asset = bet.get("asset", "?")
                side = bet.get("side", "yes").upper()
                entry = bet.get("contract_price", bet.get("fill_price", 0))
                positions.append({
                    "asset": asset,
                    "side": side,
                    "entry": entry,
                    "count": count,
                    "status": "PENDING",
                    "strike": bet.get("strike", 0),
                    "ticker": "",
                    "current_yes": 0,
                    "current_no": 0,
                    "winning": None,
                })

        self._live_positions = positions

    # ──────────────────────────────────────────────
    # Panel builders
    # ──────────────────────────────────────────────

    def _build_header(self) -> Panel:
        now = datetime.now(timezone.utc)
        uptime = now - self._start_time
        h, m = divmod(int(uptime.total_seconds()) // 60, 60)
        mode_label = "ARB" if self.arb_mode else "UPDOWN"

        dw = self.daemon._session_wins
        dl = self.daemon._session_losses
        dt = dw + dl
        dwr = dw / dt * 100 if dt > 0 else 0
        pnl = sum(b.get("pnl_dollars", 0) for b in self.daemon._completed_bets)
        pnl_color = "green" if pnl >= 0 else "red"

        header = Text()
        header.append(f"K15 {mode_label} ", style="bold white")
        if self.demo:
            header.append("[DEMO]  ", style="yellow bold")
        elif self.dry_run:
            header.append("[DRY]  ", style="magenta bold")
        else:
            header.append("[LIVE]  ", style="red bold")
        header.append(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim")
        header.append(f"  |  Tick {self._tick_count}  |  Up {h}h{m:02d}m", style="dim")
        header.append("\n")
        header.append("Balance: ", style="dim")
        header.append(f"${self._kalshi_balance:,.2f}", style="bold white")
        header.append("  |  W:", style="dim")
        header.append(f"{dw}", style="green")
        header.append(" L:", style="dim")
        header.append(f"{dl}", style="red")
        header.append(f" WR:{dwr:.0f}%", style="dim")
        header.append("  |  P&L: ", style="dim")
        header.append(f"${pnl:+.2f}", style=pnl_color)

        if self.arb_mode:
            arb_w = getattr(self.daemon, '_session_arb_wins', 0)
            dir_h = getattr(self.daemon, '_session_directional', 0)
            sold = getattr(self.daemon, '_session_cancelled', 0)
            header.append("\nARB: ", style="dim")
            header.append(f"{arb_w} locked", style="green")
            header.append("  DIR: ", style="dim")
            header.append(f"{dir_h} held", style="cyan")
            header.append("  SOLD: ", style="dim")
            header.append(f"{sold} cut", style="yellow")

        return Panel(header, title="[bold]KALSHI K15[/]", border_style="cyan")

    def _build_predictions_table(self) -> Panel:
        preds = getattr(self.daemon, "kalshi_predictions", [])

        if self.arb_mode:
            table = Table(show_header=True, header_style="bold", expand=True,
                          show_lines=False, pad_edge=False)
            table.add_column("ASSET", style="bold white", width=5)
            table.add_column("YES", justify="right", width=6)
            table.add_column("NO", justify="right", width=6)
            table.add_column("PRICE", justify="right", width=12)
            table.add_column("TARGET", justify="right", width=12)
            table.add_column("DIST", justify="right", width=8)
            table.add_column("STATE", width=14)

            for p in preds:
                asset = p.get("asset", "?")
                yes_est = p.get("yes_est", 0)
                no_est = p.get("no_est", 0)
                yes_f = p.get("yes_filled", False)
                no_f = p.get("no_filled", False)
                state = p.get("state", "")

                yes_str = f"{yes_est}c{'*' if yes_f else ' '}"
                no_str = f"{no_est}c{'*' if no_f else ' '}"
                yes_style = "green bold" if yes_f else "white"
                no_style = "green bold" if no_f else "white"

                current = p.get("current_price", 0)
                strike = p.get("strike", 0)
                c_str = f"${current:,.2f}" if current else ""
                t_str = f"${strike:,.2f}" if strike else ""
                d_str = p.get("dist", "")

                state_style = "green" if "ARB" in state else "cyan" if "FILL" in state else "dim" if state in ("SOLD", "CANCELLED") else "white"

                table.add_row(
                    asset,
                    Text(yes_str, style=yes_style),
                    Text(no_str, style=no_style),
                    c_str, t_str, d_str,
                    Text(state, style=state_style),
                )
        else:
            table = Table(show_header=True, header_style="bold", expand=True,
                          show_lines=False, pad_edge=False)
            table.add_column("ASSET", style="bold white", width=5)
            table.add_column("SIDE", width=4)
            table.add_column("LR", justify="right", width=6)
            table.add_column("TEK", justify="right", width=6)
            table.add_column("PRICE", justify="right", width=12)
            table.add_column("TARGET", justify="right", width=12)
            table.add_column("DIST", justify="right", width=8)
            table.add_column("STATE", width=14)

            for p in preds:
                asset = p.get("asset", "?")
                side = p.get("direction", "--")
                state = p.get("state", "")

                knn_s = p.get("knn_score", 0)
                tbl_s = p.get("tbl_score", 0)

                if side in ("YES", "NO"):
                    knn_dir = "Y" if side == "YES" else "N"
                    if tbl_s >= 50:
                        tbl_dir = "Y" if side == "YES" else "N"
                    else:
                        tbl_dir = "Y" if side == "NO" else "N"
                        tbl_s = 100 - tbl_s
                elif knn_s > 0:
                    knn_dir = "Y" if knn_s >= 50 else "N"
                    tbl_dir = "Y" if tbl_s >= 50 else "N"
                else:
                    knn_dir = tbl_dir = None

                def _ft(score, direction):
                    if direction is None or score == 0:
                        return Text("--", style="dim")
                    return Text(f"{score}%{direction}", style="green" if direction == "Y" else "red")

                side_text = Text(side, style="green" if side == "YES" else "red" if side == "NO" else "dim")

                current_str = ""
                target_str = ""
                dist_str = ""
                if hasattr(self.daemon, '_kalshi_pending_signals'):
                    pending = self.daemon._kalshi_pending_signals.get(asset, {})
                    target = pending.get("strike_price", 0)
                    if target:
                        symbol = f"{asset}/USDT"
                        # XRP needs 4 decimal places to see movement
                        decimals = 4 if asset == "XRP" else 2
                        target_str = f"${target:,.{decimals}f}"
                        cb = self.daemon._get_coinbase_price(symbol) if hasattr(self.daemon, '_get_coinbase_price') else None
                        if cb:
                            current_str = f"${cb:,.{decimals}f}"
                            diff = cb - target
                            dist_str = f"{'+' if diff >= 0 else ''}{diff:.{decimals}f}"

                state_style = "cyan bold" if "BETTING" in state or "BET_PLACED" in state else "dim" if state in ("DONE", "CANCELLED") else "white"

                table.add_row(
                    asset, side_text,
                    _ft(knn_s, knn_dir),
                    _ft(tbl_s, tbl_dir),
                    current_str, target_str, dist_str,
                    Text(state, style=state_style),
                )

        now_utc = datetime.now(timezone.utc)
        min_in = now_utc.minute % 15
        ws = now_utc.minute - min_in
        we = (ws + 15) % 60
        to_next = 15 - min_in
        eval_str = "~1 min" if min_in <= 10 else "next window"
        footer = f":{ws:02d}-:{we:02d} (min {min_in}/15) | eval: {eval_str} | next: ~{to_next}m"

        title = "[bold]BOTH-SIDES ARB[/]" if self.arb_mode else "[bold]PREDICTIONS[/]"
        return Panel(table, title=title, subtitle=footer, border_style="blue")

    def _build_orders_panel(self) -> Panel:
        """Show only ACTIVE orders awaiting settlement + last 3 settled results."""
        table = Table(show_header=True, header_style="bold", expand=True,
                      show_lines=False, pad_edge=False)
        table.add_column("ASSET", style="bold", width=5)
        table.add_column("SIDE", width=4)
        table.add_column("ENTRY", justify="right", width=6)
        table.add_column("QTY", justify="right", width=4)
        table.add_column("COST", justify="right", width=8)
        table.add_column("IF WIN", justify="right", width=8)
        table.add_column("STATUS", width=12)

        pending = getattr(self.daemon, '_pending_bets', [])
        completed = getattr(self.daemon, '_completed_bets', [])
        active_count = 0

        # Show UNSETTLED pending orders
        for bet in pending:
            if bet.get("result"):  # already settled, skip
                continue
            asset = bet.get("asset", "?")
            side = bet.get("side", "?").upper()
            entry = bet.get("contract_price", bet.get("fill_price", 0))
            count = bet.get("count", 0)

            if count > 0:
                cost = count * entry / 100
                profit = count - cost
                table.add_row(
                    asset,
                    Text(side, style="green" if side == "YES" else "red"),
                    f"{entry}c", str(count), f"${cost:.2f}",
                    Text(f"${profit:+.2f}", style="green"),
                    Text("PENDING", style="cyan"),
                )
                active_count += 1

        # Show resting orders (waiting for fill)
        resting = getattr(self.daemon, '_resting_orders', [])
        for order in resting:
            asset = order.get("asset", "?")
            side = order.get("side", "?").upper()
            entry = order.get("price", order.get("fill_price", 0))
            count = order.get("count", 0)
            if count > 0:
                cost = count * entry / 100
                profit = count - cost
                table.add_row(
                    asset,
                    Text(side, style="green" if side == "YES" else "red"),
                    f"{entry}c", str(count), f"${cost:.2f}",
                    Text(f"${profit:+.2f}", style="yellow"),
                    Text("RESTING", style="yellow"),
                )
                active_count += 1

        # Show last 2 settled (keeps panel clean)
        for bet in completed[-2:]:
            asset = bet.get("asset", "?")
            entry = bet.get("contract_price", 0)
            count = bet.get("count", 1)
            pnl = bet.get("pnl_dollars", 0)
            result = bet.get("result", "?")
            side = bet.get("side", bet.get("direction", "?"))
            if isinstance(side, str):
                side = side.upper()[:3]

            r_style = "green" if result == "WIN" else "red"
            table.add_row(
                asset, Text(side, style="dim"),
                f"{entry}c" if entry else "?", str(count), "",
                Text(f"${pnl:+.2f}", style=r_style),
                Text(result, style=r_style),
            )

        if active_count == 0 and len(completed) == 0:
            table.add_row("--", Text("--", style="dim"), "--", "--", "--", "--", Text("NONE", style="dim"))

        return Panel(table, title=f"[bold]ORDERS[/] ({active_count} active)", border_style="green")

    def _build_positions_panel(self) -> Panel:
        """Show position status: RESTING → FILLED → SETTLED. Updated every 15s."""
        table = Table(show_header=True, header_style="bold", expand=True,
                      show_lines=False, pad_edge=False)
        table.add_column("ASSET", style="bold", width=5)
        table.add_column("SIDE", width=4)
        table.add_column("ENTRY", justify="right", width=6)
        table.add_column("QTY", justify="right", width=4)
        table.add_column("STATUS", width=10)
        table.add_column("WINNING", width=8)
        table.add_column("UNREALIZED", justify="right", width=10)

        if self._live_positions:
            for pos in self._live_positions:
                asset = pos["asset"]
                side = pos["side"]
                entry = pos.get("entry", 0)
                count = pos.get("count", 0)
                status = pos.get("status", "OPEN")
                yes_bid = pos.get("current_yes", 0)
                no_bid = pos.get("current_no", 0)
                winning = pos.get("winning")

                # Status styling
                status_styles = {
                    "RESTING": "yellow",
                    "PARTIAL": "yellow bold",
                    "FILLED": "green",
                    "OPEN": "green",
                    "PENDING": "cyan",
                }
                status_text = Text(status, style=status_styles.get(status, "dim"))

                # Unrealized P&L (only for filled positions with bid data)
                if count > 0 and entry > 0 and (yes_bid or no_bid):
                    current_val = yes_bid if side == "YES" else no_bid
                    unrealized_cents = count * (current_val - entry)
                    unrealized = unrealized_cents / 100
                    unreal_style = "green" if unrealized >= 0 else "red"
                    unreal_text = Text(f"${unrealized:+.2f}", style=unreal_style)
                else:
                    unreal_text = Text("--", style="dim")

                # Winning indicator
                if winning:
                    win_text = Text(f"  {winning}", style="green bold" if winning == side else "red bold")
                else:
                    win_text = Text("  --", style="dim")

                side_style = "green" if side == "YES" else "red"
                qty_str = str(count) if count > 0 else "—"

                table.add_row(
                    asset,
                    Text(side, style=side_style),
                    f"{entry}c" if entry else "?",
                    qty_str,
                    status_text,
                    win_text,
                    unreal_text,
                )
        else:
            table.add_row("--", Text("--", style="dim"), "--", "--",
                          Text("--", style="dim"), Text("--", style="dim"),
                          Text("NO POSITIONS", style="dim"))

        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return Panel(table, title="[bold]LIVE POSITIONS[/]",
                     subtitle=f"Updated {ts} (every {POSITION_REFRESH_INTERVAL}s)",
                     border_style="magenta")

    def _build_log_panel(self) -> Panel:
        log_text = Text()
        # Show newest entries first so they're always visible
        for line in reversed(self._log_lines):
            log_text.append_text(line)
            log_text.append("\n")
        if not self._log_lines:
            log_text.append("Waiting for activity...", style="dim")
        return Panel(log_text, title="[bold]ACTIVITY LOG[/] (newest first)", border_style="yellow")

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=6 if not self.arb_mode else 7),
            Layout(name="body"),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2),
        )
        layout["left"].split_column(
            Layout(name="predictions", ratio=3),
            Layout(name="orders", ratio=2),
            Layout(name="positions", ratio=2),
        )
        layout["header"].update(self._build_header())
        layout["left"]["predictions"].update(self._build_predictions_table())
        layout["left"]["orders"].update(self._build_orders_panel())
        layout["left"]["positions"].update(self._build_positions_panel())
        layout["right"].update(self._build_log_panel())
        return layout

    def run(self):
        self.daemon._running = True

        def _shutdown(*_):
            self.daemon._running = False
        sig.signal(sig.SIGINT, _shutdown)
        sig.signal(sig.SIGTERM, _shutdown)

        # Startup
        self._fetch_balance()
        self.daemon.startup()

        # Initial eval
        eval_fn = self.daemon._arb_eval if self.arb_mode else self.daemon._kalshi_eval
        try:
            eval_fn()
            self.daemon._last_kalshi_eval = time.time()
        except Exception as e:
            self._log_lines.append(Text(f"EVAL ERR: {e}", style="red bold"))

        total_ticks = self.max_cycles * 15

        with Live(self._build_layout(), refresh_per_second=1, console=self._console, screen=True) as live:
            tick = 0
            last_eval_tick = 0
            seconds_elapsed = 0

            while self.daemon._running and tick < total_ticks:
                # Sleep 1 second at a time for responsive Ctrl+C
                time.sleep(1)
                if not self.daemon._running:
                    break
                seconds_elapsed += 1

                # Every 5 seconds: refresh positions, balance, settlements
                if seconds_elapsed % POSITION_REFRESH_INTERVAL == 0:
                    try:
                        self._refresh_positions()
                        self._fetch_balance()
                        # Check price watches (price-blocked signals waiting for entry)
                        if hasattr(self.daemon, 'check_price_watches'):
                            self.daemon.check_price_watches()
                        # Check settlements
                        if hasattr(self.daemon, '_check_dryrun_settlements'):
                            self.daemon._check_dryrun_settlements()
                    except Exception:
                        pass
                    live.update(self._build_layout())

                # Every 60 seconds: full eval tick
                if seconds_elapsed % 60 == 0:
                    tick += 1
                    self._tick_count = tick

                    # Refresh balance every 5 ticks
                    if tick % 5 == 0:
                        self._fetch_balance()

                    # Run eval — fire immediately at minute 1 (entry window), then every 50s
                    now_ts = time.time()
                    now_utc = datetime.now(timezone.utc)
                    min_in = now_utc.minute % 15
                    sec_in = now_utc.second
                    time_since_eval = now_ts - self.daemon._last_kalshi_eval
                    if self.arb_mode:
                        should_eval = time_since_eval >= 50
                    else:
                        # At minute 1 second 5+: fire immediately (1m candle just closed)
                        # Otherwise: every 50 seconds
                        entry_trigger = (min_in == 1 and sec_in >= 5 and time_since_eval >= 10)
                        normal_trigger = (min_in >= 1 and time_since_eval >= 50)
                        should_eval = entry_trigger or normal_trigger

                    if should_eval:
                        try:
                            eval_fn()
                        except Exception as e:
                            self._log_lines.append(Text(f"EVAL ERR: {e}", style="red bold"))
                        self.daemon._last_kalshi_eval = now_ts

                    live.update(self._build_layout())

                # Update layout every second for instant log visibility
                else:
                    live.update(self._build_layout())

        # Final trading summary
        self._console.print("\n")
        completed = self.daemon._completed_bets
        dw = self.daemon._session_wins
        dl = self.daemon._session_losses
        dt = dw + dl
        dwr = dw / dt * 100 if dt > 0 else 0
        total_pnl = sum(b.get("pnl_dollars", 0) for b in completed)
        total_risked = sum(abs(b.get("pnl_cents", 0)) for b in completed if b.get("result") == "LOSS") / 100
        total_won = sum(b.get("pnl_dollars", 0) for b in completed if b.get("result") == "WIN")
        total_lost = sum(abs(b.get("pnl_dollars", 0)) for b in completed if b.get("result") == "LOSS")

        now = datetime.now(timezone.utc)
        uptime = now - self._start_time

        summary = Table(title="SESSION SUMMARY", show_header=False, expand=True,
                        border_style="cyan", pad_edge=True)
        summary.add_column("Key", style="dim", width=20)
        summary.add_column("Value", width=30)

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        summary.add_row("Mode", mode)
        summary.add_row("Duration", str(uptime).split('.')[0])
        summary.add_row("Bets Placed", str(len(completed)))
        summary.add_row("Wins", Text(str(dw), style="green"))
        summary.add_row("Losses", Text(str(dl), style="red"))
        summary.add_row("Win Rate", f"{dwr:.1f}%")
        summary.add_row("Gross Won", Text(f"${total_won:+.2f}", style="green"))
        summary.add_row("Gross Lost", Text(f"${total_lost:.2f}", style="red"))
        summary.add_row("Net P&L", Text(f"${total_pnl:+.2f}", style="green" if total_pnl >= 0 else "red"))
        pending = len(self.daemon._pending_bets)
        if pending > 0:
            summary.add_row("Unsettled", Text(f"{pending} pending", style="yellow"))

        self._console.print(summary)

        if completed:
            trades = Table(title="TRADE LOG", expand=True, border_style="blue")
            trades.add_column("#", width=3)
            trades.add_column("ASSET", width=5)
            trades.add_column("SIDE", width=4)
            trades.add_column("ENTRY", justify="right", width=6)
            trades.add_column("QTY", justify="right", width=4)
            trades.add_column("RESULT", width=6)
            trades.add_column("P&L", justify="right", width=8)

            for i, b in enumerate(completed, 1):
                asset = b.get("asset", "?")
                side = b.get("side", b.get("direction", "?"))
                if isinstance(side, str):
                    side = side.upper()[:3]
                entry = b.get("contract_price", 0)
                count = b.get("count", 1)
                result = b.get("result", "?")
                pnl = b.get("pnl_dollars", 0)
                r_style = "green" if result == "WIN" else "red"

                trades.add_row(
                    str(i), asset,
                    Text(side, style="green" if side == "YES" else "red" if side == "NO" else "dim"),
                    f"{entry}c" if entry else "?",
                    str(count),
                    Text(result, style=r_style),
                    Text(f"${pnl:+.2f}", style=r_style),
                )

            self._console.print(trades)


def main():
    parser = argparse.ArgumentParser(description="K15 Rich Dashboard")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="Use Kalshi demo exchange (real orders, play money)")
    parser.add_argument("--arb", action="store_true")
    parser.add_argument("--cycles", type=int, default=96)
    parser.add_argument("--maxbets", type=int, default=0,
                        help="Max concurrent bets (0 = default). Highest confidence wins.")
    parser.add_argument("--maxsize", type=float, default=0,
                        help="Position size as %% of balance (0 = default 5%%). E.g. --maxsize=2.5")
    args = parser.parse_args()

    dry_run = not args.live and not args.demo
    RichDashboard(
        dry_run=dry_run, arb_mode=args.arb, max_cycles=args.cycles,
        demo=args.demo, max_bets=args.maxbets, max_size_pct=args.maxsize,
    ).run()


if __name__ == "__main__":
    main()
