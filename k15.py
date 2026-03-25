#!/usr/bin/env python3
"""K15 — Interactive Kalshi 15-minute prediction dashboard.

Usage:
    ./venv/bin/python k15.py

Commands (type at the k15> prompt):
    /           Show all commands
    /dry-run    Start V3 predictor in dry-run mode
    /live       Start V3 predictor in live mode (requires confirmation)
    /stop       Stop the running bot
    /status     Show current state
    /balance    Query Kalshi balance
    /closeall   Emergency close all positions (requires confirmation)
    /quit       Exit the program
"""
import sys
import os
import time
import select
import threading
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


# ─── State ───────────────────────────────────────────────────────────
class K15State:
    """Shared state between UI thread and daemon thread."""
    def __init__(self):
        self.daemon = None
        self.daemon_thread = None
        self.running = False
        self.mode = "STOPPED"  # STOPPED, DRY-RUN, LIVE
        self.predictor_version = "v3"
        self.activity_log: list[str] = []  # most recent first
        self.session_wins = 0
        self.session_losses = 0
        self.session_pnl = 0.0
        self.kalshi_balance = 0.0
        self.last_balance_check = 0.0

    def log(self, message: str):
        """Add an entry to the activity log."""
        ts = datetime.now(timezone.utc).strftime("%H:%M")
        styled = f"[dim]{ts}[/dim] {message}"
        self.activity_log.insert(0, styled)
        if len(self.activity_log) > 20:
            self.activity_log = self.activity_log[:20]


# ─── Layout Builder ─────────────────────────────────────────────────
def build_layout(state: K15State, input_buffer: str = "") -> Layout:
    """Build the rich Layout from current state."""
    # Top: two columns. Bottom: input prompt.
    layout = Layout()
    layout.split_column(
        Layout(name="top", ratio=1),
        Layout(name="bottom", size=3),
    )

    # Input prompt at the bottom
    from rich.text import Text as RichText
    prompt = RichText()
    prompt.append("\n k15> ", style="bold cyan")
    prompt.append(input_buffer)
    prompt.append("█", style="bold cyan")  # cursor
    layout["bottom"].update(Panel(prompt, box=box.SIMPLE, style="dim"))

    # Top splits into left and right
    layout["top"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )

    # ── Left panel: Status + Predictions ──
    left_content = build_status_panel(state)
    layout["left"].update(Panel(
        left_content,
        title=f"[bold cyan]K15 UPDOWN [{state.mode}] V3[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    # ── Right panel: Activity Log ──
    log_text = Text()
    if state.activity_log:
        for entry in state.activity_log:
            log_text.append_text(Text.from_markup(entry + "\n"))
    else:
        log_text.append("  No activity yet\n", style="dim")

    layout["right"].update(Panel(
        log_text,
        title="[bold cyan]ACTIVITY LOG[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    ))

    return layout


def build_status_panel(state: K15State) -> Text:
    """Build the left panel content."""
    t = Text()

    # Balance
    bal = state.kalshi_balance
    t.append(f"  KALSHI BALANCE: ", style="white")
    t.append(f"${bal:.2f}\n", style="bold green" if bal > 0 else "bold red")

    # Session stats
    total = state.session_wins + state.session_losses
    wr = state.session_wins / total * 100 if total > 0 else 0
    t.append(f"  OPEN BETS: ", style="white")

    # Count active bets from daemon
    active_bets = 0
    max_bets = 2
    if state.daemon:
        from config.production import MAX_CONCURRENT_KALSHI_BETS
        max_bets = MAX_CONCURRENT_KALSHI_BETS
        active_bets = len(getattr(state.daemon, '_active_kalshi_bets', {}))
    t.append(f"{active_bets}/{max_bets}", style="bold")
    t.append(f"  |  W:{state.session_wins} L:{state.session_losses}  |  WR: {wr:.1f}%\n", style="white")

    # P&L
    pnl_style = "bold green" if state.session_pnl >= 0 else "bold red"
    pnl_sign = "+" if state.session_pnl >= 0 else ""
    t.append(f"  SESSION P&L: ", style="white")
    t.append(f"{pnl_sign}${state.session_pnl:.2f}\n\n", style=pnl_style)

    # Predictions table
    predictions = []
    if state.daemon:
        predictions = getattr(state.daemon, 'kalshi_predictions', [])

    if predictions:
        t.append("  ASSET  SIDE  PROB  STATE            STRIKE\n", style="bold cyan")
        t.append("  " + "─" * 46 + "\n", style="dim")

        for pred in predictions:
            asset = pred.get("asset", "?")
            conf = pred.get("confidence", 0)
            st = pred.get("state", "")
            direction = pred.get("direction", "--")
            reason = pred.get("reason", "")

            # Extract strike from reason or pending signals
            strike_str = ""
            if state.daemon and hasattr(state.daemon, '_kalshi_pending_signals'):
                pending = state.daemon._kalshi_pending_signals.get(asset, {})
                strike = pending.get("strike_price", 0)
                if strike:
                    strike_str = f"${strike:,.2f}"

            # Color coding
            if conf >= 70:
                prob_style = "bold green"
            elif conf >= 65:
                prob_style = "green"
            elif conf <= 30:
                prob_style = "bold red"
            elif conf <= 35:
                prob_style = "red"
            else:
                prob_style = "white"

            if st == "BET_PLACED":
                state_style = "bold magenta"
            elif st == "SKIP" or direction == "--":
                state_style = "yellow"
            elif st == "SETUP":
                state_style = "yellow"
            else:
                state_style = "white"

            side_str = direction if direction != "--" else "--"
            side_style = "green" if side_str == "YES" else ("red" if side_str == "NO" else "dim")

            t.append(f"  {asset:<5} ", style="white")
            t.append(f"{side_str:<5} ", style=side_style)
            t.append(f"{conf:<5}%", style=prob_style)
            t.append(f" {st:<16} ", style=state_style)
            t.append(f"{strike_str}\n", style="dim")
    else:
        t.append("  No predictions yet. Type /dry-run to start.\n", style="dim")

    # Footer
    t.append("\n")
    now = datetime.now(timezone.utc)
    min_in_window = now.minute % 15
    window_start = now.minute - min_in_window
    window_end = window_start + 15
    next_eval_mins = 5 - (min_in_window % 5)
    if next_eval_mins == 0:
        next_eval_mins = 5

    t.append(f"  Next eval: ~{next_eval_mins} min  |  Window: :{window_start:02d}-:{window_end:02d}\n", style="dim")

    return t


# ─── Daemon Thread ───────────────────────────────────────────────────
def daemon_worker(state: K15State):
    """Run the LiveDaemon eval loop in a background thread."""
    daemon = state.daemon
    if daemon is None:
        return

    try:
        daemon.startup()
        state.log("[bold green]Bot started[/bold green]")

        # Initial eval
        try:
            daemon._kalshi_eval()
            daemon._last_kalshi_eval = time.time()
        except Exception as e:
            state.log(f"[red]Eval error: {e}[/red]")

        # Main loop
        while state.running and daemon._running:
            time.sleep(60)
            if not state.running:
                break

            # Update prices + equity
            try:
                daemon._update_prices()
                daemon._update_equity()
            except Exception:
                pass

            # Wall-clock aligned eval
            now_ts = time.time()
            current_minute = datetime.now(timezone.utc).minute
            should_eval = (current_minute % 5 == 1 and now_ts - daemon._last_kalshi_eval >= 240) \
                       or (current_minute % 15 == 12 and now_ts - daemon._last_kalshi_eval >= 50) \
                       or (current_minute % 15 == 1 and now_ts - daemon._last_kalshi_eval >= 50)
            if should_eval:
                try:
                    daemon._kalshi_eval()

                    # Check for new bets and log them
                    for pred in getattr(daemon, 'kalshi_predictions', []):
                        st = pred.get("state", "")
                        asset = pred.get("asset", "?")
                        conf = pred.get("confidence", 0)
                        direction = pred.get("direction", "--")

                        if st == "BET_PLACED" and conf > 0:
                            side_color = "green" if direction == "YES" else "red"
                            state.log(f"[{side_color}]{asset} {direction} {conf}% → BET[/{side_color}]")

                except Exception as e:
                    state.log(f"[red]Eval error: {e}[/red]")
                daemon._last_kalshi_eval = now_ts

            # Refresh balance periodically (every 60s)
            if now_ts - state.last_balance_check >= 60:
                try:
                    daemon._init_kalshi_client()
                    if daemon.kalshi_client:
                        bal = daemon.kalshi_client.get_balance()
                        state.kalshi_balance = bal.get("balance", 0) / 100  # cents to dollars
                except Exception:
                    pass
                state.last_balance_check = now_ts

    except Exception as e:
        state.log(f"[bold red]Daemon crashed: {e}[/bold red]")
    finally:
        state.running = False
        state.mode = "STOPPED"
        state.log("[yellow]Bot stopped[/yellow]")


# ─── Command Handler ─────────────────────────────────────────────────
COMMANDS = {
    "/":         "Show all commands",
    "/dry-run":  "Start V3 predictor in dry-run mode",
    "/live":     "Start V3 predictor in LIVE mode",
    "/stop":     "Stop the running bot",
    "/status":   "Show current state",
    "/balance":  "Query Kalshi balance",
    "/closeall": "Emergency close all positions",
    "/quit":     "Exit the program",
}


def handle_command(cmd: str, state: K15State, console: Console) -> bool:
    """Handle a command. Returns True if should quit."""
    cmd = cmd.strip().lower()

    if cmd == "/":
        console.print("\n[bold cyan]Available commands:[/bold cyan]")
        for c, desc in COMMANDS.items():
            console.print(f"  [bold]{c:<12}[/bold] {desc}")
        console.print()
        return False

    elif cmd == "/dry-run":
        if state.running:
            console.print("[yellow]Bot is already running. /stop first.[/yellow]")
            return False
        console.print("[cyan]Starting V3 dry-run...[/cyan]")
        from cli.live_daemon import LiveDaemon
        state.daemon = LiveDaemon(dry_run=True, kalshi_only=True, predictor_version="v3")
        state.running = True
        state.mode = "DRY-RUN"
        state.daemon._running = True
        state.daemon_thread = threading.Thread(target=daemon_worker, args=(state,), daemon=True)
        state.daemon_thread.start()
        state.log("[bold green]Started DRY-RUN V3[/bold green]")
        return False

    elif cmd == "/live":
        if state.running:
            console.print("[yellow]Bot is already running. /stop first.[/yellow]")
            return False
        confirm = console.input("[bold red]Start LIVE trading? Type YES to confirm: [/bold red]")
        if confirm.strip() != "YES":
            console.print("[yellow]Cancelled.[/yellow]")
            return False
        console.print("[bold red]Starting V3 LIVE...[/bold red]")
        from cli.live_daemon import LiveDaemon
        state.daemon = LiveDaemon(dry_run=False, kalshi_only=True, predictor_version="v3")
        state.running = True
        state.mode = "LIVE"
        state.daemon._running = True
        state.daemon_thread = threading.Thread(target=daemon_worker, args=(state,), daemon=True)
        state.daemon_thread.start()
        state.log("[bold red]Started LIVE V3[/bold red]")
        return False

    elif cmd == "/stop":
        if not state.running:
            console.print("[yellow]Bot is not running.[/yellow]")
            return False
        console.print("[yellow]Stopping bot...[/yellow]")
        state.running = False
        if state.daemon:
            state.daemon._running = False
        return False

    elif cmd == "/status":
        console.print(f"\n[cyan]Mode:[/cyan] {state.mode}")
        console.print(f"[cyan]Predictor:[/cyan] {state.predictor_version}")
        console.print(f"[cyan]Running:[/cyan] {state.running}")
        console.print(f"[cyan]Balance:[/cyan] ${state.kalshi_balance:.2f}")
        console.print(f"[cyan]W/L/WR:[/cyan] {state.session_wins}/{state.session_losses}/{state.session_wins / max(1, state.session_wins + state.session_losses) * 100:.0f}%")
        console.print()
        return False

    elif cmd == "/balance":
        console.print("[cyan]Querying Kalshi balance...[/cyan]")
        try:
            if state.daemon and state.daemon.kalshi_client:
                bal = state.daemon.kalshi_client.get_balance()
                state.kalshi_balance = bal.get("balance", 0) / 100
            else:
                from exchange.kalshi import KalshiClient
                from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
                client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE), demo=False)
                bal = client.get_balance()
                state.kalshi_balance = bal.get("balance", 0) / 100
            console.print(f"[bold green]Kalshi Balance: ${state.kalshi_balance:.2f}[/bold green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        return False

    elif cmd == "/closeall":
        confirm = console.input("[bold red]EMERGENCY CLOSE ALL positions? Type YES to confirm: [/bold red]")
        if confirm.strip() != "YES":
            console.print("[yellow]Cancelled.[/yellow]")
            return False
        console.print("[bold red]Closing all positions...[/bold red]")
        try:
            if state.daemon:
                state.daemon._init_kalshi_client()
            if state.daemon and state.daemon.kalshi_client:
                # Cancel resting orders
                orders = state.daemon.kalshi_client.get_orders(status="resting")
                for order in orders:
                    oid = order.get("order_id")
                    if oid:
                        state.daemon.kalshi_client.cancel_order(oid)
                        state.log(f"[red]Cancelled order {oid}[/red]")
                state.log("[bold red]All positions closed[/bold red]")
                console.print("[bold red]Done. All resting orders cancelled.[/bold red]")
            else:
                console.print("[yellow]No Kalshi client available.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        return False

    elif cmd == "/quit":
        if state.running:
            console.print("[yellow]Stopping bot...[/yellow]")
            state.running = False
            if state.daemon:
                state.daemon._running = False
            time.sleep(2)
        return True

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type / for help.[/yellow]")
        return False


# ─── Main ────────────────────────────────────────────────────────────
def main():
    console = Console()
    state = K15State()

    # Banner
    console.print(Panel(
        "[bold cyan]K15 UPDOWN[/bold cyan] — Kalshi 15-min Prediction Dashboard\n"
        "Type [bold]/[/bold] for commands, [bold]/dry-run[/bold] to start",
        border_style="cyan",
        box=box.DOUBLE,
    ))

    # Try to get initial balance
    try:
        from exchange.kalshi import KalshiClient
        from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
        client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE), demo=False)
        bal = client.get_balance()
        state.kalshi_balance = bal.get("balance", 0) / 100
        console.print(f"[green]Kalshi Balance: ${state.kalshi_balance:.2f}[/green]")
    except Exception:
        console.print("[yellow]Could not connect to Kalshi[/yellow]")

    # Full-screen Live display with input buffer rendered inside the layout.
    # Raw terminal mode captures keystrokes without echo.
    # Live.update() redraws in-place — no scrolling, no flicker.

    import tty
    import termios

    input_buffer = ""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        with Live(build_layout(state, input_buffer), console=console,
                  refresh_per_second=0, screen=True) as live:

            last_refresh = time.time()

            while True:
                # Check for keystroke with short timeout
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)

                if ready:
                    ch = sys.stdin.read(1)

                    if ch == '\n' or ch == '\r':
                        cmd = input_buffer.strip()
                        input_buffer = ""
                        if cmd:
                            # Temporarily exit Live to show command output
                            live.stop()
                            console.clear()
                            if handle_command(cmd, state, console):
                                break
                            time.sleep(0.5)
                            # Resume Live
                            live.start()
                    elif ch == '\x7f' or ch == '\x08':
                        if input_buffer:
                            input_buffer = input_buffer[:-1]
                    elif ch == '\x03' or ch == '\x04':
                        break
                    elif ch >= ' ':
                        input_buffer += ch

                    # Update display immediately on keystroke
                    live.update(build_layout(state, input_buffer))
                    last_refresh = time.time()

                # Auto-refresh every 2 seconds
                now = time.time()
                if now - last_refresh >= 2.0:
                    live.update(build_layout(state, input_buffer))
                    last_refresh = now

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        if state.running:
            state.running = False
            if state.daemon:
                state.daemon._running = False
            time.sleep(1)

    console.print("\n[cyan]Goodbye.[/cyan]")


if __name__ == "__main__":
    main()
