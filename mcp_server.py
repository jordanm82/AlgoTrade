#!/usr/bin/env python3
"""AlgoTrade MCP Server — gives Claude direct tool access to the trading system."""
import sys
sys.path.insert(0, '.')

import asyncio
import json
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ---------------------------------------------------------------------------
# Global daemon process handles
# ---------------------------------------------------------------------------
_daemon_process = None
_mm_process = None

server = Server("algotrade")


# ===================================================================
#  Tool definitions
# ===================================================================

@server.list_tools()
async def list_tools():
    return [
        # 1. Status
        Tool(
            name="algotrade_status",
            description="Show dashboard log contents and daemon process status.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 2. Start
        Tool(
            name="algotrade_start",
            description="Start the trading dashboard as a background subprocess.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["live", "dry-run"],
                        "description": "Trading mode: 'live' or 'dry-run'.",
                        "default": "dry-run",
                    },
                    "cycles": {
                        "type": "integer",
                        "description": "Number of 15-min signal cycles to run.",
                        "default": 50,
                    },
                    "kalshi_only": {
                        "type": "boolean",
                        "description": "Run only Kalshi 15m predictions, skip Coinbase spot trading.",
                        "default": False,
                    },
                    "predictor": {
                        "type": "string",
                        "enum": ["v1", "v2"],
                        "description": "Kalshi predictor: v1 (mean-reversion) or v2 (continuation).",
                        "default": "v1",
                    },
                },
            },
        ),
        # 3. Stop
        Tool(
            name="algotrade_stop",
            description="Stop the running trading daemon.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 4. Balances
        Tool(
            name="algotrade_balances",
            description="Get Coinbase + Kalshi account balances with token valuations.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 5. Positions
        Tool(
            name="algotrade_positions",
            description="Get all open positions from the position tracker with live P&L.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 6. Buy
        Tool(
            name="algotrade_buy",
            description="Execute a spot market buy on Coinbase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Coinbase product ID, e.g. 'SOL-USD'.",
                    },
                    "usd_amount": {
                        "type": "number",
                        "description": "USD amount to spend.",
                    },
                },
                "required": ["symbol", "usd_amount"],
            },
        ),
        # 7. Sell
        Tool(
            name="algotrade_sell",
            description="Execute a spot market sell on Coinbase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Coinbase product ID, e.g. 'SOL-USD'.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Token amount to sell (ignored if 'all' is true).",
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Sell entire balance of this token.",
                        "default": False,
                    },
                },
                "required": ["symbol"],
            },
        ),
        # 8. Close all
        Tool(
            name="algotrade_close_all",
            description="Sell all non-USD tokens on Coinbase and cancel all Kalshi orders.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 9. Kalshi bet
        Tool(
            name="algotrade_kalshi_bet",
            description="Place a Kalshi 15-min crypto prediction bet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "asset": {
                        "type": "string",
                        "enum": ["BTC", "ETH", "SOL", "XRP"],
                        "description": "Which crypto asset.",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["yes", "no"],
                        "description": "Bet 'yes' (price up) or 'no' (price down).",
                    },
                    "amount_usd": {
                        "type": "number",
                        "description": "USD amount to bet.",
                    },
                },
                "required": ["asset", "side", "amount_usd"],
            },
        ),
        # 10. Kalshi markets
        Tool(
            name="algotrade_kalshi_markets",
            description="List available Kalshi 15-min crypto contracts with current prices.",
            inputSchema={
                "type": "object",
                "properties": {
                    "asset": {
                        "type": "string",
                        "description": "Optional asset filter: 'BTC', 'ETH', 'SOL', 'XRP'.",
                    },
                },
            },
        ),
        # 11. Ticker
        Tool(
            name="algotrade_ticker",
            description="Get current price, 24h change, bid, ask, and volume for a symbol.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair, e.g. 'BTC/USDT'.",
                    },
                },
                "required": ["symbol"],
            },
        ),
        # 12. Indicators
        Tool(
            name="algotrade_indicators",
            description="Get technical indicators (RSI, BB, MACD, ATR, EMA, volume) for a pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair, e.g. 'BTC/USDT'.",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe.",
                        "default": "15m",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of candles to fetch.",
                        "default": 100,
                    },
                },
                "required": ["symbol"],
            },
        ),
        # 13. Order book
        Tool(
            name="algotrade_orderbook",
            description="Get order book imbalance, bid/ask volumes, spread, and wall detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair, e.g. 'BTC/USDT'.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Order book depth.",
                        "default": 20,
                    },
                },
                "required": ["symbol"],
            },
        ),
        # 14. Trade flow
        Tool(
            name="algotrade_tradeflow",
            description="Get net trade flow, buy ratio, and large trade bias for a pair.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair, e.g. 'BTC/USDT'.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent trades to analyze.",
                        "default": 200,
                    },
                },
                "required": ["symbol"],
            },
        ),
        # 15. Funding rates
        Tool(
            name="algotrade_funding_rates",
            description="Get current Binance USDM perpetual funding rates.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 16. Signals
        Tool(
            name="algotrade_signals",
            description="Run all strategies on all pairs RIGHT NOW and return signals with confidence scores + leading indicator data.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 17. Backtest
        Tool(
            name="algotrade_backtest",
            description="Run a quick backtest on a pair/strategy and return results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Trading pair, e.g. 'ATOM/USDT'.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["bb_grid", "rsi_mr"],
                        "description": "Strategy to backtest.",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Candle timeframe.",
                        "default": "15m",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days of historical data.",
                        "default": 30,
                    },
                    "leverage": {
                        "type": "integer",
                        "description": "Leverage multiplier.",
                        "default": 1,
                    },
                },
                "required": ["symbol", "strategy"],
            },
        ),
        # 18. Performance
        Tool(
            name="algotrade_performance",
            description="Get session performance stats from trades.csv.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 19. Trade history
        Tool(
            name="algotrade_trade_history",
            description="Get recent trades from trades.csv.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent trades to return.",
                        "default": 20,
                    },
                },
            },
        ),
        # 20. Config get
        Tool(
            name="algotrade_config_get",
            description="Get all current configuration (pair configs, risk settings, production settings).",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 21. Config set
        Tool(
            name="algotrade_config_set",
            description="Update a pair config or settings value at runtime.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pair": {
                        "type": "string",
                        "description": "Trading pair to update, e.g. 'ATOM/USDT'.",
                    },
                    "key": {
                        "type": "string",
                        "description": "Config key to update, e.g. 'bb_rsi_buy', 'leverage'.",
                    },
                    "value": {
                        "description": "New value to set.",
                    },
                },
                "required": ["pair", "key", "value"],
            },
        ),
        # 22. Force kill
        Tool(
            name="algotrade_force_kill",
            description="Force kill the daemon (SIGKILL + pkill) when normal stop doesn't work.",
            inputSchema={"type": "object", "properties": {}},
        ),
        # 23. Logs
        Tool(
            name="algotrade_logs",
            description="View recent daemon stdout/stderr log lines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of lines to return.", "default": 50},
                },
            },
        ),
        # 24. Errors
        Tool(
            name="algotrade_errors",
            description="Scan daemon logs for errors, warnings, failures, and blocked trades.",
            inputSchema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Max lines per category.", "default": 30},
                },
            },
        ),
        # MM Tools
        Tool(
            name="algotrade_mm_start",
            description="Start the market maker bot as a background subprocess.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["live", "dry-run"],
                        "description": "Trading mode.",
                        "default": "dry-run",
                    },
                },
            },
        ),
        Tool(
            name="algotrade_mm_stop",
            description="Stop the market maker bot gracefully.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="algotrade_mm_status",
            description="Show market maker dashboard state.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ===================================================================
#  Tool handler implementations
# ===================================================================

def handle_status() -> dict:
    """Return dashboard log + daemon process status."""
    log = Path("data/store/dashboard.log")
    content = log.read_text() if log.exists() else "No dashboard log found."
    pid_file = Path("/tmp/dashboard_pid.txt")
    running = False
    pid = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (OSError, ValueError):
            running = False
    return {"dashboard_log": content, "daemon_running": running, "pid": pid}


def handle_start(mode: str = "dry-run", cycles: int = 50, kalshi_only: bool = False, predictor: str = "v1") -> dict:
    """Start the dashboard as a background subprocess."""
    global _daemon_process
    flag = "--live" if mode == "live" else "--dry-run"
    cmd = [sys.executable, "dashboard.py", flag, "--cycles", str(cycles)]
    if kalshi_only:
        cmd.append("--kalshi-only")
    if predictor == "v2":
        cmd.append("--predictor")
        cmd.append("v2")
    _daemon_process = subprocess.Popen(
        cmd,
        stdout=open("/tmp/dashboard_stdout.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    Path("/tmp/dashboard_pid.txt").write_text(str(_daemon_process.pid))
    return {"pid": _daemon_process.pid, "mode": mode, "cycles": cycles, "kalshi_only": kalshi_only, "predictor": predictor}


def handle_stop() -> dict:
    """Stop the running daemon process."""
    global _daemon_process
    pid_file = Path("/tmp/dashboard_pid.txt")
    killed = False

    # Try global process handle first
    if _daemon_process is not None:
        try:
            _daemon_process.terminate()
            _daemon_process.wait(timeout=5)
            killed = True
        except Exception:
            try:
                _daemon_process.kill()
                killed = True
            except Exception:
                pass
        _daemon_process = None

    # Also try PID file
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 15)  # SIGTERM
            killed = True
        except (OSError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)

    return {"stopped": killed}


def handle_force_kill() -> dict:
    """Force kill the daemon process (SIGKILL) when stop doesn't work."""
    global _daemon_process
    pid_file = Path("/tmp/dashboard_pid.txt")
    killed = False

    if _daemon_process is not None:
        try:
            _daemon_process.kill()
            killed = True
        except Exception:
            pass
        _daemon_process = None

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 9)  # SIGKILL
            killed = True
        except (OSError, ValueError):
            pass
        pid_file.unlink(missing_ok=True)

    # Also kill any stray dashboard.py processes
    try:
        result = subprocess.run(["pkill", "-9", "-f", "dashboard.py"], capture_output=True)
        if result.returncode == 0:
            killed = True
    except Exception:
        pass

    return {"force_killed": killed}


def handle_logs(lines: int = 50) -> dict:
    """Check daemon stdout/stderr logs for recent output."""
    log_file = Path("/tmp/dashboard_stdout.log")
    if not log_file.exists():
        return {"error": "No daemon log file found", "path": str(log_file)}
    content = log_file.read_text()
    log_lines = content.strip().split("\n")
    return {
        "total_lines": len(log_lines),
        "last_lines": log_lines[-lines:],
        "path": str(log_file),
    }


def handle_errors(lines: int = 30) -> dict:
    """Scan daemon logs for errors, warnings, and failures."""
    results = {"errors": [], "warnings": [], "failures": [], "blocks": []}

    # Check daemon stdout log
    log_file = Path("/tmp/dashboard_stdout.log")
    if log_file.exists():
        for line in log_file.read_text().strip().split("\n"):
            ll = line.lower()
            if "traceback" in ll or "error" in ll or "exception" in ll:
                results["errors"].append(line.strip())
            elif "warn" in ll:
                results["warnings"].append(line.strip())
            elif "fail" in ll:
                results["failures"].append(line.strip())
            elif "block" in ll:
                results["blocks"].append(line.strip())

    # Check trades for any issues
    trades_file = Path("data/store/trades.csv")
    if trades_file.exists():
        import pandas as pd
        try:
            df = pd.read_csv(trades_file)
            if len(df) > 0:
                results["total_trades"] = len(df)
                results["last_trade"] = df.iloc[-1].to_dict()
        except Exception:
            pass

    # Trim to last N
    for key in ("errors", "warnings", "failures", "blocks"):
        results[key] = results[key][-lines:]

    results["summary"] = (
        f"{len(results['errors'])} errors, {len(results['warnings'])} warnings, "
        f"{len(results['failures'])} failures, {len(results['blocks'])} blocks"
    )
    return results


def handle_balances() -> dict:
    """Get Coinbase + Kalshi balances with token valuations."""
    from exchange.coinbase import CoinbaseExecutor
    from exchange.kalshi import KalshiClient
    from config.settings import CDP_KEY_FILE, KALSHI_KEY_FILE, KALSHI_API_KEY_ID

    cb = {}
    try:
        executor = CoinbaseExecutor(str(CDP_KEY_FILE))
        cb = executor.get_balances()
    except Exception as e:
        cb = {"error": str(e)}

    kalshi = {}
    try:
        client = KalshiClient(
            api_key_id=KALSHI_API_KEY_ID,
            private_key_path=str(KALSHI_KEY_FILE),
            demo=False,
        )
        bal = client.get_balance()
        kalshi = {
            "balance": bal.get("balance", 0) / 100,
            "portfolio": bal.get("portfolio_value", 0) / 100,
        }
    except Exception as e:
        kalshi = {"error": str(e)}

    # Convert token amounts to USD for Coinbase
    coinbase_usd = cb.get("USD", 0) if isinstance(cb, dict) and "error" not in cb else 0
    token_value = 0
    tokens = {}
    try:
        import ccxt
        exchange = ccxt.binanceus({"enableRateLimit": True})
        for currency, amount in cb.items():
            if currency in ("USD", "USDC", "USDT", "error"):
                continue
            if amount > 0:
                try:
                    ticker = exchange.fetch_ticker(f"{currency}/USDT")
                    usd_val = amount * ticker["last"]
                    tokens[currency] = {
                        "amount": amount,
                        "price": ticker["last"],
                        "usd_value": round(usd_val, 2),
                    }
                    token_value += usd_val
                except Exception:
                    tokens[currency] = {"amount": amount, "usd_value": 0}
    except Exception:
        pass

    coinbase_total = coinbase_usd + token_value
    kalshi_total = kalshi.get("balance", 0) + kalshi.get("portfolio", 0)

    return {
        "coinbase": {
            "usd": round(coinbase_usd, 2),
            "tokens": tokens,
            "total": round(coinbase_total, 2),
        },
        "kalshi": kalshi,
        "grand_total": round(coinbase_total + kalshi_total, 2),
    }


def handle_positions() -> dict:
    """Get open positions from position tracker."""
    from exchange.positions import PositionTracker

    tracker = PositionTracker()
    tracker.load_state()
    positions = tracker.open_positions()
    closed = tracker.closed_trades()
    return {
        "open_positions": positions,
        "num_open": len(positions),
        "total_exposure": tracker.total_exposure(),
        "recent_closed": closed[-10:] if closed else [],
    }


def handle_buy(symbol: str, usd_amount: float) -> dict:
    """Execute a spot market buy on Coinbase."""
    from exchange.coinbase import CoinbaseExecutor
    from config.settings import CDP_KEY_FILE

    executor = CoinbaseExecutor(str(CDP_KEY_FILE))
    result = executor.market_buy(symbol, usd_amount)
    return {"action": "buy", "symbol": symbol, "usd_amount": usd_amount, "result": result}


def handle_sell(symbol: str, amount: float = 0, sell_all: bool = False) -> dict:
    """Execute a spot market sell on Coinbase."""
    from exchange.coinbase import CoinbaseExecutor
    from config.settings import CDP_KEY_FILE

    executor = CoinbaseExecutor(str(CDP_KEY_FILE))

    if sell_all:
        # Get balance for this token
        balances = executor.get_balances()
        currency = symbol.split("-")[0]
        amount = balances.get(currency, 0)
        if amount <= 0:
            return {"error": f"No {currency} balance to sell."}

    if amount <= 0:
        return {"error": "Must specify amount or set all=true."}

    result = executor.market_sell(symbol, amount)
    return {"action": "sell", "symbol": symbol, "amount": amount, "result": result}


def handle_close_all() -> dict:
    """Sell all non-USD tokens on Coinbase and cancel all Kalshi orders."""
    from exchange.coinbase import CoinbaseExecutor
    from exchange.kalshi import KalshiClient
    from config.settings import CDP_KEY_FILE, KALSHI_KEY_FILE, KALSHI_API_KEY_ID

    results = {"coinbase_sells": [], "kalshi_cancels": [], "errors": []}

    # Coinbase: sell all tokens
    try:
        executor = CoinbaseExecutor(str(CDP_KEY_FILE))
        balances = executor.get_balances()
        for currency, amount in balances.items():
            if currency in ("USD", "USDC", "USDT"):
                continue
            if amount > 0:
                try:
                    cb_symbol = f"{currency}-USD"
                    result = executor.market_sell(cb_symbol, amount)
                    results["coinbase_sells"].append({
                        "symbol": cb_symbol,
                        "amount": amount,
                        "result": result,
                    })
                except Exception as e:
                    results["errors"].append(f"Sell {currency}: {e}")
    except Exception as e:
        results["errors"].append(f"Coinbase: {e}")

    # Kalshi: cancel all resting orders
    try:
        client = KalshiClient(
            api_key_id=KALSHI_API_KEY_ID,
            private_key_path=str(KALSHI_KEY_FILE),
            demo=False,
        )
        orders = client.get_orders(status="resting")
        for order in orders:
            try:
                order_id = order.get("order_id", "")
                if order_id:
                    client.cancel_order(order_id)
                    results["kalshi_cancels"].append(order_id)
            except Exception as e:
                results["errors"].append(f"Cancel {order_id}: {e}")
    except Exception as e:
        results["errors"].append(f"Kalshi: {e}")

    return results


def handle_kalshi_bet(asset: str, side: str, amount_usd: float) -> dict:
    """Place a Kalshi 15-min crypto prediction bet."""
    from exchange.kalshi import KalshiClient
    from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID

    client = KalshiClient(
        api_key_id=KALSHI_API_KEY_ID,
        private_key_path=str(KALSHI_KEY_FILE),
        demo=False,
    )

    # Find current 15-min contracts for this asset
    series_tickers = {
        "BTC": ["KXBTC", "KXBTCD", "KXBTCUSD"],
        "ETH": ["KXETH", "KXETHD", "KXETHUSD"],
        "SOL": ["KXSOL", "KXSOLD", "KXSOLUSD"],
        "XRP": ["KXXRP", "KXXRPD", "KXXRPUSD"],
    }

    markets = []
    for series in series_tickers.get(asset, []):
        try:
            found = client.get_markets(series_ticker=series, status="open")
            if found:
                markets.extend(found)
                break
        except Exception:
            continue

    if not markets:
        return {"error": f"No open Kalshi markets found for {asset}."}

    # Find the one expiring soonest (within 5-30 min)
    import time
    now = time.time()
    candidates = []
    for m in markets:
        exp = m.get("expiration_time") or m.get("close_time", "")
        if exp and "T" in exp:
            try:
                exp_ts = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
                mins = (exp_ts - now) / 60
                if 2 < mins < 30:
                    m["_mins_to_expiry"] = round(mins, 1)
                    candidates.append(m)
            except Exception:
                pass

    if not candidates:
        return {"error": f"No near-term contracts found for {asset}."}

    target = sorted(candidates, key=lambda x: x["_mins_to_expiry"])[0]
    ticker = target["ticker"]

    # Get order book to find the ask price
    try:
        ob = client.get_orderbook(ticker)
    except Exception:
        ob = {}

    # Determine price in cents
    if side == "yes":
        asks = ob.get("yes", [])
        if asks:
            price_cents = asks[0][0] if isinstance(asks[0], list) else asks[0].get("price", 50)
        else:
            price_cents = 50
    else:
        asks = ob.get("no", [])
        if asks:
            price_cents = asks[0][0] if isinstance(asks[0], list) else asks[0].get("price", 50)
        else:
            price_cents = 50

    # Calculate number of contracts
    contracts = max(1, int(amount_usd * 100 / price_cents))

    result = client.place_order(
        ticker=ticker,
        side=side,
        count=contracts,
        price_cents=price_cents,
        order_type="limit",
    )

    return {
        "action": "kalshi_bet",
        "asset": asset,
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "price_cents": price_cents,
        "mins_to_expiry": target.get("_mins_to_expiry"),
        "result": result,
    }


def handle_kalshi_markets(asset: str = None) -> dict:
    """List available Kalshi 15-min crypto contracts."""
    from exchange.kalshi import KalshiClient
    from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
    import time

    client = KalshiClient(
        api_key_id=KALSHI_API_KEY_ID,
        private_key_path=str(KALSHI_KEY_FILE),
        demo=False,
    )

    all_series = {
        "BTC": ["KXBTC", "KXBTCD", "KXBTCUSD"],
        "ETH": ["KXETH", "KXETHD", "KXETHUSD"],
        "SOL": ["KXSOL", "KXSOLD", "KXSOLUSD"],
        "XRP": ["KXXRP", "KXXRPD", "KXXRPUSD"],
    }

    if asset:
        search = {asset: all_series.get(asset, [])}
    else:
        search = all_series

    now = time.time()
    results = []

    for asset_name, series_list in search.items():
        for series in series_list:
            try:
                markets = client.get_markets(series_ticker=series, status="open")
                for m in markets:
                    exp = m.get("expiration_time") or m.get("close_time", "")
                    mins = None
                    if exp and "T" in exp:
                        try:
                            exp_ts = datetime.fromisoformat(exp.replace("Z", "+00:00")).timestamp()
                            mins = round((exp_ts - now) / 60, 1)
                        except Exception:
                            pass
                    results.append({
                        "asset": asset_name,
                        "ticker": m.get("ticker"),
                        "title": m.get("title", m.get("subtitle", "")),
                        "yes_price": m.get("yes_ask", m.get("last_price")),
                        "no_price": m.get("no_ask"),
                        "volume": m.get("volume"),
                        "mins_to_expiry": mins,
                    })
                if markets:
                    break  # found markets for this series
            except Exception:
                continue

    results.sort(key=lambda x: x.get("mins_to_expiry") or 9999)
    return {"markets": results, "count": len(results)}


def handle_ticker(symbol: str) -> dict:
    """Get current ticker data for a symbol."""
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    ticker = fetcher.ticker(symbol)

    return {
        "symbol": symbol,
        "last": ticker.get("last"),
        "bid": ticker.get("bid"),
        "ask": ticker.get("ask"),
        "high": ticker.get("high"),
        "low": ticker.get("low"),
        "volume": ticker.get("baseVolume"),
        "quote_volume": ticker.get("quoteVolume"),
        "change_pct": ticker.get("percentage"),
        "timestamp": ticker.get("datetime"),
    }


def handle_indicators(symbol: str, timeframe: str = "15m", limit: int = 100) -> dict:
    """Get technical indicators for a pair."""
    from data.fetcher import DataFetcher
    from data.indicators import add_indicators
    import pandas as pd

    fetcher = DataFetcher()
    df = fetcher.ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_indicators(df)

    last = df.iloc[-1]

    def safe_float(val):
        if pd.isna(val):
            return None
        return round(float(val), 6)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(df),
        "last_close": safe_float(last["close"]),
        "rsi": safe_float(last.get("rsi")),
        "bb_lower": safe_float(last.get("bb_lower")),
        "bb_middle": safe_float(last.get("bb_middle")),
        "bb_upper": safe_float(last.get("bb_upper")),
        "macd": safe_float(last.get("macd")),
        "macd_hist": safe_float(last.get("macd_hist")),
        "macd_signal": safe_float(last.get("macd_signal")),
        "atr": safe_float(last.get("atr")),
        "ema_12": safe_float(last.get("ema_12")),
        "ema_26": safe_float(last.get("ema_26")),
        "sma_20": safe_float(last.get("sma_20")),
        "sma_50": safe_float(last.get("sma_50")),
        "volume": safe_float(last.get("volume")),
        "vol_sma_20": safe_float(last.get("vol_sma_20")),
    }


def handle_orderbook(symbol: str, depth: int = 20) -> dict:
    """Get order book imbalance, bid/ask volumes, spread, wall detection."""
    from data.market_data import get_order_book_imbalance

    return get_order_book_imbalance(symbol, depth=depth)


def handle_tradeflow(symbol: str, limit: int = 200) -> dict:
    """Get net trade flow, buy ratio, and large trade bias."""
    from data.market_data import get_trade_flow

    return get_trade_flow(symbol, limit=limit)


def handle_funding_rates() -> dict:
    """Get current Binance USDM perpetual funding rates."""
    from data.fetcher import DataFetcher

    fetcher = DataFetcher()
    rates = fetcher.funding_rates_all()

    # Return top 20 by absolute annualized rate
    top = []
    for r in rates[:20]:
        top.append({
            "symbol": r.get("symbol"),
            "funding_rate": r.get("fundingRate"),
            "annualized": r.get("annualized"),
            "timestamp": r.get("datetime"),
        })
    return {"funding_rates": top, "count": len(top)}


def handle_signals() -> dict:
    """Run all strategies on all pairs RIGHT NOW and return signals."""
    from data.fetcher import DataFetcher
    from data.indicators import add_indicators
    from data.market_data import get_order_book_imbalance, get_trade_flow
    from config.pair_config import ALL_PAIRS, COINBASE_MAP, get_pair_config
    from config.production import STOP_LOSS_PCT
    from strategy.strategies.kalshi_predictor import KalshiPredictor
    import pandas as pd

    fetcher = DataFetcher()
    predictor = KalshiPredictor()

    all_signals = []
    kalshi_predictions = []
    pair_summaries = {}

    for symbol in ALL_PAIRS:
        try:
            df = fetcher.ohlcv(symbol, timeframe="15m", limit=200)
            df = add_indicators(df)
        except Exception as e:
            pair_summaries[symbol] = {"error": str(e)}
            continue

        if df is None or len(df) < 25:
            pair_summaries[symbol] = {"error": "Insufficient data"}
            continue

        last = df.iloc[-1]
        close = float(last["close"])
        rsi = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50
        bb_lower = float(last.get("bb_lower", 0)) if pd.notna(last.get("bb_lower")) else 0
        bb_upper = float(last.get("bb_upper", 0)) if pd.notna(last.get("bb_upper")) else 0
        bb_mid = float(last.get("bb_middle", 0)) if pd.notna(last.get("bb_middle")) else 0
        atr = float(last.get("atr", 0)) if pd.notna(last.get("atr")) else 0

        cfg = get_pair_config(symbol)
        coinbase_sym = COINBASE_MAP.get(symbol, symbol)
        leverage = cfg.get("leverage", 1)

        pair_summaries[symbol] = {
            "price": close, "rsi": round(rsi, 1),
            "bb_lower": round(bb_lower, 6), "bb_mid": round(bb_mid, 6),
            "bb_upper": round(bb_upper, 6), "atr": round(atr, 6),
        }

        # BB Grid signals
        if "bb_grid" in cfg.get("enabled_strategies", []):
            if close < bb_lower and rsi < cfg["bb_rsi_buy"]:
                all_signals.append({
                    "symbol": symbol, "coinbase_symbol": coinbase_sym,
                    "action": "BUY", "strategy": f"bb_grid_{leverage}x",
                    "price": close, "rsi": round(rsi, 1),
                    "stop": round(close * (1 - STOP_LOSS_PCT), 6),
                    "take_profit": round(bb_mid, 6),
                })
            if close > bb_upper and rsi > cfg["bb_rsi_short"]:
                all_signals.append({
                    "symbol": symbol, "coinbase_symbol": coinbase_sym,
                    "action": "SHORT", "strategy": f"bb_grid_{leverage}x",
                    "price": close, "rsi": round(rsi, 1),
                    "stop": round(close * (1 + STOP_LOSS_PCT), 6),
                    "take_profit": round(bb_mid, 6),
                })

        # RSI Mean Reversion signals
        if "rsi_mr" in cfg.get("enabled_strategies", []):
            if rsi < cfg["rsi_mr_oversold"]:
                all_signals.append({
                    "symbol": symbol, "coinbase_symbol": coinbase_sym,
                    "action": "BUY", "strategy": f"rsi_mr_{leverage}x",
                    "price": close, "rsi": round(rsi, 1),
                    "stop": round(close * (1 - STOP_LOSS_PCT), 6),
                })
            if rsi > cfg["rsi_mr_overbought"]:
                all_signals.append({
                    "symbol": symbol, "coinbase_symbol": coinbase_sym,
                    "action": "SHORT", "strategy": f"rsi_mr_{leverage}x",
                    "price": close, "rsi": round(rsi, 1),
                    "stop": round(close * (1 + STOP_LOSS_PCT), 6),
                })

        # Leading indicators + Kalshi prediction
        try:
            ob_data = get_order_book_imbalance(symbol)
            tf_data = get_trade_flow(symbol)
            market_data = {"order_book": ob_data, "trade_flow": tf_data}
            pair_summaries[symbol]["order_book"] = ob_data
            pair_summaries[symbol]["trade_flow"] = tf_data

            sig = predictor.score(df, market_data=market_data)
            if sig:
                asset = symbol.split("/")[0]
                kalshi_predictions.append({
                    "asset": asset,
                    "direction": sig.direction,
                    "confidence": sig.confidence,
                    "components": sig.components,
                    "ob_imbalance": ob_data.get("imbalance", 0),
                    "net_flow": tf_data.get("net_flow", 0),
                })
        except Exception:
            pass

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signals": all_signals,
        "num_signals": len(all_signals),
        "kalshi_predictions": kalshi_predictions,
        "pair_summaries": pair_summaries,
    }


def handle_backtest(symbol: str, strategy: str, timeframe: str = "15m",
                    days: int = 30, leverage: int = 1) -> dict:
    """Run a quick backtest and return results."""
    from data.fetcher import DataFetcher
    from data.indicators import add_indicators
    from strategy.compound_backtest import compound_backtest
    from config.pair_config import get_pair_config
    import pandas as pd

    fetcher = DataFetcher()

    # Calculate how many candles we need
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 15)
    limit = min(1000, int(days * 24 * 60 / minutes))

    df = fetcher.ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = add_indicators(df)

    cfg = get_pair_config(symbol)

    if strategy == "bb_grid":
        buy_thresh = cfg["bb_rsi_buy"]
        short_thresh = cfg["bb_rsi_short"]

        def buy_fn(row, prev):
            if pd.isna(row.get("bb_lower")) or pd.isna(row.get("rsi")):
                return False
            return float(row["close"]) < float(row["bb_lower"]) and float(row["rsi"]) < buy_thresh

        def sell_fn(row, prev):
            if pd.isna(row.get("bb_middle")):
                return False
            return float(row["close"]) > float(row["bb_middle"])

        def short_fn(row, prev):
            if pd.isna(row.get("bb_upper")) or pd.isna(row.get("rsi")):
                return False
            return float(row["close"]) > float(row["bb_upper"]) and float(row["rsi"]) > short_thresh

        def cover_fn(row, prev):
            if pd.isna(row.get("bb_middle")):
                return False
            return float(row["close"]) < float(row["bb_middle"])

        result = compound_backtest(
            df, buy_fn=buy_fn, sell_fn=sell_fn,
            short_fn=short_fn, cover_fn=cover_fn,
            leverage=leverage, stop_loss_pct=0.03,
        )
    elif strategy == "rsi_mr":
        oversold = cfg["rsi_mr_oversold"]
        overbought = cfg["rsi_mr_overbought"]
        exit_long = cfg["rsi_mr_exit_long"]
        exit_short = cfg["rsi_mr_exit_short"]

        def buy_fn(row, prev):
            if pd.isna(row.get("rsi")):
                return False
            return float(row["rsi"]) < oversold

        def sell_fn(row, prev):
            if pd.isna(row.get("rsi")):
                return False
            return float(row["rsi"]) > exit_long

        def short_fn(row, prev):
            if pd.isna(row.get("rsi")):
                return False
            return float(row["rsi"]) > overbought

        def cover_fn(row, prev):
            if pd.isna(row.get("rsi")):
                return False
            return float(row["rsi"]) < exit_short

        result = compound_backtest(
            df, buy_fn=buy_fn, sell_fn=sell_fn,
            short_fn=short_fn, cover_fn=cover_fn,
            leverage=leverage, stop_loss_pct=0.03,
        )
    else:
        return {"error": f"Unknown strategy: {strategy}"}

    # Strip the full trades list and equity curve for brevity
    summary = {k: v for k, v in result.items() if k not in ("trades", "equity_curve")}
    summary["symbol"] = symbol
    summary["strategy"] = strategy
    summary["timeframe"] = timeframe
    summary["days"] = days
    summary["leverage"] = leverage
    # Include last 5 trades for context
    summary["recent_trades"] = result.get("trades", [])[-5:]
    return summary


def handle_performance() -> dict:
    """Get session performance stats from trades.csv."""
    from data.store import DataStore
    from config.settings import DATA_DIR

    store = DataStore(DATA_DIR)
    df = store.load_trades()

    if df.empty:
        return {"message": "No trades recorded yet.", "num_trades": 0}

    num_trades = len(df)
    if "pnl" in df.columns:
        pnl_col = "pnl"
    elif "pnl_usd" in df.columns:
        pnl_col = "pnl_usd"
    else:
        return {"num_trades": num_trades, "columns": list(df.columns)}

    df[pnl_col] = df[pnl_col].astype(float)
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] <= 0]
    total_pnl = float(df[pnl_col].sum())
    avg_win = float(wins[pnl_col].mean()) if len(wins) > 0 else 0
    avg_loss = float(losses[pnl_col].mean()) if len(losses) > 0 else 0
    win_rate = len(wins) / num_trades * 100 if num_trades > 0 else 0
    gross_profit = float(wins[pnl_col].sum()) if len(wins) > 0 else 0
    gross_loss = abs(float(losses[pnl_col].sum())) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "num_trades": num_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "best_trade": round(float(df[pnl_col].max()), 2),
        "worst_trade": round(float(df[pnl_col].min()), 2),
    }


def handle_trade_history(limit: int = 20) -> dict:
    """Get recent trades from trades.csv."""
    from data.store import DataStore
    from config.settings import DATA_DIR

    store = DataStore(DATA_DIR)
    df = store.load_trades()

    if df.empty:
        return {"trades": [], "count": 0}

    recent = df.tail(limit)
    trades = recent.to_dict(orient="records")
    return {"trades": trades, "count": len(trades)}


def handle_config_get() -> dict:
    """Get all current configuration."""
    from config.pair_config import PAIR_CONFIG, DEFAULT_PAIR_CONFIG, ALL_PAIRS
    from config.settings import (
        MAX_POSITION_PCT, MAX_CONCURRENT, MAX_LEVERAGE, MAX_DAILY_DRAWDOWN_PCT,
        MIN_BALANCE_USD, DEFAULT_ATR_MULTIPLIER, TICK_INTERVAL, SIGNAL_INTERVAL,
    )
    from config.production import (
        POSITION_SIZE_PCT, STOP_LOSS_PCT, MAX_CONCURRENT_POSITIONS,
        BB_GRID_CONFIG, RSI_MR_CONFIG,
    )

    return {
        "pair_config": PAIR_CONFIG,
        "default_pair_config": DEFAULT_PAIR_CONFIG,
        "all_pairs": ALL_PAIRS,
        "risk_settings": {
            "max_position_pct": MAX_POSITION_PCT,
            "max_concurrent": MAX_CONCURRENT,
            "max_leverage": MAX_LEVERAGE,
            "max_daily_drawdown_pct": MAX_DAILY_DRAWDOWN_PCT,
            "min_balance_usd": MIN_BALANCE_USD,
        },
        "production": {
            "position_size_pct": POSITION_SIZE_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
            "bb_grid_config": BB_GRID_CONFIG,
            "rsi_mr_config": RSI_MR_CONFIG,
        },
        "intervals": {
            "tick_interval": TICK_INTERVAL,
            "signal_interval": SIGNAL_INTERVAL,
        },
    }


def handle_config_set(pair: str, key: str, value) -> dict:
    """Update a pair config value at runtime."""
    from config.pair_config import PAIR_CONFIG, DEFAULT_PAIR_CONFIG

    if pair not in PAIR_CONFIG:
        return {"error": f"Unknown pair: {pair}. Known: {list(PAIR_CONFIG.keys())}"}

    cfg = PAIR_CONFIG[pair]
    if key not in cfg and key not in DEFAULT_PAIR_CONFIG:
        return {"error": f"Unknown key: {key}. Known: {list(DEFAULT_PAIR_CONFIG.keys())}"}

    old_value = cfg.get(key)
    # Type coerce
    if isinstance(old_value, int) and not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            pass
    elif isinstance(old_value, float) and not isinstance(value, float):
        try:
            value = float(value)
        except (ValueError, TypeError):
            pass

    cfg[key] = value
    PAIR_CONFIG[pair] = cfg

    return {
        "pair": pair,
        "key": key,
        "old_value": old_value,
        "new_value": value,
        "updated_config": cfg,
    }


# ===================================================================
#  Market maker handlers
# ===================================================================

def handle_mm_start(mode="dry-run"):
    global _mm_process
    if _daemon_process and _daemon_process.poll() is None:
        return {"error": "Predictor daemon is running. Stop it first (algotrade_stop)."}
    if _mm_process and _mm_process.poll() is None:
        return {"error": "Market maker already running.", "pid": _mm_process.pid}
    cmd = [sys.executable, "-m", "kalshi_mm.mm_daemon", "--mode", mode]
    log_path = "/tmp/mm_stdout.log"
    with open(log_path, "w") as log_f:
        _mm_process = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
    return {"status": "started", "mode": mode, "pid": _mm_process.pid}


def handle_mm_stop():
    global _mm_process
    if not _mm_process or _mm_process.poll() is not None:
        return {"status": "not running"}
    _mm_process.terminate()
    try:
        _mm_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _mm_process.kill()
    _mm_process = None
    return {"status": "stopped"}


def handle_mm_status():
    dash_path = Path("data/store/mm_dashboard.log")
    result = {}
    if dash_path.exists():
        result["dashboard"] = dash_path.read_text()
    else:
        result["dashboard"] = "No dashboard data yet."
    if _mm_process and _mm_process.poll() is None:
        result["running"] = True
        result["pid"] = _mm_process.pid
    else:
        result["running"] = False
    return result


# ===================================================================
#  Dispatcher
# ===================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "algotrade_status":
            result = handle_status()
        elif name == "algotrade_start":
            result = handle_start(
                mode=arguments.get("mode", "dry-run"),
                cycles=arguments.get("cycles", 50),
                kalshi_only=arguments.get("kalshi_only", False),
                predictor=arguments.get("predictor", "v1"),
            )
        elif name == "algotrade_stop":
            result = handle_stop()
        elif name == "algotrade_balances":
            result = handle_balances()
        elif name == "algotrade_positions":
            result = handle_positions()
        elif name == "algotrade_buy":
            result = handle_buy(
                symbol=arguments["symbol"],
                usd_amount=arguments["usd_amount"],
            )
        elif name == "algotrade_sell":
            result = handle_sell(
                symbol=arguments["symbol"],
                amount=arguments.get("amount", 0),
                sell_all=arguments.get("all", False),
            )
        elif name == "algotrade_close_all":
            result = handle_close_all()
        elif name == "algotrade_kalshi_bet":
            result = handle_kalshi_bet(
                asset=arguments["asset"],
                side=arguments["side"],
                amount_usd=arguments["amount_usd"],
            )
        elif name == "algotrade_kalshi_markets":
            result = handle_kalshi_markets(
                asset=arguments.get("asset"),
            )
        elif name == "algotrade_ticker":
            result = handle_ticker(symbol=arguments["symbol"])
        elif name == "algotrade_indicators":
            result = handle_indicators(
                symbol=arguments["symbol"],
                timeframe=arguments.get("timeframe", "15m"),
                limit=arguments.get("limit", 100),
            )
        elif name == "algotrade_orderbook":
            result = handle_orderbook(
                symbol=arguments["symbol"],
                depth=arguments.get("depth", 20),
            )
        elif name == "algotrade_tradeflow":
            result = handle_tradeflow(
                symbol=arguments["symbol"],
                limit=arguments.get("limit", 200),
            )
        elif name == "algotrade_funding_rates":
            result = handle_funding_rates()
        elif name == "algotrade_signals":
            result = handle_signals()
        elif name == "algotrade_backtest":
            result = handle_backtest(
                symbol=arguments["symbol"],
                strategy=arguments["strategy"],
                timeframe=arguments.get("timeframe", "15m"),
                days=arguments.get("days", 30),
                leverage=arguments.get("leverage", 1),
            )
        elif name == "algotrade_performance":
            result = handle_performance()
        elif name == "algotrade_trade_history":
            result = handle_trade_history(
                limit=arguments.get("limit", 20),
            )
        elif name == "algotrade_config_get":
            result = handle_config_get()
        elif name == "algotrade_config_set":
            result = handle_config_set(
                pair=arguments["pair"],
                key=arguments["key"],
                value=arguments["value"],
            )
        elif name == "algotrade_force_kill":
            result = handle_force_kill()
        elif name == "algotrade_logs":
            result = handle_logs(lines=arguments.get("lines", 50))
        elif name == "algotrade_errors":
            result = handle_errors(lines=arguments.get("lines", 30))
        elif name == "algotrade_mm_start":
            result = handle_mm_start(mode=arguments.get("mode", "dry-run"))
        elif name == "algotrade_mm_stop":
            result = handle_mm_stop()
        elif name == "algotrade_mm_status":
            result = handle_mm_status()
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        result = {"error": str(e), "tool": name}

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


# ===================================================================
#  Main entry point
# ===================================================================

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
