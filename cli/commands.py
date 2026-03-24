import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from config.settings import CDP_KEY_FILE, DATA_DIR, MAX_CONCURRENT
from data.fetcher import DataFetcher
from data.store import DataStore
from exchange.coinbase import CoinbaseExecutor
from exchange.positions import PositionTracker
from risk.manager import RiskManager


class CLI:
    def __init__(self):
        self.executor = CoinbaseExecutor(str(CDP_KEY_FILE))
        self.fetcher = DataFetcher()
        self.store = DataStore(DATA_DIR)
        self.tracker = PositionTracker(max_concurrent=MAX_CONCURRENT)
        self._init_risk()

    def _init_risk(self):
        balances = self.executor.get_balances()
        portfolio = sum(balances.values())
        self.risk = RiskManager(portfolio_value=portfolio)

    def status(self) -> dict:
        """Current positions, balances, exposure."""
        return {
            "positions": self.tracker.open_positions(),
            "exposure": self.tracker.total_exposure(),
            "balances": self.executor.get_balances(),
            "closed_trades": self.tracker.closed_trades(),
        }

    def pending(self) -> dict:
        """Pending signals awaiting review."""
        snapshot = self.store.load_latest_snapshot()
        if not snapshot:
            return {"signals": [], "message": "No pending signals"}
        return {"signals": snapshot.get("signals", [])}

    def snapshot(self) -> dict:
        """Latest full market snapshot."""
        return self.store.load_latest_snapshot() or {}

    def buy(self, symbol: str, usd_amount: float, stop: float | None = None,
            source: str = "") -> dict:
        """Execute a spot buy."""
        ok, reason = self.risk.check_entry(
            size_usd=usd_amount, leverage=1,
            current_positions=len(self.tracker.open_positions()),
        )
        if not ok:
            return {"error": reason}

        result = self.executor.market_buy(symbol, usd_amount)
        if result.get("success"):
            ticker = self.fetcher.ticker(symbol.replace("-USD", "/USDT"), source="binance")
            price = ticker.get("last", 0)
            stop_price = stop or price * 0.96  # default 4% stop
            self.tracker.open(symbol, "BUY", usd_amount, price, stop_price, price * 1.08)
            self.store.append_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "side": "BUY", "size_usd": usd_amount,
                "price": price, "stop": stop_price, "source": source,
            })
        return result

    def sell(self, symbol: str, base_size: float) -> dict:
        """Execute a spot sell or close position."""
        return self.executor.market_sell(symbol, base_size)

    def short(self, symbol: str, usd_amount: float, leverage: int = 1,
              stop: float | None = None) -> dict:
        """Open a perp short."""
        ok, reason = self.risk.check_entry(
            size_usd=usd_amount, leverage=leverage,
            current_positions=len(self.tracker.open_positions()),
        )
        if not ok:
            return {"error": reason}

        ticker = self.fetcher.ticker(symbol.replace("-PERP-INTX", "/USDT:USDT"))
        price = ticker.get("last", 0)
        base_size = usd_amount / price if price > 0 else 0

        result = self.executor.open_perp_short(symbol, base_size, leverage)
        if result.get("success"):
            stop_price = stop or price * 1.04
            self.tracker.open(symbol, "SELL", usd_amount, price, stop_price, price * 0.92)
        return result

    def long_perp(self, symbol: str, usd_amount: float, leverage: int = 1,
                  stop: float | None = None) -> dict:
        """Open a perp long."""
        ok, reason = self.risk.check_entry(
            size_usd=usd_amount, leverage=leverage,
            current_positions=len(self.tracker.open_positions()),
        )
        if not ok:
            return {"error": reason}

        ticker = self.fetcher.ticker(symbol.replace("-PERP-INTX", "/USDT:USDT"))
        price = ticker.get("last", 0)
        base_size = usd_amount / price if price > 0 else 0

        result = self.executor.open_perp_long(symbol, base_size, leverage)
        if result.get("success"):
            stop_price = stop or price * 0.96
            self.tracker.open(symbol, "BUY", usd_amount, price, stop_price, price * 1.08)
        return result

    def close(self, symbol: str) -> dict:
        """Close a position (spot sell or perp close)."""
        if "PERP" in symbol:
            result = self.executor.close_perp(symbol)
        else:
            positions = self.tracker.open_positions()
            pos = next((p for p in positions if p["symbol"] == symbol), None)
            if not pos:
                return {"error": f"No open position for {symbol}"}
            base_size = pos["size_usd"] / pos["current_price"] if pos["current_price"] > 0 else 0
            result = self.executor.market_sell(symbol, base_size)

        ticker = self.fetcher.ticker(
            symbol.replace("-PERP-INTX", "/USDT:USDT").replace("-USD", "/USDT"),
            source="binance"
        )
        exit_price = ticker.get("last", 0)
        closed = self.tracker.close(symbol, exit_price)
        return {"order": result, "closed_position": closed}

    def close_all(self) -> list[dict]:
        """Close all open positions."""
        results = []
        for pos in self.tracker.open_positions():
            results.append(self.close(pos["symbol"]))
        return results

    def cancel(self, order_id: str) -> dict:
        return self.executor.cancel_order(order_id)


def main():
    """CLI entry point: python -m cli.commands <command> [args]"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python -m cli.commands <command> [args]"}))
        sys.exit(1)

    cmd = sys.argv[1]
    cli = CLI()

    if cmd == "status":
        result = cli.status()
    elif cmd == "pending":
        result = cli.pending()
    elif cmd == "snapshot":
        result = cli.snapshot()
    elif cmd == "buy" and len(sys.argv) >= 4:
        stop = float(sys.argv[4]) if len(sys.argv) > 4 else None
        result = cli.buy(sys.argv[2], float(sys.argv[3]), stop=stop)
    elif cmd == "sell" and len(sys.argv) >= 4:
        result = cli.sell(sys.argv[2], float(sys.argv[3]))
    elif cmd == "short" and len(sys.argv) >= 4:
        leverage = int(sys.argv[4].replace("x", "")) if len(sys.argv) > 4 else 1
        result = cli.short(sys.argv[2], float(sys.argv[3]), leverage=leverage)
    elif cmd == "close" and len(sys.argv) >= 3:
        result = cli.close(sys.argv[2])
    elif cmd == "close-all":
        result = cli.close_all()
    elif cmd == "cancel" and len(sys.argv) >= 3:
        result = cli.cancel(sys.argv[2])
    else:
        result = {"error": f"Unknown command: {cmd}"}

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
