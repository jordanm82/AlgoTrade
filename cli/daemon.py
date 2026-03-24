# cli/daemon.py
import json
import time
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from config.settings import (
    CDP_KEY_FILE, DATA_DIR, MAX_CONCURRENT,
    TICK_INTERVAL, SIGNAL_INTERVAL,
)
from config.tokens import SPOT_PAIRS, BINANCE_SPOT
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from data.store import DataStore
from exchange.coinbase import CoinbaseExecutor
from exchange.positions import PositionTracker
from risk.manager import RiskManager
from strategy.base import BaseStrategy


class Daemon:
    def __init__(self, strategies: list[BaseStrategy] | None = None):
        self.executor = CoinbaseExecutor(str(CDP_KEY_FILE))
        self.fetcher = DataFetcher()
        self.store = DataStore(DATA_DIR)
        self.tracker = PositionTracker(max_concurrent=MAX_CONCURRENT)
        self.strategies = strategies or []
        self._running = False

        balances = self.executor.get_balances()
        portfolio = sum(balances.values())
        self.risk = RiskManager(portfolio_value=portfolio)
        self.risk.record_daily_start(portfolio)

    def _enforce_stops(self):
        """Check and execute stop-losses. Runs every tick."""
        stopped = self.tracker.check_stops()
        for pos in stopped:
            symbol = pos["symbol"]
            print(f"[STOP] {symbol} hit stop at {pos['stop_price']}")
            if "PERP" in symbol:
                self.executor.close_perp(symbol)
            else:
                base_size = pos["size_usd"] / pos["current_price"] if pos["current_price"] > 0 else 0
                self.executor.market_sell(symbol, base_size)
            self.tracker.close(symbol, pos["current_price"])
            self.store.append_trade({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol, "side": "STOP_CLOSE",
                "size_usd": pos["size_usd"], "price": pos["current_price"],
                "stop": pos["stop_price"], "source": "daemon_stop",
            })

    def _update_prices(self):
        """Update current prices for all open positions."""
        for pos in self.tracker.open_positions():
            symbol = pos["symbol"]
            try:
                binance_sym = symbol.replace("-USD", "/USDT").replace("-PERP-INTX", "/USDT:USDT")
                ticker = self.fetcher.ticker(binance_sym)
                self.tracker.update_price(symbol, ticker["last"])
            except Exception as e:
                print(f"[WARN] Price update failed for {symbol}: {e}")

    def _collect_snapshot(self, dataframes: dict) -> dict:
        """Collect signals from all strategies and build snapshot."""
        all_signals = []
        for strat in self.strategies:
            for symbol, df in dataframes.items():
                try:
                    signals = strat.signals(df)
                    for s in signals:
                        all_signals.append({
                            "strategy": strat.name,
                            "symbol": s.symbol or symbol,
                            "direction": s.direction,
                            "strength": s.strength,
                            "stop_price": s.stop_price,
                            "take_profit": s.take_profit,
                            "metadata": s.metadata,
                        })
                except Exception as e:
                    print(f"[WARN] Strategy {strat.name} failed on {symbol}: {e}")

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": all_signals,
            "positions": self.tracker.open_positions(),
            "exposure": self.tracker.total_exposure(),
            "halted": self.risk.is_halted(self.tracker.total_exposure()),
        }
        return snapshot

    def _fetch_all_data(self) -> dict:
        """Fetch OHLCV + indicators for all monitored pairs."""
        dataframes = {}
        for binance_sym, coinbase_sym in BINANCE_SPOT.items():
            try:
                df = self.fetcher.ohlcv(binance_sym, "1h", limit=100)
                df = add_indicators(df)
                dataframes[coinbase_sym] = df
            except Exception as e:
                print(f"[WARN] Data fetch failed for {binance_sym}: {e}")
        return dataframes

    def tick(self):
        """Single daemon tick -- update prices and enforce stops."""
        self._update_prices()
        self._enforce_stops()

        balances = self.executor.get_balances()
        portfolio = sum(balances.values()) + self.tracker.total_exposure()
        self.risk.update_portfolio_value(portfolio)

    def signal_tick(self):
        """Signal collection tick -- fetch data, run strategies, write snapshot."""
        dataframes = self._fetch_all_data()
        snapshot = self._collect_snapshot(dataframes)
        self.store.save_snapshot(snapshot)

        n_signals = len(snapshot["signals"])
        if n_signals > 0:
            print(f"[SIGNAL] {n_signals} signals pending review")
        if snapshot["halted"]:
            print("[HALT] Daily drawdown limit reached -- trading halted")

    def run(self):
        """Main daemon loop."""
        self._running = True
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_running", False))

        print(f"[DAEMON] Started at {datetime.now(timezone.utc).isoformat()}")
        print(f"[DAEMON] Monitoring {list(BINANCE_SPOT.values())}")
        print(f"[DAEMON] {len(self.strategies)} strategies loaded")

        last_signal = 0
        while self._running:
            try:
                self.tick()
                now = time.time()
                if now - last_signal >= SIGNAL_INTERVAL:
                    self.signal_tick()
                    last_signal = now
                time.sleep(TICK_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(TICK_INTERVAL)

        print("[DAEMON] Stopped")


def main():
    daemon = Daemon()
    daemon.run()


if __name__ == "__main__":
    main()
