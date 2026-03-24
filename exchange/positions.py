from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Position:
    symbol: str
    side: str  # "BUY" or "SELL"
    size_usd: float
    entry_price: float
    stop_price: float
    take_profit: float
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    units: float = 0.0
    original_size_usd: float = 0.0   # initial position size
    tp_10_hit: bool = False           # +10% level hit, sold 25%
    tp_20_hit: bool = False           # +20% level hit, sold another 25%
    trailing_stop: float = 0.0        # trailing stop price (0 = not active)
    peak_price: float = 0.0           # highest price since entry (for trailing)

    def update(self, price: float):
        self.current_price = price
        if self.side == "BUY":
            self.unrealized_pnl = (price - self.entry_price) / self.entry_price * self.size_usd
        else:
            self.unrealized_pnl = (self.entry_price - price) / self.entry_price * self.size_usd

    def stop_hit(self) -> bool:
        if self.current_price == 0:
            return False
        if self.side == "BUY":
            return self.current_price <= self.stop_price
        else:
            return self.current_price >= self.stop_price

    def check_profit_taking(self) -> list[dict]:
        """Check if any profit-taking levels are hit.
        Returns list of actions: [{"action": "partial_sell", "pct": 25, "reason": "TP +10%"}, ...]
        """
        actions = []
        if self.current_price == 0 or self.entry_price == 0:
            return actions

        pct_gain = (self.current_price - self.entry_price) / self.entry_price * 100
        if self.side == "SELL":  # short
            pct_gain = (self.entry_price - self.current_price) / self.entry_price * 100

        # Update peak price for trailing stop
        if self.side == "BUY":
            if self.current_price > self.peak_price:
                self.peak_price = self.current_price
        else:
            if self.peak_price == 0 or self.current_price < self.peak_price:
                self.peak_price = self.current_price

        # +10% -> sell 25%
        if pct_gain >= 10 and not self.tp_10_hit:
            self.tp_10_hit = True
            actions.append({"action": "partial_sell", "pct": 25, "reason": f"TP +10% (gain={pct_gain:.1f}%)"})

        # +20% -> sell another 25%
        if pct_gain >= 20 and not self.tp_20_hit:
            self.tp_20_hit = True
            actions.append({"action": "partial_sell", "pct": 25, "reason": f"TP +20% (gain={pct_gain:.1f}%)"})

        # Activate trailing stop after first TP hit (trail at 5% below peak)
        if self.tp_10_hit and self.peak_price > 0:
            if self.side == "BUY":
                self.trailing_stop = self.peak_price * 0.95
            else:
                self.trailing_stop = self.peak_price * 1.05

        # Check trailing stop
        if self.trailing_stop > 0:
            if self.side == "BUY" and self.current_price <= self.trailing_stop:
                actions.append({"action": "trailing_stop", "reason": f"Trail stop hit at ${self.trailing_stop:.4f}"})
            elif self.side == "SELL" and self.current_price >= self.trailing_stop:
                actions.append({"action": "trailing_stop", "reason": f"Trail stop hit at ${self.trailing_stop:.4f}"})

        return actions

    def reduce_size(self, pct: float):
        """Reduce position by pct percent."""
        reduction = self.size_usd * (pct / 100)
        self.size_usd -= reduction
        self.units -= self.units * (pct / 100)
        return reduction

    def to_dict(self) -> dict:
        return asdict(self)


class PositionTracker:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._positions: dict[str, Position] = {}
        self._closed: list[dict] = []

    def open(self, symbol: str, side: str, size_usd: float,
             entry_price: float, stop_price: float, take_profit: float):
        units = size_usd / entry_price if entry_price > 0 else 0.0
        self._positions[symbol] = Position(
            symbol=symbol, side=side, size_usd=size_usd,
            entry_price=entry_price, stop_price=stop_price,
            take_profit=take_profit, current_price=entry_price,
            units=units, original_size_usd=size_usd,
            peak_price=entry_price,
        )

    def close(self, symbol: str, exit_price: float) -> dict:
        pos = self._positions.pop(symbol)
        pos.update(exit_price)
        record = pos.to_dict()
        record["exit_price"] = exit_price
        record["pnl_usd"] = pos.unrealized_pnl
        record["closed_at"] = datetime.now(timezone.utc).isoformat()
        self._closed.append(record)
        return record

    def can_open(self) -> bool:
        return len(self._positions) < self.max_concurrent

    def open_positions(self) -> list[dict]:
        return [p.to_dict() for p in self._positions.values()]

    def update_price(self, symbol: str, price: float):
        if symbol in self._positions:
            self._positions[symbol].update(price)

    def check_stops(self) -> list[dict]:
        triggered = []
        for pos in self._positions.values():
            if pos.stop_hit():
                triggered.append(pos.to_dict())
        return triggered

    def total_exposure(self) -> float:
        return sum(p.size_usd + p.unrealized_pnl for p in self._positions.values())

    def closed_trades(self) -> list[dict]:
        return self._closed
