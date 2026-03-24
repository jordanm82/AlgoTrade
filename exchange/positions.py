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

    def to_dict(self) -> dict:
        return asdict(self)


class PositionTracker:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._positions: dict[str, Position] = {}
        self._closed: list[dict] = []

    def open(self, symbol: str, side: str, size_usd: float,
             entry_price: float, stop_price: float, take_profit: float):
        self._positions[symbol] = Position(
            symbol=symbol, side=side, size_usd=size_usd,
            entry_price=entry_price, stop_price=stop_price,
            take_profit=take_profit, current_price=entry_price,
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
