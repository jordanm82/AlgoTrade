# strategy/strategies/grid_bot.py
"""Grid Bot — places buy/sell orders at fixed percentage intervals.
Profits from oscillations in ranging markets."""
import pandas as pd
from strategy.base import BaseStrategy, Signal


class GridBot(BaseStrategy):
    name = "grid_bot"

    def __init__(self, grid_pct: float = 0.01, num_grids: int = 5):
        self.grid_pct = grid_pct
        self.num_grids = num_grids
        self._reference_price: float | None = None
        self._filled_buys: set[int] = set()
        self._filled_sells: set[int] = set()

    def set_reference_price(self, price: float):
        self._reference_price = price
        self._filled_buys.clear()
        self._filled_sells.clear()

    def get_grid_levels(self) -> dict:
        if self._reference_price is None:
            return {"buy_levels": [], "sell_levels": []}
        ref = self._reference_price
        buy_levels = [ref * (1 - self.grid_pct * (i + 1)) for i in range(self.num_grids)]
        sell_levels = [ref * (1 + self.grid_pct * (i + 1)) for i in range(self.num_grids)]
        return {"buy_levels": buy_levels, "sell_levels": sell_levels}

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        if len(df) < 2:
            return []
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(last["close"])
        prev_close = float(prev["close"])

        if self._reference_price is None:
            return []

        grids = self.get_grid_levels()
        signals = []

        for i, level in enumerate(grids["buy_levels"]):
            if i not in self._filled_buys and prev_close > level >= close:
                self._filled_buys.add(i)
                sell_target = grids["sell_levels"][0] if i == 0 else grids["buy_levels"][i - 1]
                signals.append(Signal(
                    symbol="", direction="BUY",
                    strength=min(1.0, (i + 1) / self.num_grids),
                    stop_price=level * (1 - self.grid_pct * 2),
                    take_profit=sell_target,
                    metadata={"reason": f"Grid buy level {i+1}", "grid_price": level},
                ))
                if i in self._filled_sells:
                    self._filled_sells.discard(i)

        for i, level in enumerate(grids["sell_levels"]):
            if i not in self._filled_sells and prev_close < level <= close:
                self._filled_sells.add(i)
                buy_target = grids["buy_levels"][0] if i == 0 else grids["sell_levels"][i - 1]
                signals.append(Signal(
                    symbol="", direction="SELL",
                    strength=min(1.0, (i + 1) / self.num_grids),
                    stop_price=level * (1 + self.grid_pct * 2),
                    take_profit=buy_target,
                    metadata={"reason": f"Grid sell level {i+1}", "grid_price": level},
                ))
                if i in self._filled_buys:
                    self._filled_buys.discard(i)

        return signals
