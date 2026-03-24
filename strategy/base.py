# strategy/base.py
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class Signal:
    symbol: str
    direction: str  # "BUY" or "SELL"
    strength: float  # 0.0 to 1.0
    stop_price: float
    take_profit: float
    metadata: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    name: str = "unnamed"

    @abstractmethod
    def signals(self, df: pd.DataFrame) -> list[Signal]:
        """Analyze DataFrame and return trade signals."""
        ...
