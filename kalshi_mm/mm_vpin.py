# kalshi_mm/mm_vpin.py
"""VPIN computation and kill switch logic for market maker."""
from collections import deque

from kalshi_mm.mm_config import (
    VPIN_SAFE, VPIN_CAUTION, VPIN_SPOT_WEIGHT, VPIN_KALSHI_WEIGHT,
    VOLATILITY_SPIKE_PCT,
)


def compute_spot_vpin(trade_flow: dict) -> float:
    """Compute VPIN proxy from BinanceUS spot trade flow.
    Uses abs(net_flow) — already |buy_vol - sell_vol| / total_vol."""
    net_flow = trade_flow.get("net_flow", 0)
    return abs(net_flow)


def compute_kalshi_ob_heuristic(
    prev_yes_vol: float, curr_yes_vol: float,
    prev_no_vol: float, curr_no_vol: float,
) -> float:
    """Orderbook-change heuristic. One-sided shifts = informed flow."""
    delta_yes = curr_yes_vol - prev_yes_vol
    delta_no = curr_no_vol - prev_no_vol
    denom = abs(delta_yes) + abs(delta_no) + 1
    return abs(delta_yes - delta_no) / denom


def compute_blended_vpin(spot_vpin: float, kalshi_heuristic: float) -> float:
    """Blend spot VPIN and Kalshi OB heuristic."""
    return VPIN_SPOT_WEIGHT * spot_vpin + VPIN_KALSHI_WEIGHT * kalshi_heuristic


class KillSwitch:
    """Monitors flow toxicity and volatility for kill switch decisions."""

    def __init__(self):
        self._vpin_history: deque[float] = deque(maxlen=10)
        self._prices: deque[float] = deque(maxlen=2)

    def should_go_dark(self, vpin: float) -> bool:
        return vpin >= VPIN_CAUTION

    def get_spread_state(self, vpin: float) -> str:
        if vpin < VPIN_SAFE:
            return "SAFE"
        elif vpin < VPIN_CAUTION:
            return "CAUTION"
        return "TOXIC"

    def record_vpin(self, vpin: float):
        self._vpin_history.append(vpin)

    def vpin_rising(self) -> bool:
        if len(self._vpin_history) < 3:
            return False
        vals = list(self._vpin_history)
        return vals[-1] > vals[-2] > vals[-3]

    def record_price(self, price: float):
        self._prices.append(price)

    def volatility_spike(self) -> bool:
        if len(self._prices) < 2:
            return False
        old, new = self._prices[0], self._prices[1]
        if old == 0:
            return False
        return abs(new - old) / old > VOLATILITY_SPIKE_PCT
