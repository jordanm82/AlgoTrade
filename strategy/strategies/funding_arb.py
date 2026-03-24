# strategy/strategies/funding_arb.py
"""Funding Rate Arbitrage — long spot, short perp when funding is high positive.
Or short spot, long perp when funding is deeply negative.

When funding rate is highly positive (>0.03% per period = ~33% annualized),
longs are paying shorts. We long spot (no funding) and short perp (collect funding).

When funding rate is highly negative (<-0.01%),
shorts are paying longs. We short spot and long perp (collect funding).

This is a delta-neutral strategy — we're hedged, profiting from the funding payments.
"""
import pandas as pd
from strategy.base import BaseStrategy, Signal


class FundingArb(BaseStrategy):
    name = "funding_arb"

    def __init__(
        self,
        high_rate_threshold: float = 0.0003,   # 0.03% per period (~33% annualized)
        low_rate_threshold: float = -0.0001,    # -0.01% per period (~-11% annualized)
        exit_rate_threshold: float = 0.0001,    # exit when funding normalizes
    ):
        self.high_threshold = high_rate_threshold
        self.low_threshold = low_rate_threshold
        self.exit_threshold = exit_rate_threshold

    def check_funding(self, funding_rate: float, price: float, atr: float) -> list[Signal]:
        """Check funding rate for arbitrage opportunity.

        Args:
            funding_rate: current funding rate per period (e.g. 0.0005 = 0.05%)
            price: current spot price
            atr: current ATR for stop calculation
        """
        signals = []

        if funding_rate > self.high_threshold:
            # High positive funding: longs paying shorts
            # Strategy: long spot + short perp = collect funding while hedged
            annualized = funding_rate * 3 * 365 * 100  # assuming 8h periods, 3x/day
            strength = min(1.0, funding_rate / (self.high_threshold * 3))
            signals.append(Signal(
                symbol="",
                direction="BUY",  # long spot side
                strength=strength,
                stop_price=price * 0.97,  # wider stop — this is hedged
                take_profit=price * 1.03,  # target is funding collection, not price move
                metadata={
                    "reason": f"Funding arb: rate={funding_rate*100:.4f}% (~{annualized:.0f}% ann)",
                    "funding_rate": funding_rate,
                    "annualized_pct": annualized,
                    "arb_type": "long_spot_short_perp",
                },
            ))

        elif funding_rate < self.low_threshold:
            # Deeply negative funding: shorts paying longs
            # Strategy: short spot + long perp = collect funding while hedged
            annualized = funding_rate * 3 * 365 * 100
            strength = min(1.0, abs(funding_rate) / abs(self.low_threshold * 3))
            signals.append(Signal(
                symbol="",
                direction="SELL",  # short spot side
                strength=strength,
                stop_price=price * 1.03,
                take_profit=price * 0.97,
                metadata={
                    "reason": f"Funding arb: rate={funding_rate*100:.4f}% (~{annualized:.0f}% ann)",
                    "funding_rate": funding_rate,
                    "annualized_pct": annualized,
                    "arb_type": "short_spot_long_perp",
                },
            ))

        return signals

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        """Not used directly — funding arb uses check_funding() instead."""
        return []
