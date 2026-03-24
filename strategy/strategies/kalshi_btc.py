# strategy/strategies/kalshi_btc.py
"""Kalshi BTC 15-minute prediction market strategy.

Uses our RSI/BB analysis to bet on BTC's short-term direction via Kalshi contracts.

Logic:
- When BTC RSI < 30 (oversold): bet YES on "BTC above $X" (price will bounce)
- When BTC RSI > 70 (overbought): bet NO on "BTC above $X" (price will drop)
- When BB Grid signals: use those for additional confluence

The advantage over spot trading:
- Fixed risk: you can only lose what you pay for the contract
- High leverage: buying YES at $0.30 = 233% return if right
- Short duration: 15-minute contracts settle automatically
- No stop-loss needed: max loss is the contract cost
"""
import pandas as pd
from strategy.base import BaseStrategy, Signal


class KalshiBTCStrategy(BaseStrategy):
    name = "kalshi_btc"

    def __init__(self, rsi_oversold: int = 30, rsi_overbought: int = 70,
                 min_edge_cents: int = 10):
        """
        min_edge_cents: minimum expected edge in cents to place a bet.
        If we think fair value is 80c but market is at 65c, edge is 15c.
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_edge = min_edge_cents

    def evaluate(self, df: pd.DataFrame, market: dict) -> dict | None:
        """Evaluate whether to bet on a specific Kalshi BTC market.

        Args:
            df: BTC OHLCV with indicators (from our data pipeline)
            market: Kalshi market dict with yes_bid, yes_ask, etc.

        Returns:
            dict with bet recommendation or None
        """
        if len(df) < 20 or "rsi" not in df.columns:
            return None

        last = df.iloc[-1]
        rsi = last.get("rsi")
        bb_lower = last.get("bb_lower")
        bb_upper = last.get("bb_upper")
        bb_mid = last.get("bb_middle")
        close = float(last["close"])

        if pd.isna(rsi):
            return None

        yes_ask = market.get("yes_ask", 50)
        no_ask = market.get("no_ask", 50)
        strike = market.get("floor_strike", 0)
        mins_to_exp = market.get("_mins_to_expiry", 0)

        if mins_to_exp < 5:
            return None  # too close to expiry

        # Estimate probability based on our indicators
        if rsi < self.rsi_oversold:
            # BTC likely to bounce up
            # If current price is already above strike, high probability of YES
            if close > strike:
                est_prob = min(95, 70 + (self.rsi_oversold - rsi))  # RSI 20 -> 80% prob
                if est_prob - yes_ask >= self.min_edge:
                    return {
                        "side": "yes",
                        "ticker": market["ticker"],
                        "est_probability": est_prob,
                        "market_price": yes_ask,
                        "edge_cents": est_prob - yes_ask,
                        "reason": f"RSI oversold ({rsi:.1f}), price ${close:,.0f} > strike ${strike:,.0f}",
                        "rsi": float(rsi),
                    }

        elif rsi > self.rsi_overbought:
            # BTC likely to drop
            # If current price is near/below strike, bet NO
            if close <= strike * 1.005:  # within 0.5% of strike
                est_prob = min(95, 70 + (rsi - self.rsi_overbought))  # RSI 80 -> 80% prob
                if est_prob - no_ask >= self.min_edge:
                    return {
                        "side": "no",
                        "ticker": market["ticker"],
                        "est_probability": est_prob,
                        "market_price": no_ask,
                        "edge_cents": est_prob - no_ask,
                        "reason": f"RSI overbought ({rsi:.1f}), price ${close:,.0f} near strike ${strike:,.0f}",
                        "rsi": float(rsi),
                    }

        # BB confluence check
        if pd.notna(bb_lower) and close < float(bb_lower) and rsi < 40:
            if close > strike:
                est_prob = min(90, 65 + (40 - rsi))
                if est_prob - yes_ask >= self.min_edge:
                    return {
                        "side": "yes",
                        "ticker": market["ticker"],
                        "est_probability": est_prob,
                        "market_price": yes_ask,
                        "edge_cents": est_prob - yes_ask,
                        "reason": f"BB+RSI confluence ({rsi:.1f}), bouncing from BB lower",
                        "rsi": float(rsi),
                    }

        return None

    def signals(self, df: pd.DataFrame) -> list[Signal]:
        """Standard interface -- not used for Kalshi, use evaluate() instead."""
        return []
