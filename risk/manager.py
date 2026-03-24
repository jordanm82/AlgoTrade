from config.settings import (
    MAX_POSITION_PCT, MAX_CONCURRENT, MAX_LEVERAGE,
    MAX_DAILY_DRAWDOWN_PCT, MIN_BALANCE_USD,
    ATR_MIN_MULTIPLIER, ATR_MAX_MULTIPLIER,
)


class RiskManager:
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self._daily_start: float | None = None

    def check_entry(
        self, size_usd: float, leverage: int, current_positions: int
    ) -> tuple[bool, str]:
        """Validate a trade entry against risk limits."""
        if self.portfolio_value < MIN_BALANCE_USD:
            return False, f"Below minimum balance (${MIN_BALANCE_USD})"

        max_size = self.portfolio_value * MAX_POSITION_PCT
        if size_usd > max_size:
            return False, f"Position size ${size_usd:.0f} exceeds max ${max_size:.0f} ({MAX_POSITION_PCT:.0%})"

        if current_positions >= MAX_CONCURRENT:
            return False, f"Max concurrent positions ({MAX_CONCURRENT}) reached"

        if leverage > MAX_LEVERAGE:
            return False, f"Leverage {leverage}x exceeds max {MAX_LEVERAGE}x"

        return True, "OK"

    def validate_stop(
        self, entry: float, stop: float, atr: float, side: str
    ) -> tuple[bool, str]:
        """Validate stop distance is within ATR bounds."""
        if atr <= 0:
            return False, "ATR must be positive"

        distance = abs(entry - stop)
        atr_multiple = distance / atr

        if side == "BUY" and stop >= entry:
            return False, "Buy stop must be below entry"
        if side == "SELL" and stop <= entry:
            return False, "Sell stop must be above entry"

        if atr_multiple < ATR_MIN_MULTIPLIER:
            return False, f"Stop too tight ({atr_multiple:.1f}x ATR, min {ATR_MIN_MULTIPLIER}x)"
        if atr_multiple > ATR_MAX_MULTIPLIER:
            return False, f"Stop too wide ({atr_multiple:.1f}x ATR, max {ATR_MAX_MULTIPLIER}x)"

        return True, f"OK ({atr_multiple:.1f}x ATR)"

    def record_daily_start(self, value: float):
        self._daily_start = value

    def is_halted(self, current_value: float) -> bool:
        """Check if daily drawdown limit is breached."""
        if self._daily_start is None:
            return False
        drawdown = (self._daily_start - current_value) / self._daily_start
        return drawdown > MAX_DAILY_DRAWDOWN_PCT

    def update_portfolio_value(self, value: float):
        self.portfolio_value = value
