# config/production.py
"""Production strategy configuration — validated via 6-month backtest."""

# Pairs to monitor on 15m timeframe (our validated winners)
MONITORED_PAIRS_15M = [
    "ATOM/USDT", "FIL/USDT", "DOT/USDT",
    "UNI/USDT", "LTC/USDT", "SHIB/USDT",
]

# Pairs for 2x leverage BB Grid (top performers only)
LEVERAGE_PAIRS = ["ATOM/USDT", "FIL/USDT", "DOT/USDT"]

# Coinbase mapping for execution
PAIR_TO_COINBASE = {
    "ATOM/USDT": "ATOM-USD", "FIL/USDT": "FIL-USD", "DOT/USDT": "DOT-USD",
    "UNI/USDT": "UNI-USD", "LTC/USDT": "LTC-USD", "SHIB/USDT": "SHIB-USD",
}

# Strategy parameters (from backtest validation)
BB_GRID_CONFIG = {
    "rsi_buy_threshold": 35,     # buy when close < BB lower AND RSI < 35
    "rsi_short_threshold": 65,   # short when close > BB upper AND RSI > 65
    "exit_at_bb_mid": True,      # exit when price returns to BB middle
}

RSI_MR_CONFIG = {
    "oversold": 30,
    "overbought": 70,
    "exit_long": 65,       # close long when RSI > 65
    "exit_short": 35,      # close short when RSI < 35
}

# Risk settings
POSITION_SIZE_PCT = 0.10   # 10% of equity per trade, compounding
MAX_LEVERAGE = 2           # 2x max on validated pairs, 1x on others
MAX_CONCURRENT_POSITIONS = 6  # up from 3 — we're trading 6 pairs
STOP_LOSS_PCT = 0.03       # 3% hard stop
