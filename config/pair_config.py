# config/pair_config.py
"""Per-pair strategy configuration — thresholds tuned from 6-month backtest results.
Pairs with better backtest WR get tighter thresholds (more signals).
Pairs with worse WR get wider thresholds (fewer, higher-quality signals)."""

PAIR_CONFIG = {
    # Top performers — tighter thresholds, 2x leverage
    "ATOM/USDT": {
        "rsi_oversold": 35,      # wider entry — ATOM backtested 87.7% WR
        "rsi_overbought": 65,
        "rsi_mr_oversold": 32,
        "rsi_mr_overbought": 68,
        "rsi_mr_exit_long": 60,
        "rsi_mr_exit_short": 40,
        "bb_rsi_buy": 38,        # wider BB entry — high WR supports it
        "bb_rsi_short": 62,
        "leverage": 2,
        "enabled_strategies": ["bb_grid", "rsi_mr"],
    },
    "FIL/USDT": {
        "rsi_oversold": 33,
        "rsi_overbought": 67,
        "rsi_mr_oversold": 30,
        "rsi_mr_overbought": 70,
        "rsi_mr_exit_long": 65,
        "rsi_mr_exit_short": 35,
        "bb_rsi_buy": 35,
        "bb_rsi_short": 65,
        "leverage": 2,
        "enabled_strategies": ["bb_grid", "rsi_mr"],
    },
    "DOT/USDT": {
        "rsi_oversold": 32,
        "rsi_overbought": 68,
        "rsi_mr_oversold": 30,
        "rsi_mr_overbought": 70,
        "rsi_mr_exit_long": 65,
        "rsi_mr_exit_short": 35,
        "bb_rsi_buy": 35,
        "bb_rsi_short": 65,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
    },
    # Good performers — standard thresholds, 1x leverage
    "UNI/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "LTC/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "SHIB/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    # New pairs — conservative thresholds until validated
    "BTC/USDT": {
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "bb_rsi_buy": 30,
        "bb_rsi_short": 70,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "ETH/USDT": {
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "bb_rsi_buy": 30,
        "bb_rsi_short": 70,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "SOL/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "XRP/USDT": {
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "bb_rsi_buy": 30,
        "bb_rsi_short": 70,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "DOGE/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "ADA/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "AVAX/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
    "LINK/USDT": {
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_rsi_buy": 33,
        "bb_rsi_short": 67,
        "leverage": 1,
        "enabled_strategies": ["bb_grid"],
    },
}

# Default config for pairs not in PAIR_CONFIG
DEFAULT_PAIR_CONFIG = {
    "rsi_oversold": 28,
    "rsi_overbought": 72,
    "bb_rsi_buy": 30,
    "bb_rsi_short": 70,
    "leverage": 1,
    "enabled_strategies": ["bb_grid"],
}

def get_pair_config(symbol: str) -> dict:
    """Get config for a pair, falling back to defaults."""
    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)

# All monitored pairs (expanded from 6 to 14)
ALL_PAIRS = list(PAIR_CONFIG.keys())

# Pairs eligible for funding rate arbitrage (need perps on Coinbase)
FUNDING_ARB_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Mapping to Coinbase symbols
COINBASE_MAP = {
    "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD", "SOL/USDT": "SOL-USD",
    "XRP/USDT": "XRP-USD", "DOGE/USDT": "DOGE-USD", "ADA/USDT": "ADA-USD",
    "AVAX/USDT": "AVAX-USD", "LINK/USDT": "LINK-USD", "DOT/USDT": "DOT-USD",
    "SHIB/USDT": "SHIB-USD", "LTC/USDT": "LTC-USD", "UNI/USDT": "UNI-USD",
    "ATOM/USDT": "ATOM-USD", "FIL/USDT": "FIL-USD",
}
