"""Per-pair strategy configuration — optimized via 6-month parameter sweep."""

PAIR_CONFIG = {
    "BTC/USDT": {
        "bb_rsi_buy": 28, "bb_rsi_short": 70,
        "rsi_mr_oversold": 28, "rsi_mr_overbought": 75,
        "rsi_mr_exit_long": 70, "rsi_mr_exit_short": 33,
        "leverage": 2,
        "enabled_strategies": ["rsi_mr"],
        # 6mo: wr=62.5% ret=0.7% n=56 trades
    },
    "ETH/USDT": {
        "bb_rsi_buy": 30, "bb_rsi_short": 65,
        "rsi_mr_oversold": 30, "rsi_mr_overbought": 70,
        "rsi_mr_exit_long": 65, "rsi_mr_exit_short": 35,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=65.2% ret=1.1% n=296 trades
    },
    "SOL/USDT": {
        "bb_rsi_buy": 32, "bb_rsi_short": 68,
        "rsi_mr_oversold": 32, "rsi_mr_overbought": 73,
        "rsi_mr_exit_long": 68, "rsi_mr_exit_short": 37,
        "leverage": 2,
        "enabled_strategies": ["rsi_mr"],
        # 6mo: wr=61.9% ret=2.4% n=118 trades
    },
    # XRP: DISABLED — wr=62.8% but ret=-3.7% (losses bigger than wins)
    # DOGE: DISABLED — wr=64.1% but ret=-6.8% (losses bigger than wins)
    # ADA: DISABLED — wr=69.4% but ret=-2.7% (marginally negative)
    "AVAX/USDT": {
        "bb_rsi_buy": 28, "bb_rsi_short": 65,
        "rsi_mr_oversold": 28, "rsi_mr_overbought": 70,
        "rsi_mr_exit_long": 65, "rsi_mr_exit_short": 33,
        "leverage": 2,
        "enabled_strategies": ["rsi_mr"],
        # 6mo: wr=66.9% ret=1.9% n=236 trades
    },
    # LINK: DISABLED — wr=69.3% but ret=-0.9% (marginally negative)
    "DOT/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=76.2% ret=100.8% n=564 trades
    },
    "SHIB/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=74.2% ret=54.0% n=563 trades
    },
    "LTC/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=75.9% ret=81.5% n=489 trades
    },
    "UNI/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=67.6% ret=75.8% n=518 trades
    },
    "ATOM/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=88.2% ret=592.1% n=551 trades
    },
    "FIL/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo: wr=71.5% ret=325.1% n=492 trades
    },
}

DEFAULT_PAIR_CONFIG = {
    "bb_rsi_buy": 30, "bb_rsi_short": 70,
    "rsi_mr_oversold": 28, "rsi_mr_overbought": 72,
    "rsi_mr_exit_long": 65, "rsi_mr_exit_short": 35,
    "leverage": 1,
    "enabled_strategies": ["bb_grid"],
}

def get_pair_config(symbol: str) -> dict:
    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)

ALL_PAIRS = list(PAIR_CONFIG.keys())
FUNDING_ARB_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

COINBASE_MAP = {sym: sym.replace("/USDT", "-USD") for sym in PAIR_CONFIG.keys()}
