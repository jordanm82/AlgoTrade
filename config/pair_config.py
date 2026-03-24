"""Per-pair strategy configuration — optimized via 6-month parameter sweep.
Tiered leverage: 3x on high-confidence (WR>=75%, DD>-15%, PF>=2.5), 2x on standard."""

PAIR_CONFIG = {
    # === TIER 1: 3x LEVERAGE (high confidence) ===
    "ATOM/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 3,
        "enabled_strategies": ["bb_grid"],
        # 6mo 3x: wr=88.4% ret=1811.9% n=551 dd=-8.5% pf=16.96
    },
    "DOT/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 3,
        "enabled_strategies": ["bb_grid"],
        # 6mo 3x: wr=76.4% ret=199.7% n=564 dd=-10.6% pf=2.70
    },
    "LTC/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 3,
        "enabled_strategies": ["bb_grid"],
        # 6mo 3x: wr=76.1% ret=155.6% n=489 dd=-9.4% pf=3.09
    },

    # === TIER 2: 2x LEVERAGE (standard confidence) ===
    "FIL/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo 2x: wr=71.5% ret=325.1% n=492 dd=-17.9% pf=3.64
    },
    "UNI/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo 2x: wr=67.6% ret=75.8% n=518 dd=-12.0% pf=2.05
    },
    "SHIB/USDT": {
        "bb_rsi_buy": 38, "bb_rsi_short": 62,
        "rsi_mr_oversold": 38, "rsi_mr_overbought": 67,
        "rsi_mr_exit_long": 62, "rsi_mr_exit_short": 43,
        "leverage": 2,
        "enabled_strategies": ["bb_grid"],
        # 6mo 2x: wr=74.4% ret=54.5% n=563 dd=-5.8% pf=2.13
    },
    "SOL/USDT": {
        "bb_rsi_buy": 32, "bb_rsi_short": 68,
        "rsi_mr_oversold": 32, "rsi_mr_overbought": 73,
        "rsi_mr_exit_long": 68, "rsi_mr_exit_short": 37,
        "leverage": 2,
        "enabled_strategies": ["rsi_mr"],
        # 6mo 2x: wr=67.6% ret=2.2% n=108 dd=-12.0% pf=1.11
    },
    "AVAX/USDT": {
        "bb_rsi_buy": 28, "bb_rsi_short": 65,
        "rsi_mr_oversold": 28, "rsi_mr_overbought": 70,
        "rsi_mr_exit_long": 65, "rsi_mr_exit_short": 33,
        "leverage": 2,
        "enabled_strategies": ["rsi_mr"],
        # 6mo 2x: wr=69.1% ret=6.2% n=217 dd=-6.9% pf=1.17
    },

    # === DISABLED (net negative at 3x) ===
    # BTC/USDT: wr=58.1% ret=-1.1% at 3x — not profitable
    # ETH/USDT: wr=68.0% ret=-4.3% at 3x — not profitable
    # XRP/USDT: wr=62.8% ret=-3.7% — losses bigger than wins
    # DOGE/USDT: wr=64.1% ret=-6.8% — losses bigger than wins
    # ADA/USDT: wr=69.4% ret=-2.7% — marginally negative
    # LINK/USDT: wr=69.3% ret=-0.9% — marginally negative
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
