import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "store"
TRADES_DIR = PROJECT_ROOT / "data" / "trades"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"

# Coinbase
CDP_KEY_FILE = PROJECT_ROOT / "cdp_api_key.json"

# Risk limits (hard guardrails)
MAX_POSITION_PCT = 0.10
MAX_CONCURRENT = 6
MAX_LEVERAGE = 3
MAX_DAILY_DRAWDOWN_PCT = 0.05
MIN_BALANCE_USD = 100.0

# Stop-loss
DEFAULT_ATR_MULTIPLIER = 2.0
ATR_MIN_MULTIPLIER = 1.0
ATR_MAX_MULTIPLIER = 3.0
ATR_PERIOD = 14

# Data defaults
DEFAULT_TIMEFRAME = "1h"
DEFAULT_CANDLE_LIMIT = 500

# Daemon intervals (seconds)
TICK_INTERVAL = 60
SIGNAL_INTERVAL = 300
SNAPSHOT_INTERVAL = 300
HOURLY_INTERVAL = 3600
DAILY_INTERVAL = 86400
