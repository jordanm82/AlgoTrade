import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "store"
TRADES_DIR = PROJECT_ROOT / "data" / "trades"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"

# Coinbase
CDP_KEY_FILE = PROJECT_ROOT / "cdp_api_key.json"

# Kalshi (production)
KALSHI_KEY_FILE = PROJECT_ROOT / "KalshiPrimaryKey.txt"
KALSHI_API_KEY_ID = "719d39fa-9119-47e9-98a9-da7af90fb24b"

# Kalshi (demo)
KALSHI_DEMO_KEY_FILE = PROJECT_ROOT / "KalshiDemoKeys.txt"

# Risk limits (hard guardrails)
MAX_POSITION_PCT = 0.10
MAX_CONCURRENT = 3
MAX_LEVERAGE = 3
MAX_DAILY_DRAWDOWN_PCT = 0.05
MIN_BALANCE_USD = 10.0  # lowered from $100 — starting capital is $100, need room to trade

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
