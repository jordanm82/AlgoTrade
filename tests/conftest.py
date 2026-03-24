import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_ohlcv():
    """Generate realistic OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "timestamp": dates,
        "open": close + np.random.randn(n) * 50,
        "high": close + abs(np.random.randn(n) * 200),
        "low": close - abs(np.random.randn(n) * 200),
        "close": close,
        "volume": abs(np.random.randn(n) * 1000) + 500,
    }).set_index("timestamp")

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    (tmp_path / "store").mkdir()
    (tmp_path / "trades").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "snapshots").mkdir()
    return tmp_path
