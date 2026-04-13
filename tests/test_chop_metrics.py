# tests/test_chop_metrics.py
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from cli.kalshi_daemon import KalshiDaemon


def _make_df(n: int = 30, atr: float = 100.0, bb_upper: float = 105.0,
             bb_lower: float = 95.0, sma_20: float = 100.0,
             atr_variable: bool = False) -> pd.DataFrame:
    """Build a synthetic indicator-ready dataframe with n rows."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    atr_col = (
        np.linspace(atr * 0.5, atr * 1.5, n) if atr_variable
        else np.full(n, atr)
    )
    return pd.DataFrame(
        {
            "atr": atr_col,
            "bb_upper": np.full(n, bb_upper),
            "bb_lower": np.full(n, bb_lower),
            "sma_20": np.full(n, sma_20),
            "close": np.full(n, sma_20),
        },
        index=idx,
    )


def _make_daemon(cache: dict) -> KalshiDaemon:
    """Bare daemon with only the cache populated. Avoids touching I/O."""
    d = KalshiDaemon.__new__(KalshiDaemon)
    d._kalshi_cached_dataframes = cache
    return d


def test_compute_chop_metrics_full_data_all_assets():
    """All four assets have both 15m and 1h cached — all 8 fields non-None."""
    cache = {}
    for pair in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"):
        cache[pair] = _make_df(n=30, atr_variable=True)
        cache[f"{pair}_1h"] = _make_df(n=30, atr_variable=True)

    d = _make_daemon(cache)
    out = d._compute_chop_metrics("BTC")

    assert set(out.keys()) == {
        "bbw_15m", "atr_pct_15m", "bbw_1h", "atr_pct_1h",
        "bbw_15m_mkt", "atr_pct_15m_mkt", "bbw_1h_mkt", "atr_pct_1h_mkt",
    }
    # Per-asset BBW: (105 - 95) / 100 * 100 = 10.0
    assert out["bbw_15m"] == pytest.approx(10.0)
    assert out["bbw_1h"] == pytest.approx(10.0)
    # ATR percentile: with variable ATR, final value is near the max => ≈1.0
    assert 0.0 <= out["atr_pct_15m"] <= 1.0
    assert 0.0 <= out["atr_pct_1h"] <= 1.0
    # Market-wide matches per-asset since all four are identical
    assert out["bbw_15m_mkt"] == pytest.approx(10.0)
    assert out["bbw_1h_mkt"] == pytest.approx(10.0)
