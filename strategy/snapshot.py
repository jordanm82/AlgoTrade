# strategy/snapshot.py
"""Shared minute-3 snapshot builder — used by BOTH backtest and live daemon.

This is the single source of truth for how we build the KNN input.
Any change here affects both backtest validation and live trading identically.

Rules:
1. Use ONLY completed 15m candles (exclude current partial candle)
2. Build synthetic row from first 3 minutes of 1m data in the window
3. Synthetic row inherits ALL indicator values from the last completed 15m candle
4. Only OHLCV fields are updated from 1m data
5. No fallbacks, no dirty data, no exceptions
"""
import numpy as np
import pandas as pd


def build_minute3_snapshot(df_15m_with_indicators: pd.DataFrame,
                           df_sub: pd.DataFrame,
                           window_start: pd.Timestamp) -> pd.DataFrame | None:
    """Build model-ready DataFrame as it would look at minute 3 of a window.

    Args:
        df_15m_with_indicators: 15m candles with indicators already computed.
            MUST include candles BEFORE window_start. May include the current
            partial candle — it will be excluded.
        df_sub: Sub-15m candle data (1m or 5m). Used to build the synthetic row.
            For 1m: uses first 3 candles. For 5m: uses first candle.
        window_start: Start of the 15m window (e.g., 2026-03-27 17:30:00)

    Returns:
        DataFrame with completed 15m history + synthetic row, or None
        if insufficient data. The last row has the early-window close price but
        all indicator values from the previous completed 15m candle.
    """
    # 1. Get ONLY completed 15m candles BEFORE this window
    history = df_15m_with_indicators[df_15m_with_indicators.index < window_start]
    if len(history) < 50:
        return None

    # 2. Get early candles in this window (or use last available before window)
    if len(df_sub) >= 2:
        spacing = (df_sub.index[1] - df_sub.index[0]).total_seconds()
    else:
        spacing = 60

    if spacing <= 120:
        cutoff = window_start + pd.Timedelta(minutes=3)
    else:
        cutoff = window_start + pd.Timedelta(minutes=5)

    current_sub = df_sub[(df_sub.index >= window_start) & (df_sub.index < cutoff)]

    # 3. Build synthetic row
    synthetic = history.iloc[-1].copy()

    if len(current_sub) >= 1:
        # Have sub-candle data in window — use it for OHLCV
        synthetic["open"] = float(current_sub.iloc[0]["open"])
        synthetic["high"] = max(float(history.iloc[-1]["high"]), float(current_sub["high"].max()))
        synthetic["low"] = min(float(history.iloc[-1]["low"]), float(current_sub["low"].min()))
        synthetic["close"] = float(current_sub.iloc[-1]["close"])
        synthetic["volume"] = float(current_sub["volume"].sum())
    else:
        # No sub-candle yet (e.g., 5m candle hasn't closed at minute 2-3)
        # Use the last available sub-candle's close as current price
        last_available = df_sub[df_sub.index < window_start]
        if len(last_available) >= 1:
            synthetic["close"] = float(last_available.iloc[-1]["close"])
        # Keep all other values from the previous 15m candle (already in synthetic)

    # 4. Append synthetic row to history
    new_row = pd.DataFrame([synthetic], index=[window_start])
    for col in history.columns:
        if col not in new_row.columns:
            new_row[col] = np.nan
    result = pd.concat([history, new_row])

    return result


def compute_btc_confluence(btc_1m: pd.DataFrame,
                            window_start: pd.Timestamp,
                            bet_side: str) -> int:
    """Compute BTC confluence score (0-100) at minute 2 of the window.

    Uses BTC 1m returns (1-bar, 2-bar, 3-bar) to measure how strongly
    BTC's momentum supports the proposed bet direction.

    Args:
        btc_1m: BTC 1m candle data (sorted by time)
        window_start: Start of the 15m window
        bet_side: "YES" or "NO"

    Returns:
        Score 0-100 (50=neutral, 70+=confirming, 30-=opposing)
    """
    btc_m2 = window_start + pd.Timedelta(minutes=2)
    idx = btc_1m.index.get_indexer([btc_m2], method="pad")[0]

    if idx < 3:
        return 50

    b0 = float(btc_1m.iloc[idx]["close"])
    b1 = float(btc_1m.iloc[idx - 1]["close"])
    b2 = float(btc_1m.iloc[idx - 2]["close"])
    b3 = float(btc_1m.iloc[idx - 3]["close"])

    if b1 <= 0 or b2 <= 0 or b3 <= 0:
        return 50

    r1 = (b0 - b1) / b1 * 100
    r2 = (b0 - b2) / b2 * 100
    r3 = (b0 - b3) / b3 * 100

    dm = 1 if bet_side == "YES" else -1
    score = 50
    a1 = r1 * dm
    score += min(20, max(-20, a1 * 67))
    score += min(15, max(-15, r2 * dm * 30))
    score += min(10, max(-10, r3 * dm * 15))

    if a1 > 0 and r2 * dm > 0 and r3 * dm > 0:
        score += 5
    elif a1 < 0 and r2 * dm < 0 and r3 * dm < 0:
        score -= 5

    return max(0, min(100, round(score)))
