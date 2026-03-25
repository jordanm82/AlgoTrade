# data/indicators.py
import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame. Returns a new DataFrame."""
    out = df.copy()

    # Moving averages
    out["sma_20"] = ta.sma(out["close"], length=20)
    out["sma_50"] = ta.sma(out["close"], length=50)
    out["ema_12"] = ta.ema(out["close"], length=12)
    out["ema_26"] = ta.ema(out["close"], length=26)

    # RSI
    out["rsi"] = ta.rsi(out["close"], length=14)

    # MACD
    macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
    out["macd"] = macd.iloc[:, 0]
    out["macd_hist"] = macd.iloc[:, 1]
    out["macd_signal"] = macd.iloc[:, 2]

    # Bollinger Bands
    bbands = ta.bbands(out["close"], length=20, std=2)
    out["bb_lower"] = bbands.iloc[:, 0]
    out["bb_middle"] = bbands.iloc[:, 1]
    out["bb_upper"] = bbands.iloc[:, 2]

    # ATR
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=14)

    # Volume SMA
    out["vol_sma_20"] = ta.sma(out["volume"], length=20)

    # Stochastic RSI
    stochrsi = ta.stochrsi(out["close"], length=14, rsi_length=14, k=3, d=3)
    out["stochrsi_k"] = stochrsi.iloc[:, 0]
    out["stochrsi_d"] = stochrsi.iloc[:, 1]

    # Rate of Change (5-period)
    out["roc_5"] = ta.roc(out["close"], length=5)

    return out
