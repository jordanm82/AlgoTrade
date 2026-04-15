"""Shared Kalshi feature builders for M0/M10 parity.

This module centralizes feature math so train/backtest/live use the same
construction path.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def get_avg_price_5m(
    first_5m: pd.DataFrame,
    second_5m: pd.DataFrame,
    target_time,
    field: str = "open",
) -> float | None:
    """Average the first 5m candle value from two exchanges at target_time."""
    prices = []
    for df_5m in [first_5m, second_5m]:
        if df_5m.empty:
            return None
        mask = (df_5m.index >= target_time) & (df_5m.index < target_time + timedelta(minutes=5))
        if mask.sum() <= 0:
            return None
        prices.append(float(df_5m[mask].iloc[0][field]))
    if len(prices) != 2:
        return None
    return sum(prices) / 2.0


def compute_confluence_features(
    *,
    alt_keys: list,
    ws_naive,
    get_15m_df,
    get_1h_df,
    get_anchor_price,
    get_strike,
) -> dict:
    """Compute confluence bundle for alternate assets.

    Callbacks:
    - get_15m_df(key) -> pd.DataFrame | None
    - get_1h_df(key) -> pd.DataFrame | None
    - get_anchor_price(key) -> float | None
    - get_strike(key) -> float | None
    """
    alt_rsi_15m, alt_rsi_1h, alt_momentum, alt_distances = [], [], [], []

    for key in alt_keys:
        alt_15m = get_15m_df(key)
        alt_15m_filt = None
        if alt_15m is not None:
            alt_15m_filt = alt_15m[alt_15m.index < ws_naive]
        if alt_15m_filt is not None and len(alt_15m_filt) >= 2:
            rsi_val = float(alt_15m_filt.iloc[-1].get("rsi", 50))
            alt_rsi_15m.append(rsi_val)
            alt_momentum.append(1 if rsi_val >= 50 else -1)

        alt_1h = get_1h_df(key)
        if alt_1h is not None:
            alt_1h_filt = alt_1h[alt_1h.index < ws_naive]
            if len(alt_1h_filt) >= 2:
                alt_rsi_1h.append(float(alt_1h_filt.iloc[-1].get("rsi", 50)))

        if alt_15m_filt is None or len(alt_15m_filt) < 2:
            continue
        alt_price = get_anchor_price(key)
        alt_strike = get_strike(key)
        if alt_price is None or alt_strike is None:
            continue
        alt_atr = float(alt_15m_filt.iloc[-1].get("atr", 0))
        if alt_atr > 0:
            alt_distances.append((float(alt_price) - float(alt_strike)) / alt_atr)

    return {
        "alt_rsi_avg": sum(alt_rsi_15m) / len(alt_rsi_15m) if alt_rsi_15m else 50,
        "alt_rsi_1h_avg": sum(alt_rsi_1h) / len(alt_rsi_1h) if alt_rsi_1h else 50,
        "alt_momentum_align": sum(alt_momentum) if alt_momentum else 0,
        "alt_distance_avg": sum(alt_distances) / len(alt_distances) if alt_distances else 0,
    }


def build_common_feature_vector(
    prev,
    df_1h,
    df_4h,
    ws_naive,
    distance,
    *,
    kalshi_extra=None,
    atr_pctile_val=0.5,
):
    """Build parity-safe feature vector from a point-in-time snapshot."""
    sma_val = float(prev.get("sma_20", 0))
    adx_val = float(prev.get("adx", 20))
    close_val = float(prev.get("close", 0))
    ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
    pve = float(prev.get("price_vs_ema", 0))
    hr = float(prev.get("hourly_return", 0))
    if pd.isna(pve) or np.isinf(pve):
        pve = 0
    if pd.isna(hr) or np.isinf(hr):
        hr = 0

    feat = {
        "macd_15m": float(prev.get("macd_hist", 0)),
        "norm_return": float(prev.get("norm_return", 0)) if pd.notna(prev.get("norm_return")) else 0,
        "ema_slope": float(prev.get("ema_slope", 0)) if pd.notna(prev.get("ema_slope")) else 0,
        "roc_5": float(prev.get("roc_5", 0)),
        "macd_1h": 0.0,
        "price_vs_ema": pve,
        "hourly_return": hr,
        "trend_direction": adx_val * ts_sign,
        "vol_ratio": float(prev.get("vol_ratio", 1)) if pd.notna(prev.get("vol_ratio")) else 1,
        "adx": adx_val,
        "rsi_1h": 50.0,
        "rsi_4h": 50.0,
        "distance_from_strike": distance,
    }
    if df_1h is not None:
        m1h = df_1h[df_1h.index < ws_naive]
        if len(m1h) >= 20:
            feat["rsi_1h"] = float(m1h.iloc[-1].get("rsi", 50))
            feat["macd_1h"] = float(m1h.iloc[-1].get("macd_hist", 0))
    if df_4h is not None:
        m4h = df_4h[df_4h.index < ws_naive]
        if len(m4h) >= 10:
            feat["rsi_4h"] = float(m4h.iloc[-1].get("rsi", 50))

    kx = kalshi_extra or {}
    feat["strike_delta"] = kx.get("strike_delta", 0.0)
    feat["strike_trend_3"] = kx.get("strike_trend_3", 0.0)
    hour = ws_naive.hour if hasattr(ws_naive, "hour") else 12
    feat["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    feat["rsi_alignment"] = (1 if feat["rsi_1h"] >= 50 else -1) * (1 if feat["rsi_4h"] >= 50 else -1)
    feat["atr_percentile"] = atr_pctile_val
    feat["rsi_15m"] = float(prev.get("rsi", 50))
    bb_upper = float(prev.get("bb_upper", 0))
    bb_lower = float(prev.get("bb_lower", 0))
    bb_mid = float(prev.get("sma_20", 0))
    feat["bbw"] = ((bb_upper - bb_lower) / bb_mid * 100) if bb_mid > 0 else 0

    feat["alt_rsi_avg"] = kx.get("alt_rsi_avg", 50)
    feat["alt_rsi_1h_avg"] = kx.get("alt_rsi_1h_avg", 50)
    feat["alt_momentum_align"] = kx.get("alt_momentum_align", 0)
    feat["alt_distance_avg"] = kx.get("alt_distance_avg", 0)

    feat["return_4h"] = kx.get("return_4h", 0)
    feat["return_12h"] = kx.get("return_12h", 0)
    feat["price_vs_sma_1h"] = kx.get("price_vs_sma_1h", 0)
    feat["lower_lows_4h"] = kx.get("lower_lows_4h", 0)
    feat["trend_strength"] = kx.get("trend_strength", 0)

    feat["pve_x_trend"] = feat["price_vs_ema"] * feat["trend_strength"]
    feat["pve_x_return12h"] = feat["price_vs_ema"] * feat["return_12h"]
    feat["slope_x_trend"] = feat.get("ema_slope", 0) * feat["trend_strength"]
    feat["slope_x_return12h"] = feat.get("ema_slope", 0) * feat["return_12h"]
    feat["rsi1h_x_r12h"] = (feat.get("rsi_1h", 50) - 50) * feat["return_12h"]
    feat["rsi4h_x_r12h"] = (feat.get("rsi_4h", 50) - 50) * feat["return_12h"]
    feat["rsi1h_x_r4h"] = (feat.get("rsi_1h", 50) - 50) * feat.get("return_4h", 0)
    feat["dist_x_r12h"] = feat.get("distance_from_strike", 0) * feat["return_12h"]

    if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
        return None
    return feat


def _compute_intra_from_averages(
    *,
    c1_open: float,
    c1_close: float,
    c1_high: float,
    c1_low: float,
    c1_vol: float,
    c2_close: float,
    c2_high: float,
    c2_low: float,
    c2_vol: float,
    atr: float,
) -> dict | None:
    if atr <= 0 or c1_vol <= 0:
        return None
    return {
        "price_move_atr": (c2_close - c1_open) / atr,
        "candle1_range_atr": (c1_high - c1_low) / atr,
        "candle2_range_atr": (c2_high - c2_low) / atr,
        "momentum_shift": (c2_close - c1_close) / atr,
        "volume_acceleration": c2_vol / c1_vol,
    }


def compute_m10_intra_from_exchange_dfs(
    cb_5m: pd.DataFrame,
    bs_5m: pd.DataFrame,
    ws_naive,
    atr: float,
) -> dict | None:
    """Compute strict M10 intra-window features from two 5m histories."""
    min5 = ws_naive + timedelta(minutes=5)
    min10 = ws_naive + timedelta(minutes=10)
    c1_rows, c2_rows = [], []
    for df_5m in [cb_5m, bs_5m]:
        if df_5m.empty:
            return None
        m1 = (df_5m.index >= ws_naive) & (df_5m.index < min5)
        m2 = (df_5m.index >= min5) & (df_5m.index < min10)
        if m1.sum() <= 0 or m2.sum() <= 0:
            return None
        c1_rows.append(df_5m[m1].iloc[0])
        c2_rows.append(df_5m[m2].iloc[0])
    if len(c1_rows) != 2 or len(c2_rows) != 2:
        return None

    return _compute_intra_from_averages(
        c1_open=sum(float(c["open"]) for c in c1_rows) / 2.0,
        c1_close=sum(float(c["close"]) for c in c1_rows) / 2.0,
        c1_high=sum(float(c["high"]) for c in c1_rows) / 2.0,
        c1_low=sum(float(c["low"]) for c in c1_rows) / 2.0,
        c1_vol=sum(float(c["volume"]) for c in c1_rows) / 2.0,
        c2_close=sum(float(c["close"]) for c in c2_rows) / 2.0,
        c2_high=sum(float(c["high"]) for c in c2_rows) / 2.0,
        c2_low=sum(float(c["low"]) for c in c2_rows) / 2.0,
        c2_vol=sum(float(c["volume"]) for c in c2_rows) / 2.0,
        atr=atr,
    )


def compute_m10_intra_from_window_candles(c1: dict, c2: dict, atr: float) -> dict | None:
    """Compute strict M10 intra-window features from averaged window candles."""
    try:
        return _compute_intra_from_averages(
            c1_open=float(c1["open"]),
            c1_close=float(c1["close"]),
            c1_high=float(c1["high"]),
            c1_low=float(c1["low"]),
            c1_vol=float(c1["volume"]),
            c2_close=float(c2["close"]),
            c2_high=float(c2["high"]),
            c2_low=float(c2["low"]),
            c2_vol=float(c2["volume"]),
            atr=float(atr),
        )
    except Exception:
        return None

