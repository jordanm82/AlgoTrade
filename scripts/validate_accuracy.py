#!/usr/bin/env python3
"""Check model accuracy on recent data: are predictions actually correct?

For each completed 15m candle in the last 7 days, predicts YES/NO/SKIP
and checks against the actual next candle outcome (close >= open).

Usage:
    ./venv/bin/python scripts/validate_accuracy.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.strategies.kalshi_predictor_v3 import KalshiPredictorV3

FEATURE_NAMES = [
    "rsi_15m", "stochrsi_15m", "macd_15m", "norm_return",
    "vol_ratio", "bb_position", "ema_slope", "adx", "roc_5",
    "rsi_1h", "macd_1h", "rsi_4h",
    "price_vs_ema", "hourly_return", "trend_direction",
]

ASSETS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
}


def fetch_candles(fetcher, symbol, timeframe, days):
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    batch_size = 300
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms; time.sleep(0.3); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.3)
        except Exception as e:
            since += batch_size * candle_ms; time.sleep(1)
    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def extract_features(df, df_1h_ind, df_4h_ind, row_idx):
    """Extract features using training logic."""
    pct = df["close"].pct_change()
    df_feat = df.copy()
    df_feat["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
    df_feat["vol_ratio"] = df_feat["volume"] / df_feat["vol_sma_20"]
    bb_range = df_feat["bb_upper"] - df_feat["bb_lower"]
    df_feat["bb_position"] = (df_feat["close"] - df_feat["bb_lower"]) / bb_range.replace(0, np.nan)
    df_feat["ema_slope"] = df_feat["ema_12"].pct_change(3) * 100
    df_feat["price_vs_ema"] = (df_feat["close"] - df_feat["sma_20"]) / df_feat["atr"].replace(0, np.nan)
    df_feat["hourly_return"] = df_feat["close"].pct_change(4) * 100

    i = row_idx
    t = df_feat.index[i]
    r = df_feat.iloc[i]

    vals = [
        float(r.get("rsi", np.nan)),
        float(r.get("stochrsi_k", np.nan)),
        float(r.get("macd_hist", np.nan)),
        float(r.get("norm_return", np.nan)),
        float(r.get("vol_ratio", np.nan)),
        float(r.get("bb_position", np.nan)),
        float(r.get("ema_slope", np.nan)),
        float(r.get("adx", np.nan)),
        float(r.get("roc_5", np.nan)),
    ]
    if any(pd.isna(v) or np.isinf(v) for v in vals):
        return None

    if df_1h_ind is not None:
        mask = df_1h_ind.index <= t
        if mask.sum() >= 20:
            r1h = df_1h_ind.loc[mask].iloc[-1]
            vals.append(float(r1h.get("rsi", 50)))
            vals.append(float(r1h.get("macd_hist", 0)))
        else:
            return None
    else:
        vals.extend([50.0, 0.0])

    if df_4h_ind is not None:
        mask = df_4h_ind.index <= t
        if mask.sum() >= 10:
            vals.append(float(df_4h_ind.loc[mask].iloc[-1].get("rsi", 50)))
        else:
            return None
    else:
        vals.append(50.0)

    pve = float(r.get("price_vs_ema", np.nan))
    hr = float(r.get("hourly_return", np.nan))
    adx_val = float(r.get("adx", 20))
    close_val = float(r.get("close", 0))
    sma_val = float(r.get("sma_20", 0))

    if pd.isna(pve) or np.isinf(pve):
        pve = 0
    if pd.isna(hr) or np.isinf(hr):
        hr = 0

    if sma_val > 0:
        trend_sign = 1 if close_val >= sma_val else -1
    else:
        trend_sign = 0

    vals.append(pve)
    vals.append(hr)
    vals.append(adx_val * trend_sign)

    if any(pd.isna(v) or np.isinf(v) for v in vals):
        return None
    return vals


def main():
    print("=" * 70)
    print("RECENT ACCURACY CHECK — Last 30 days")
    print("Does the model predict correctly on RECENT completed candles?")
    print("=" * 70)

    fetcher = DataFetcher()
    predictor = KalshiPredictorV3()

    if predictor._knn is None:
        print("ERROR: No model loaded!")
        return

    scaler = predictor._knn_scaler
    model = predictor._knn

    # Track overall stats
    all_wins, all_losses = 0, 0
    yes_wins, yes_losses = 0, 0
    no_wins, no_losses = 0, 0
    skip_up, skip_down = 0, 0

    for asset_name, symbol in ASSETS.items():
        print(f"\n{'─' * 70}")
        print(f"  {asset_name}")
        print(f"{'─' * 70}")

        # Fetch enough data for indicators (30 days) but only test last 7
        df_15m = fetch_candles(fetcher, symbol, "15m", 30)
        df_1h = fetch_candles(fetcher, symbol, "1h", 30)
        df_4h = fetch_candles(fetcher, symbol, "4h", 30)

        if df_15m.empty:
            print(f"  No data!")
            continue

        df_ind = add_indicators(df_15m.copy())
        df_1h_ind = add_indicators(df_1h.copy()) if df_1h is not None else None
        df_4h_ind = add_indicators(df_4h.copy()) if df_4h is not None else None

        # Last 30 days = ~2880 15m candles
        candles_7d = min(2880, len(df_ind) - 220)

        a_wins, a_losses, a_yes_w, a_yes_l, a_no_w, a_no_l, a_skip = 0, 0, 0, 0, 0, 0, 0

        for offset in range(candles_7d, 1, -1):  # need offset >= 2 for next candle outcome
            row_idx = len(df_ind) - offset
            if row_idx < 210 or row_idx >= len(df_ind) - 1:
                continue

            feats = extract_features(df_ind, df_1h_ind, df_4h_ind, row_idx)
            if feats is None:
                continue

            X = np.array(feats).reshape(1, -1)
            X_scaled = scaler.transform(X)
            prob = float(model.predict_proba(X_scaled)[0][1])

            # Actual outcome: did NEXT candle close >= its open?
            next_candle = df_ind.iloc[row_idx + 1]
            actual_up = float(next_candle["close"]) >= float(next_candle["open"])

            if prob >= 0.55:
                # Predicted YES (UP)
                if actual_up:
                    a_yes_w += 1
                else:
                    a_yes_l += 1
            elif prob <= 0.45:
                # Predicted NO (DOWN)
                if not actual_up:
                    a_no_w += 1
                else:
                    a_no_l += 1
            else:
                a_skip += 1
                if actual_up:
                    skip_up += 1
                else:
                    skip_down += 1

        a_wins = a_yes_w + a_no_w
        a_losses = a_yes_l + a_no_l
        total_bets = a_wins + a_losses
        wr = a_wins / total_bets * 100 if total_bets > 0 else 0
        yes_total = a_yes_w + a_yes_l
        no_total = a_no_w + a_no_l
        yes_wr = a_yes_w / yes_total * 100 if yes_total > 0 else 0
        no_wr = a_no_w / no_total * 100 if no_total > 0 else 0

        # Base rate: what % of candles were UP?
        test_candles = df_ind.iloc[len(df_ind) - candles_7d:len(df_ind) - 1]
        base_up = (test_candles["close"] >= test_candles["open"]).mean() * 100

        print(f"  Base rate UP: {base_up:.1f}%")
        print(f"  Bets: {total_bets} ({yes_total}Y/{no_total}N/{a_skip}SKIP)")
        print(f"  Overall WR: {wr:.1f}%")
        print(f"  YES WR: {yes_wr:.1f}% ({a_yes_w}W/{a_yes_l}L)")
        print(f"  NO  WR: {no_wr:.1f}% ({a_no_w}W/{a_no_l}L)")

        # Simulated P&L at 50c contracts
        pnl = a_wins * 0.50 - a_losses * 0.50
        print(f"  Sim P&L (50c): ${pnl:+.2f}")

        all_wins += a_wins
        all_losses += a_losses
        yes_wins += a_yes_w
        yes_losses += a_yes_l
        no_wins += a_no_w
        no_losses += a_no_l

    # Grand total
    grand_total = all_wins + all_losses
    grand_wr = all_wins / grand_total * 100 if grand_total > 0 else 0
    yes_total = yes_wins + yes_losses
    no_total = no_wins + no_losses
    yes_wr = yes_wins / yes_total * 100 if yes_total > 0 else 0
    no_wr = no_wins / no_total * 100 if no_total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"GRAND TOTAL (all assets, 7 days)")
    print(f"{'=' * 70}")
    print(f"  Total bets: {grand_total} ({yes_total}Y / {no_total}N)")
    print(f"  Overall WR: {grand_wr:.1f}% ({all_wins}W / {all_losses}L)")
    print(f"  YES WR: {yes_wr:.1f}% ({yes_wins}W / {yes_losses}L)")
    print(f"  NO  WR: {no_wr:.1f}% ({no_wins}W / {no_losses}L)")
    grand_pnl = all_wins * 0.50 - all_losses * 0.50
    print(f"  Sim P&L (50c): ${grand_pnl:+.2f}")

    print(f"\n  Walk-forward backtest reference: 74.2% WR")
    gap = grand_wr - 74.2
    if abs(gap) < 5:
        print(f"  ✓ Within 5pp of backtest — model generalizes!")
    elif gap < -5:
        print(f"  ⚠ {abs(gap):.1f}pp below backtest — potential regime shift or remaining parity issue")
    else:
        print(f"  ↑ {gap:.1f}pp above backtest — possible overfit on recent data")


if __name__ == "__main__":
    main()
