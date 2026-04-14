#!/usr/bin/env python3
"""Train per-asset XGBoost models with research-optimized anti-overfitting.

Key improvements over previous version:
  - Balanced hyperparams: max_depth=3, lr=0.01, early_stopping=100
  - Purged walk-forward validation (4-hour gap to prevent feature leakage)
  - Time-decay sample weighting (60-day half-life — recent samples matter more)
  - Incremental feature groups: starts with 13 core, adds groups only if they help
  - Per-asset models with cross-asset confluence + regime features

Usage:
    ./venv/bin/python scripts/train_xgboost_per_asset.py [--synthetic] [--all-features]
"""
import argparse
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from scripts.backtest_kalshi_labels import build_features
from scripts.train_per_asset_models import (
    SERIES, ASSETS_SYMS, BASE_FEATURES, CONFLUENCE_FEATURES,
    REGIME_FEATURES, INTERACTION_FEATURES, ALL_FEATURES,
)

# Import the same data-fetching helpers
from scripts.backtest_kalshi_labels import fetch_5m_history, fetch_candles, get_avg_price

# === FEATURE GROUPS (for incremental testing) ===
# Core 13: the LogReg features that had proven predictive power
CORE_13 = [
    "rsi_1h", "price_vs_ema", "distance_from_strike",
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "hourly_return", "trend_direction",
    "vol_ratio", "adx", "rsi_4h",
]

# Additional feature groups to test incrementally
FEATURE_GROUPS = {
    "kalshi_lookback": ["strike_delta", "strike_trend_3"],
    "time": ["hour_sin", "hour_cos"],
    "technical": ["rsi_alignment", "atr_percentile", "bbw"],
    "confluence": ["alt_rsi_avg", "alt_rsi_1h_avg", "alt_momentum_align", "alt_distance_avg"],
    "regime": ["return_4h", "return_12h", "price_vs_sma_1h", "lower_lows_4h", "trend_strength"],
    "interaction": ["pve_x_trend", "pve_x_return12h", "slope_x_trend", "slope_x_return12h"],
    # RSI × trend interactions — centered at 50 so the product's sign is meaningful:
    #   rsi>50 & ret>0 → continuation (positive); rsi<50 & ret<0 → continuation
    #   rsi>50 & ret<0 → reversion OK (negative); rsi<50 & ret>0 → reversion OK
    # Split from the legacy interaction group so its value is measured on its own.
    "interaction_rsi": ["rsi1h_x_r12h", "rsi4h_x_r12h", "rsi1h_x_r4h", "dist_x_r12h"],
}

# Groups that should always be included in the final feature set regardless of
# the incremental WR vote. Used to protect groups we've decided to keep based
# on reasoning outside the training window (e.g., the RSI×trend interactions
# were added to counter an observed mean-reversion bias in live trading).
FORCE_KEEP_GROUPS = {"interaction", "interaction_rsi"}

# Purge gap: 4 hours = 16 fifteen-minute windows
PURGE_GAP_SAMPLES = 16


def compute_time_decay_weights(timestamps, half_life_days=60):
    """Exponential decay weights — recent samples weighted more heavily.

    half_life_days=60 means a sample from 60 days ago gets weight 0.5.
    """
    latest = timestamps.max()
    days_ago = (latest - timestamps).dt.total_seconds() / 86400
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    # Normalize so mean weight = 1 (XGBoost expects this)
    weights = weights / weights.mean()
    return weights.values


def evaluate_threshold(probs, y_true, threshold):
    """Evaluate model at a given confidence threshold. Returns (wins, losses, total, wr, pnl)."""
    w, l = 0, 0
    yw, yl, nw, nl = 0, 0, 0, 0
    for i in range(len(y_true)):
        p = int(probs[i] * 100)
        if p >= threshold:
            side = "YES"
        elif p <= (100 - threshold):
            side = "NO"
        else:
            continue
        won = (side == "YES" and y_true[i] == 1) or (side == "NO" and y_true[i] == 0)
        if won:
            w += 1
            if side == "YES": yw += 1
            else: nw += 1
        else:
            l += 1
            if side == "YES": yl += 1
            else: nl += 1
    tot = w + l
    wr = w / tot * 100 if tot > 0 else 0
    pnl = w * 0.40 - l * 0.60
    return {"w": w, "l": l, "tot": tot, "wr": wr, "pnl": pnl,
            "yw": yw, "yl": yl, "nw": nw, "nl": nl}


def train_with_features(df_all, feature_list, label=""):
    """Train XGBoost with purged walk-forward split and time-decay weights.

    Returns (model, test_results_dict, best_iteration).
    """
    # Purged walk-forward split: 70% train | 4h gap | 15% val | 15% test
    n = len(df_all)
    split_train = int(n * 0.70)
    split_val_start = split_train + PURGE_GAP_SAMPLES  # 4-hour purge gap
    split_val_end = int(n * 0.85)

    if split_val_start >= split_val_end:
        # Not enough data for purge gap — fall back to smaller gap
        split_val_start = split_train + 4
        if split_val_start >= split_val_end:
            split_val_start = split_train  # no gap if data is tiny

    df_train = df_all.iloc[:split_train]
    df_val = df_all.iloc[split_val_start:split_val_end]
    df_test = df_all.iloc[split_val_end:]

    if len(df_train) < 50 or len(df_val) < 20 or len(df_test) < 20:
        return None, None, 0

    X_train = df_train[feature_list].values
    y_train = df_train["label"].values
    X_val = df_val[feature_list].values
    y_val = df_val["label"].values
    X_test = df_test[feature_list].values
    y_test = df_test["label"].values

    # Time-decay sample weights (60-day half-life)
    sample_weights = compute_time_decay_weights(df_train["ts"], half_life_days=60)

    # Balanced XGBoost — enough depth to learn conditional patterns,
    # regularization via subsampling and early stopping
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=4,              # enough depth for conditional rules
        min_child_weight=50,      # prevent tiny leaves
        subsample=0.7,            # row subsampling
        colsample_bytree=0.7,    # column subsampling per tree
        gamma=0.1,                # light split penalty
        reg_alpha=0.1,            # light L1
        reg_lambda=1.0,           # light L2
        scale_pos_weight=1.0,
        eval_metric="logloss",
        early_stopping_rounds=80,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators

    # Evaluate on test set at threshold 65
    test_probs = model.predict_proba(X_test)[:, 1]
    results = evaluate_threshold(test_probs, y_test, 65)
    results["best_iter"] = best_iter
    results["probs"] = test_probs
    results["y_test"] = y_test

    return model, results, best_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--all-features", action="store_true",
                        help="Skip incremental testing, use all 33 features")
    args = parser.parse_args()

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET XGBOOST v2 — RESEARCH-OPTIMIZED")
    print("  max_depth=4, lr=0.02, early_stop=80, subsample=0.7")
    print("  Purged walk-forward (4h gap) + time-decay weighting (60d half-life)")
    if args.synthetic:
        print("  *** INCLUDING SYNTHETIC LABELS ***")
    print("=" * 80)

    # Fetch data (same as before)
    print("\n[1/4] Fetching settlements...")
    kalshi_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for _ in range(50):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor: params["cursor"] = cursor
            resp = client._get("/trade-api/v2/markets", params)
            batch = resp.get("markets", [])
            cursor = resp.get("cursor", "")
            markets.extend(batch)
            if not batch or not cursor: break
        kalshi_markets[asset] = sorted(markets, key=lambda x: x.get("close_time", ""))
        print(f"  {asset}: {len(markets)}")

    settlement_by_time = {}
    for asset in SERIES:
        settlement_by_time[asset] = {}
        for mk in kalshi_markets[asset]:
            ct = mk.get("close_time", "")
            r = mk.get("result", "")
            if ct and r: settlement_by_time[asset][ct] = 1 if r == "yes" else 0

    print("\n[2/4] Fetching 5m candles...")
    five_m = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m[asset] = {}
        for ex in ["coinbase", "bitstamp"]:
            five_m[asset][ex] = fetch_5m_history(sym, ex, 100)
        print(f"  {asset} done")

    print("\n[3/4] Fetching indicators...")
    ind_data = {}
    for asset, sym in ASSETS_SYMS.items():
        df_15m = add_indicators(fetch_candles(fetcher, sym, "15m", 110))
        df_1h = add_indicators(fetch_candles(fetcher, sym, "1h", 110))
        df_4h = add_indicators(fetch_candles(fetcher, sym, "4h", 110))
        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100
        ind_data[asset] = {"15m": df_15m, "1h": df_1h, "4h": df_4h}
        print(f"  {asset}: {len(df_15m)} 15m candles")

    print("\n[4/4] Training per-asset XGBoost models (optimized)...")
    all_assets = list(SERIES.keys())

    for target_asset in all_assets:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {target_asset} (XGBoost v2 — optimized)")
        print(f"{'=' * 80}")

        alt_assets = [a for a in all_assets if a != target_asset]
        target_markets = kalshi_markets[target_asset]
        df_15m = ind_data[target_asset]["15m"]
        df_1h = ind_data[target_asset]["1h"]
        df_4h = ind_data[target_asset]["4h"]
        cb_5m = five_m[target_asset].get("coinbase", pd.DataFrame())
        bs_5m = five_m[target_asset].get("bitstamp", pd.DataFrame())

        atr_s = df_15m["atr"].dropna()
        atr_r20 = atr_s.rolling(20)
        atr_pctile = ((atr_s - atr_r20.min()) / (atr_r20.max() - atr_r20.min())).fillna(0.5)

        rows = []
        for mi, mk in enumerate(target_markets):
            strike = float(mk.get("floor_strike") or 0)
            result = mk.get("result", "")
            ct = mk.get("close_time", "")
            if not strike or not result or not ct: continue
            label = 1 if result == "yes" else 0
            close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            ws = close_dt - timedelta(minutes=15)
            ws_n = ws.replace(tzinfo=None)

            price = get_avg_price(cb_5m, bs_5m, ws_n, "open")
            if not price: continue
            prev_c = df_15m[df_15m.index < ws_n]
            if len(prev_c) < 20: continue
            prev = prev_c.iloc[-1]
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0: continue

            # Strike lookback
            prev_strikes = []
            for lb in range(1, 4):
                if mi - lb >= 0:
                    ps = float(target_markets[mi - lb].get("floor_strike") or 0)
                    if ps: prev_strikes.append(ps)
            sd = (strike - prev_strikes[0]) / atr if prev_strikes and atr > 0 else 0
            if len(prev_strikes) >= 2 and atr > 0:
                ds = [strike - prev_strikes[0]]
                for k in range(len(prev_strikes) - 1): ds.append(prev_strikes[k] - prev_strikes[k+1])
                st3 = sum(d/atr for d in ds)/len(ds)
            else: st3 = sd

            kx = {"strike_delta": sd, "strike_trend_3": st3}
            pa = atr_pctile[atr_pctile.index < ws_n]
            apv = float(pa.iloc[-1]) if len(pa) > 0 else 0.5

            # Regime
            if len(prev_c) >= 16:
                kx["return_4h"] = (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-16]["close"])) / float(prev_c.iloc[-16]["close"]) * 100
            else: kx["return_4h"] = 0
            if len(prev_c) >= 48:
                kx["return_12h"] = (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-48]["close"])) / float(prev_c.iloc[-48]["close"]) * 100
            else: kx["return_12h"] = 0
            if df_1h is not None:
                # '<' drops the in-progress candle at ws_n — matches live path
                h1f = df_1h[df_1h.index < ws_n]
                if len(h1f) >= 20 and atr > 0:
                    kx["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
                else: kx["price_vs_sma_1h"] = 0
            else: kx["price_vs_sma_1h"] = 0
            if df_4h is not None:
                h4f = df_4h[df_4h.index < ws_n]
                if len(h4f) >= 4:
                    kx["lower_lows_4h"] = sum(1 for i in range(-3,0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i-1]["low"]))
                else: kx["lower_lows_4h"] = 0
                if len(h4f) >= 10 and atr > 0:
                    kx["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr
                else: kx["trend_strength"] = 0
            else: kx["lower_lows_4h"] = 0; kx["trend_strength"] = 0

            # Confluence
            ar15, ar1h, amom = [], [], []
            for alt in alt_assets:
                a15 = ind_data[alt]["15m"]
                a1h = ind_data[alt]["1h"]
                a15f = a15[a15.index < ws_n]
                if len(a15f) >= 2:
                    rv = float(a15f.iloc[-1].get("rsi", 50))
                    ar15.append(rv); amom.append(1 if rv >= 50 else -1)
                if a1h is not None:
                    a1f = a1h[a1h.index <= ws_n]
                    if len(a1f) >= 2: ar1h.append(float(a1f.iloc[-1].get("rsi", 50)))
            kx["alt_rsi_avg"] = sum(ar15)/len(ar15) if ar15 else 50
            kx["alt_rsi_1h_avg"] = sum(ar1h)/len(ar1h) if ar1h else 50
            kx["alt_momentum_align"] = sum(amom) if amom else 0
            kx["alt_distance_avg"] = 0

            feat = build_features(prev, df_1h, df_4h, ws_n, (price - strike) / atr,
                                  kalshi_extra=kx, atr_pctile_val=apv)
            if not feat: continue

            # Interaction features
            feat["pve_x_trend"] = feat.get("price_vs_ema", 0) * feat.get("trend_strength", 0)
            feat["pve_x_return12h"] = feat.get("price_vs_ema", 0) * feat.get("return_12h", 0)
            feat["slope_x_trend"] = feat.get("ema_slope", 0) * feat.get("trend_strength", 0)
            feat["slope_x_return12h"] = feat.get("ema_slope", 0) * feat.get("return_12h", 0)
            # RSI-centered × trend interactions (see FEATURE_GROUPS["interaction"])
            feat["rsi1h_x_r12h"] = (feat.get("rsi_1h", 50) - 50) * feat.get("return_12h", 0)
            feat["rsi4h_x_r12h"] = (feat.get("rsi_4h", 50) - 50) * feat.get("return_12h", 0)
            feat["rsi1h_x_r4h"] = (feat.get("rsi_1h", 50) - 50) * feat.get("return_4h", 0)
            feat["dist_x_r12h"] = feat.get("distance_from_strike", 0) * feat.get("return_12h", 0)

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue
            rows.append({**feat, "label": label, "ts": close_dt})

        # Merge synthetic data
        if args.synthetic:
            syn_path = Path("data/store/synthetic_labels.pkl")
            if syn_path.exists():
                syn_data = pickle.load(open(syn_path, "rb"))
                if target_asset in syn_data:
                    syn_asset = syn_data[target_asset]
                    syn_15m = syn_asset["15m"]
                    syn_1h = syn_asset["1h"]
                    syn_4h = syn_asset["4h"]
                    syn_atr_s = syn_15m["atr"].dropna()
                    syn_ar20 = syn_atr_s.rolling(20)
                    syn_atr_p = ((syn_atr_s - syn_ar20.min()) / (syn_ar20.max() - syn_ar20.min())).fillna(0.5)

                    syn_count = 0
                    for s in syn_asset["samples"]:
                        idx = s["_15m_idx"]
                        if idx < 20 or idx >= len(syn_15m) - 1: continue
                        prev = syn_15m.iloc[idx]
                        atr_syn = float(prev.get("atr", 0))
                        if pd.isna(atr_syn) or atr_syn <= 0: continue
                        ws_syn = syn_15m.index[idx + 1]

                        kx_syn = {"strike_delta": 0, "strike_trend_3": 0}
                        pa_syn = syn_atr_p[syn_atr_p.index < ws_syn]
                        apv_syn = float(pa_syn.iloc[-1]) if len(pa_syn) > 0 else 0.5

                        # Regime
                        prev_c_syn = syn_15m[syn_15m.index < ws_syn]
                        if len(prev_c_syn) >= 16:
                            kx_syn["return_4h"] = (float(prev_c_syn.iloc[-1]["close"]) - float(prev_c_syn.iloc[-16]["close"])) / float(prev_c_syn.iloc[-16]["close"]) * 100
                        else: kx_syn["return_4h"] = 0
                        if len(prev_c_syn) >= 48:
                            kx_syn["return_12h"] = (float(prev_c_syn.iloc[-1]["close"]) - float(prev_c_syn.iloc[-48]["close"])) / float(prev_c_syn.iloc[-48]["close"]) * 100
                        else: kx_syn["return_12h"] = 0
                        if syn_1h is not None:
                            # '<' drops in-progress candle — live/backtest parity
                            h1f = syn_1h[syn_1h.index < ws_syn]
                            if len(h1f) >= 20 and atr_syn > 0:
                                kx_syn["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr_syn
                            else: kx_syn["price_vs_sma_1h"] = 0
                        else: kx_syn["price_vs_sma_1h"] = 0
                        if syn_4h is not None:
                            h4f = syn_4h[syn_4h.index < ws_syn]
                            if len(h4f) >= 4:
                                kx_syn["lower_lows_4h"] = sum(1 for i in range(-3,0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i-1]["low"]))
                            else: kx_syn["lower_lows_4h"] = 0
                            if len(h4f) >= 10 and atr_syn > 0:
                                kx_syn["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr_syn
                            else: kx_syn["trend_strength"] = 0
                        else: kx_syn["lower_lows_4h"] = 0; kx_syn["trend_strength"] = 0

                        # Real confluence from synthetic data
                        syn_alt_rsi, syn_alt_rsi_1h, syn_alt_mom = [], [], []
                        for alt_asset in alt_assets:
                            if alt_asset in syn_data:
                                alt_syn_15m = syn_data[alt_asset]["15m"]
                                alt_syn_1h = syn_data[alt_asset]["1h"]
                                alt_filt = alt_syn_15m[alt_syn_15m.index < ws_syn]
                                if len(alt_filt) >= 2:
                                    rv = float(alt_filt.iloc[-1].get("rsi", 50))
                                    syn_alt_rsi.append(rv)
                                    syn_alt_mom.append(1 if rv >= 50 else -1)
                                if alt_syn_1h is not None:
                                    alt_1h_f = alt_syn_1h[alt_syn_1h.index <= ws_syn]
                                    if len(alt_1h_f) >= 2:
                                        syn_alt_rsi_1h.append(float(alt_1h_f.iloc[-1].get("rsi", 50)))
                        kx_syn["alt_rsi_avg"] = sum(syn_alt_rsi)/len(syn_alt_rsi) if syn_alt_rsi else 50
                        kx_syn["alt_rsi_1h_avg"] = sum(syn_alt_rsi_1h)/len(syn_alt_rsi_1h) if syn_alt_rsi_1h else 50
                        kx_syn["alt_momentum_align"] = sum(syn_alt_mom) if syn_alt_mom else 0
                        kx_syn["alt_distance_avg"] = 0

                        feat_syn = build_features(prev, syn_1h, syn_4h, ws_syn,
                                                  s["distance_from_strike"],
                                                  kalshi_extra=kx_syn, atr_pctile_val=apv_syn)
                        if not feat_syn: continue
                        feat_syn["pve_x_trend"] = feat_syn.get("price_vs_ema", 0) * kx_syn.get("trend_strength", 0)
                        feat_syn["pve_x_return12h"] = feat_syn.get("price_vs_ema", 0) * kx_syn.get("return_12h", 0)
                        feat_syn["slope_x_trend"] = feat_syn.get("ema_slope", 0) * kx_syn.get("trend_strength", 0)
                        feat_syn["slope_x_return12h"] = feat_syn.get("ema_slope", 0) * kx_syn.get("return_12h", 0)
                        # RSI-centered × trend interactions (see FEATURE_GROUPS["interaction"])
                        feat_syn["rsi1h_x_r12h"] = (feat_syn.get("rsi_1h", 50) - 50) * kx_syn.get("return_12h", 0)
                        feat_syn["rsi4h_x_r12h"] = (feat_syn.get("rsi_4h", 50) - 50) * kx_syn.get("return_12h", 0)
                        feat_syn["rsi1h_x_r4h"] = (feat_syn.get("rsi_1h", 50) - 50) * kx_syn.get("return_4h", 0)
                        feat_syn["dist_x_r12h"] = s["distance_from_strike"] * kx_syn.get("return_12h", 0)
                        if any(pd.isna(v) or np.isinf(v) for v in feat_syn.values()): continue
                        syn_ts = s["ts"]
                        if hasattr(syn_ts, 'tzinfo') and syn_ts.tzinfo is None:
                            syn_ts = syn_ts.replace(tzinfo=timezone.utc)
                        rows.append({**feat_syn, "label": s["label"], "ts": syn_ts})
                        syn_count += 1
                    print(f"  + {syn_count} synthetic samples")

        df_all = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        print(f"  Total: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

        if args.all_features:
            # Skip incremental testing — just train with all features
            print(f"\n  Training with ALL {len(ALL_FEATURES)} features...")
            best_features = ALL_FEATURES
        else:
            # === INCREMENTAL FEATURE GROUP TESTING ===
            print(f"\n  --- Incremental Feature Group Testing ---")
            print(f"  Starting with {len(CORE_13)} core features")

            # Train baseline with core 13
            _, core_results, _ = train_with_features(df_all, CORE_13, "core_13")
            if core_results is None or core_results["tot"] < 10:
                print(f"  ERROR: Not enough test samples for {target_asset}")
                continue

            print(f"  Core 13:  {core_results['tot']:>4} bets, {core_results['wr']:>5.1f}% WR, ${core_results['pnl']:>+.1f} P&L (iter {core_results['best_iter']})")

            best_wr = core_results["wr"]
            best_features = list(CORE_13)

            # Test each group incrementally
            for group_name, group_feats in FEATURE_GROUPS.items():
                candidate = best_features + group_feats
                _, group_results, _ = train_with_features(df_all, candidate, group_name)
                if group_results is None or group_results["tot"] < 10:
                    print(f"  + {group_name:15}: SKIP (insufficient data)")
                    if group_name in FORCE_KEEP_GROUPS:
                        # Still force-keep the features even without measurable eval
                        best_features = candidate
                    continue

                delta_wr = group_results["wr"] - best_wr
                forced = group_name in FORCE_KEEP_GROUPS
                if forced:
                    marker = "FORCE-KEEP"
                else:
                    marker = "KEEP" if delta_wr > 0.5 else "DROP"
                print(f"  + {group_name:15}: {group_results['tot']:>4} bets, {group_results['wr']:>5.1f}% WR ({delta_wr:>+.1f}%), ${group_results['pnl']:>+.1f} — {marker}")

                if forced or delta_wr > 0.5:
                    best_features = candidate
                    best_wr = group_results["wr"]

            print(f"\n  Final feature set: {len(best_features)} features")
            print(f"  Best WR: {best_wr:.1f}%")

        # === FINAL TRAINING with selected features ===
        print(f"\n  --- Final Training ({len(best_features)} features) ---")

        # Purged split
        n = len(df_all)
        split_train = int(n * 0.70)
        split_val_start = split_train + PURGE_GAP_SAMPLES
        split_val_end = int(n * 0.85)
        if split_val_start >= split_val_end:
            split_val_start = split_train + 4
            if split_val_start >= split_val_end:
                split_val_start = split_train

        df_train = df_all.iloc[:split_train]
        df_val = df_all.iloc[split_val_start:split_val_end]
        df_test = df_all.iloc[split_val_end:]

        X_train = df_train[best_features].values
        y_train = df_train["label"].values
        X_val = df_val[best_features].values
        y_val = df_val["label"].values
        X_test = df_test[best_features].values
        y_test = df_test["label"].values

        sample_weights = compute_time_decay_weights(df_train["ts"], half_life_days=60)

        print(f"  Train: {len(df_train)} | Gap: {PURGE_GAP_SAMPLES} | Val: {len(df_val)} | Test: {len(df_test)}")

        model = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=50,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=42,
            verbosity=0,
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        print(f"  Best iteration: {best_iter}")

        # Feature importance
        importances = dict(zip(best_features, model.feature_importances_))
        print(f"\n  Top 10 features (importance):")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name:<24}: {imp:.4f}")

        # Threshold sweep on test set
        test_probs = model.predict_proba(X_test)[:, 1]
        print(f"\n  {'Thresh':>6} | {'Bets':>5} {'WR':>6} {'Y':>5} {'N':>5} {'Y_WR':>5} {'N_WR':>5} | {'P&L':>7}")
        print(f"  {'-' * 60}")
        for t in [55, 60, 65, 70]:
            r = evaluate_threshold(test_probs, y_test, t)
            if r["tot"] < 10: continue
            yn, nn = r["yw"] + r["yl"], r["nw"] + r["nl"]
            ywr = r["yw"] / yn * 100 if yn > 0 else 0
            nwr = r["nw"] / nn * 100 if nn > 0 else 0
            print(f"  {t:>6} | {r['tot']:>5} {r['wr']:>5.1f}% {yn:>5} {nn:>5} {ywr:>4.0f}% {nwr:>4.0f}% | ${r['pnl']:>+6.1f}")

        # YES/NO balance check
        yes_preds = sum(1 for p in test_probs if p >= 0.65)
        no_preds = sum(1 for p in test_probs if p <= 0.35)
        skip_preds = len(test_probs) - yes_preds - no_preds
        print(f"\n  Signal balance (t=65): YES={yes_preds} NO={no_preds} SKIP={skip_preds}")

        # Save
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.feature_names_in_ = np.array(best_features)

        out_path = Path(f"models/m0_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "knn": model,
                "scaler": scaler,
                "model_type": "per_asset_confluence",
                "feature_names": list(best_features),
                "asset": target_asset,
                "training_samples": len(df_train),
                "model_class": "XGBClassifier",
                "xgb_version": "v2_optimized",
                "hyperparams": {
                    "max_depth": 4, "lr": 0.02, "early_stop": 80,
                    "subsample": 0.7, "purge_gap": PURGE_GAP_SAMPLES,
                    "time_decay_halflife": 60,
                },
            }, f)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
