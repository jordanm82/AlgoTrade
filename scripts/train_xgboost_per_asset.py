#!/usr/bin/env python3
"""Train per-asset XGBoost models with anti-overfitting safeguards.

Key differences from LogReg:
  - Can learn conditional relationships (IF dip AND downtrend THEN NO)
  - Uses randomness in subsampling to prevent memorization
  - Small learning rate + early stopping to find optimal complexity
  - Per-asset models with cross-asset confluence + regime features

Usage:
    ./venv/bin/python scripts/train_xgboost_per_asset.py [--synthetic]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET XGBOOST MODELS")
    print("Anti-overfitting: subsample=0.7, colsample=0.7, lr=0.03, early stopping")
    if args.synthetic:
        print("*** INCLUDING SYNTHETIC LABELS ***")
    print("=" * 80)

    # Fetch data (same as LogReg training)
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

    print("\n[4/4] Training per-asset XGBoost models...")
    all_assets = list(SERIES.keys())

    for target_asset in all_assets:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {target_asset} (XGBoost)")
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
                h1f = df_1h[df_1h.index <= ws_n]
                if len(h1f) >= 20 and atr > 0:
                    kx["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
                else: kx["price_vs_sma_1h"] = 0
            else: kx["price_vs_sma_1h"] = 0
            if df_4h is not None:
                h4f = df_4h[df_4h.index <= ws_n]
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
                            h1f = syn_1h[syn_1h.index <= ws_syn]
                            if len(h1f) >= 20 and atr_syn > 0:
                                kx_syn["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr_syn
                            else: kx_syn["price_vs_sma_1h"] = 0
                        else: kx_syn["price_vs_sma_1h"] = 0
                        if syn_4h is not None:
                            h4f = syn_4h[syn_4h.index <= ws_syn]
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
                        if any(pd.isna(v) or np.isinf(v) for v in feat_syn.values()): continue
                        syn_ts = s["ts"]
                        if hasattr(syn_ts, 'tzinfo') and syn_ts.tzinfo is None:
                            syn_ts = syn_ts.replace(tzinfo=timezone.utc)
                        rows.append({**feat_syn, "label": s["label"], "ts": syn_ts})
                        syn_count += 1
                    print(f"  + {syn_count} synthetic samples")

        df_all = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        print(f"  Total: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

        # Walk-forward split: 70% train, 15% validation (early stopping), 15% test
        split_train = int(len(df_all) * 0.70)
        split_val = int(len(df_all) * 0.85)
        df_train = df_all.iloc[:split_train]
        df_val = df_all.iloc[split_train:split_val]
        df_test = df_all.iloc[split_val:]

        X_train = df_train[ALL_FEATURES].values
        y_train = df_train["label"].values
        X_val = df_val[ALL_FEATURES].values
        y_val = df_val["label"].values
        X_test = df_test[ALL_FEATURES].values
        y_test = df_test["label"].values

        print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

        # XGBoost with anti-overfitting
        model = XGBClassifier(
            n_estimators=2000,          # high max, early stopping will find optimal
            learning_rate=0.03,         # small step size for fine-grained early stopping
            max_depth=4,                # shallow trees to prevent memorization
            min_child_weight=50,        # need 50+ samples per leaf
            subsample=0.7,              # random 70% of rows per tree
            colsample_bytree=0.7,       # random 70% of features per tree
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            scale_pos_weight=1.0,       # balanced classes
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

        # Fit with early stopping on validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        print(f"  Best iteration: {best_iter}")

        # Feature importance
        importances = dict(zip(ALL_FEATURES, model.feature_importances_))
        print(f"\n  Top 10 features (importance):")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name:<24}: {imp:.4f}")

        # Test
        test_probs = model.predict_proba(X_test)[:, 1]
        print(f"\n  {'Thresh':>6} | {'Bets':>5} {'WR':>6} {'Y':>5} {'N':>5} {'Y_WR':>5} {'N_WR':>5} | {'P&L':>7}")
        print(f"  {'-' * 60}")
        for t in [55, 60, 65, 70]:
            w, l, yw, yl, nw, nl = 0, 0, 0, 0, 0, 0
            for i in range(len(y_test)):
                p = int(test_probs[i] * 100)
                if p >= t: side = "YES"
                elif p <= (100 - t): side = "NO"
                else: continue
                won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
                if won:
                    w += 1
                    if side == "YES": yw += 1
                    else: nw += 1
                else:
                    l += 1
                    if side == "YES": yl += 1
                    else: nl += 1
            tot = w + l
            if tot < 10: continue
            wr = w / tot * 100
            yn, nn = yw + yl, nw + nl
            ywr = yw / yn * 100 if yn > 0 else 0
            nwr = nw / nn * 100 if nn > 0 else 0
            pnl = w * 0.40 - l * 0.60
            print(f"  {t:>6} | {tot:>5} {wr:>5.1f}% {yn:>5} {nn:>5} {ywr:>4.0f}% {nwr:>4.0f}% | ${pnl:>+6.1f}")

        # Save — use same format as LogReg for compatibility
        # The predictor checks feature_names_in_ on the scaler, so we need a scaler
        scaler = StandardScaler()
        scaler.fit(X_train)  # fit scaler for parity check compatibility
        scaler.feature_names_in_ = np.array(ALL_FEATURES)

        out_path = Path(f"models/m0_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "knn": model,  # key name kept for compatibility with predictor
                "scaler": scaler,
                "model_type": "per_asset_confluence",
                "feature_names": ALL_FEATURES,
                "asset": target_asset,
                "training_samples": len(df_train),
                "model_class": "XGBClassifier",
            }, f)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
