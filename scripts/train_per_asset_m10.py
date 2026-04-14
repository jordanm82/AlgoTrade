#!/usr/bin/env python3
"""Train per-asset M10 exit models with research-optimized XGBoost.

Same optimizations as M0 v2:
  - Ultra-conservative: max_depth=2, lr=0.005, early_stopping=200
  - Purged walk-forward (4h gap) + time-decay weighting (60d half-life)
  - Additional 5m intra-window features (what happened during minutes 0-10)

Usage:
    ./venv/bin/python scripts/train_per_asset_m10.py
"""
import pickle, sys, time
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from scripts.backtest_kalshi_labels import fetch_5m_history, fetch_candles, get_avg_price, build_features

SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

BASE_FEATURES = [
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
    "vol_ratio", "adx", "rsi_1h", "rsi_4h", "distance_from_strike",
    "strike_delta", "strike_trend_3",
    "hour_sin", "hour_cos",
    "rsi_alignment", "atr_percentile", "rsi_15m", "bbw",
]

CONFLUENCE_FEATURES = [
    "alt_rsi_avg", "alt_rsi_1h_avg", "alt_momentum_align",
    "alt_distance_avg",
]

INTRA_WINDOW_FEATURES = [
    "price_move_atr",
    "candle1_range_atr",
    "candle2_range_atr",
    "momentum_shift",
    "volume_acceleration",
]

REGIME_FEATURES = [
    "return_4h", "return_12h", "price_vs_sma_1h",
    "lower_lows_4h", "trend_strength",
]

INTERACTION_FEATURES = [
    "pve_x_trend", "pve_x_return12h", "slope_x_trend", "slope_x_return12h",
    # RSI-centered × trend interactions for continuation vs reversion
    "rsi1h_x_r12h", "rsi4h_x_r12h", "rsi1h_x_r4h", "dist_x_r12h",
]

ALL_FEATURES = BASE_FEATURES + CONFLUENCE_FEATURES + INTRA_WINDOW_FEATURES + REGIME_FEATURES + INTERACTION_FEATURES

# Purge gap: 4 hours = 16 fifteen-minute windows
PURGE_GAP_SAMPLES = 16


def compute_time_decay_weights(timestamps, half_life_days=60):
    """Exponential decay weights — recent samples weighted more heavily."""
    latest = timestamps.max()
    days_ago = (latest - timestamps).dt.total_seconds() / 86400
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    weights = weights / weights.mean()
    return weights.values


def main():
    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET M10 MODELS v2 — RESEARCH-OPTIMIZED")
    print("  max_depth=4, lr=0.02, early_stop=80, subsample=0.7")
    print("  Purged walk-forward (4h gap) + time-decay weighting (60d half-life)")
    print("=" * 80)

    print("\n[1/4] Fetching settlements...")
    kalshi_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for _ in range(30):
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
            if ct and r:
                settlement_by_time[asset][ct] = 1 if r == "yes" else 0

    print("\n[2/4] Fetching 5m candles...")
    five_m = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m[asset] = {
            "coinbase": fetch_5m_history(sym, "coinbase", 100),
            "bitstamp": fetch_5m_history(sym, "bitstamp", 100),
        }
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

    print("\n[4/4] Training per-asset M10 models (optimized)...")
    all_assets = list(SERIES.keys())

    for target_asset in all_assets:
        print(f"\n{'=' * 80}")
        print(f"M10 TRAINING: {target_asset} (XGBoost v2 — optimized)")
        print(f"{'=' * 80}")

        alt_assets = [a for a in all_assets if a != target_asset]
        target_markets = kalshi_markets[target_asset]
        df_15m = ind_data[target_asset]["15m"]
        df_1h = ind_data[target_asset]["1h"]
        df_4h = ind_data[target_asset]["4h"]
        cb_5m = five_m[target_asset]["coinbase"]
        bs_5m = five_m[target_asset]["bitstamp"]

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

            # Price at MINUTE 10
            min10 = ws_n + timedelta(minutes=10)
            price = get_avg_price(cb_5m, bs_5m, min10, "open")
            if not price: continue

            prev_c = df_15m[df_15m.index < ws_n]
            if len(prev_c) < 20: continue
            prev = prev_c.iloc[-1]
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0: continue

            # Kalshi lookback
            prev_strikes = []
            for lb in range(1, 4):
                if mi - lb >= 0:
                    ps = float(target_markets[mi - lb].get("floor_strike") or 0)
                    if ps: prev_strikes.append(ps)
            sd = (strike - prev_strikes[0]) / atr if prev_strikes and atr > 0 else 0
            if len(prev_strikes) >= 2 and atr > 0:
                ds = [strike - prev_strikes[0]]
                for k in range(len(prev_strikes) - 1):
                    ds.append(prev_strikes[k] - prev_strikes[k + 1])
                st3 = sum(d / atr for d in ds) / len(ds)
            else:
                st3 = sd
            kx = {"strike_delta": sd, "strike_trend_3": st3}
            pa = atr_pctile[atr_pctile.index < ws_n]
            apv = float(pa.iloc[-1]) if len(pa) > 0 else 0.5

            # Distance at minute 10
            feat = build_features(prev, df_1h, df_4h, ws_n, (price - strike) / atr,
                                  kalshi_extra=kx, atr_pctile_val=apv)
            if not feat: continue

            # Confluence
            alt_rsi_15m, alt_rsi_1h, alt_momentum = [], [], []
            alt_distances = []
            for alt in alt_assets:
                alt_15m = ind_data[alt]["15m"]
                alt_1h_df = ind_data[alt]["1h"]
                alt_prev_candles = alt_15m[alt_15m.index < ws_n]
                if len(alt_prev_candles) >= 2:
                    ar = alt_prev_candles.iloc[-1]
                    rsi_val = float(ar.get("rsi", 50))
                    alt_rsi_15m.append(rsi_val)
                    alt_momentum.append(1 if rsi_val >= 50 else -1)
                if alt_1h_df is not None:
                    alt_1h_prev = alt_1h_df[alt_1h_df.index <= ws_n]
                    if len(alt_1h_prev) >= 2:
                        alt_rsi_1h.append(float(alt_1h_prev.iloc[-1].get("rsi", 50)))
                alt_price = get_avg_price(five_m[alt]["coinbase"], five_m[alt]["bitstamp"], min10, "open")
                if alt_price and len(alt_prev_candles) >= 2:
                    alt_atr = float(alt_prev_candles.iloc[-1].get("atr", 0))
                    for alt_mk in kalshi_markets[alt]:
                        if alt_mk.get("close_time") == ct:
                            alt_strike = float(alt_mk.get("floor_strike") or 0)
                            if alt_strike and alt_atr > 0:
                                alt_distances.append((alt_price - alt_strike) / alt_atr)
                            break

            feat["alt_rsi_avg"] = sum(alt_rsi_15m) / len(alt_rsi_15m) if alt_rsi_15m else 50
            feat["alt_rsi_1h_avg"] = sum(alt_rsi_1h) / len(alt_rsi_1h) if alt_rsi_1h else 50
            feat["alt_momentum_align"] = sum(alt_momentum) if alt_momentum else 0
            feat["alt_distance_avg"] = sum(alt_distances) / len(alt_distances) if alt_distances else 0

            # === Intra-window 5m features (what happened during minutes 0-10) ===
            min5 = ws_n + timedelta(minutes=5)
            c1_prices = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty: continue
                mask = (df_5m.index >= ws_n) & (df_5m.index < min5)
                if mask.sum() > 0:
                    c1_prices.append(df_5m[mask].iloc[0])
            c2_prices = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty: continue
                mask = (df_5m.index >= min5) & (df_5m.index < min10)
                if mask.sum() > 0:
                    c2_prices.append(df_5m[mask].iloc[0])

            if c1_prices and c2_prices and atr > 0:
                c1_open = sum(float(c["open"]) for c in c1_prices) / len(c1_prices)
                c1_close = sum(float(c["close"]) for c in c1_prices) / len(c1_prices)
                c1_high = sum(float(c["high"]) for c in c1_prices) / len(c1_prices)
                c1_low = sum(float(c["low"]) for c in c1_prices) / len(c1_prices)
                c1_vol = sum(float(c["volume"]) for c in c1_prices) / len(c1_prices)

                c2_open = sum(float(c["open"]) for c in c2_prices) / len(c2_prices)
                c2_close = sum(float(c["close"]) for c in c2_prices) / len(c2_prices)
                c2_high = sum(float(c["high"]) for c in c2_prices) / len(c2_prices)
                c2_low = sum(float(c["low"]) for c in c2_prices) / len(c2_prices)
                c2_vol = sum(float(c["volume"]) for c in c2_prices) / len(c2_prices)

                feat["price_move_atr"] = (c2_close - c1_open) / atr
                feat["candle1_range_atr"] = (c1_high - c1_low) / atr
                feat["candle2_range_atr"] = (c2_high - c2_low) / atr
                feat["momentum_shift"] = (c2_close - c1_close) / atr
                feat["volume_acceleration"] = c2_vol / c1_vol if c1_vol > 0 else 1.0
            else:
                feat["price_move_atr"] = 0
                feat["candle1_range_atr"] = 0
                feat["candle2_range_atr"] = 0
                feat["momentum_shift"] = 0
                feat["volume_acceleration"] = 1.0

            # === REGIME FEATURES ===
            hr4_candles = df_15m[df_15m.index < ws_n]
            if len(hr4_candles) >= 16:
                feat["return_4h"] = (float(hr4_candles.iloc[-1]["close"]) - float(hr4_candles.iloc[-16]["close"])) / float(hr4_candles.iloc[-16]["close"]) * 100
            else:
                feat["return_4h"] = 0
            if len(hr4_candles) >= 48:
                feat["return_12h"] = (float(hr4_candles.iloc[-1]["close"]) - float(hr4_candles.iloc[-48]["close"])) / float(hr4_candles.iloc[-48]["close"]) * 100
            else:
                feat["return_12h"] = 0
            if df_1h is not None:
                # '<' drops the in-progress candle at ws_n — live/backtest parity
                h1f = df_1h[df_1h.index < ws_n]
                if len(h1f) >= 20 and atr > 0:
                    feat["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
                else:
                    feat["price_vs_sma_1h"] = 0
            else:
                feat["price_vs_sma_1h"] = 0
            if df_4h is not None:
                h4f = df_4h[df_4h.index < ws_n]
                if len(h4f) >= 4:
                    feat["lower_lows_4h"] = sum(1 for i in range(-3,0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i-1]["low"]))
                else:
                    feat["lower_lows_4h"] = 0
                if len(h4f) >= 10 and atr > 0:
                    feat["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr
                else:
                    feat["trend_strength"] = 0
            else:
                feat["lower_lows_4h"] = 0
                feat["trend_strength"] = 0

            # Interaction features
            _pve = feat.get("price_vs_ema", 0)
            _es = feat.get("ema_slope", 0)
            _ts = feat.get("trend_strength", 0)
            _r12 = feat.get("return_12h", 0)
            feat["pve_x_trend"] = _pve * _ts
            feat["pve_x_return12h"] = _pve * _r12
            feat["slope_x_trend"] = _es * _ts
            feat["slope_x_return12h"] = _es * _r12
            # RSI-centered × trend interactions
            _rsi1h_c = feat.get("rsi_1h", 50) - 50
            _rsi4h_c = feat.get("rsi_4h", 50) - 50
            _r4 = feat.get("return_4h", 0)
            _dist = feat.get("distance_from_strike", 0)
            feat["rsi1h_x_r12h"] = _rsi1h_c * _r12
            feat["rsi4h_x_r12h"] = _rsi4h_c * _r12
            feat["rsi1h_x_r4h"] = _rsi1h_c * _r4
            feat["dist_x_r12h"] = _dist * _r12

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue
            rows.append({**feat, "label": label, "ts": close_dt})

        df_all = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        print(f"  Samples: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

        # Purged walk-forward split
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

        X_train = df_train[ALL_FEATURES].values
        y_train = df_train["label"].values
        X_val = df_val[ALL_FEATURES].values
        y_val = df_val["label"].values
        X_test = df_test[ALL_FEATURES].values
        y_test = df_test["label"].values

        # Time-decay sample weights
        latest = df_train["ts"].max()
        days_ago = (latest - df_train["ts"]).dt.total_seconds() / 86400
        sample_weights = np.exp(-np.log(2) * days_ago / 60)
        sample_weights = sample_weights / sample_weights.mean()

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
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights.values,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        print(f"  Best iteration: {best_iter}")

        # Top features by importance
        importances = dict(zip(ALL_FEATURES, model.feature_importances_))
        print(f"\n  Top 10 features:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            marker = " [5m]" if name in INTRA_WINDOW_FEATURES else ""
            print(f"    {name:<24}: {imp:.4f}{marker}")

        # Threshold sweep
        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.feature_names_in_ = np.array(ALL_FEATURES)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"\n  {'Thresh':>6} | {'Bets':>5} {'WR':>6} | {'P&L':>7}")
        print(f"  {'-' * 35}")
        for t in [55, 60, 65, 70, 75, 80]:
            w, l = 0, 0
            for i in range(len(y_test)):
                p = int(probs[i] * 100)
                if p >= t: side = "YES"
                elif p <= (100 - t): side = "NO"
                else: continue
                won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
                if won: w += 1
                else: l += 1
            tot = w + l
            if tot < 10: continue
            wr = w / tot * 100
            pnl = w * 0.40 - l * 0.60
            print(f"  {t:>6} | {tot:>5} {wr:>5.1f}% | ${pnl:>+6.1f}")

        # YES/NO balance check
        yes_preds = sum(1 for p in probs if p >= 0.80)
        no_preds = sum(1 for p in probs if p <= 0.20)
        skip_preds = len(probs) - yes_preds - no_preds
        print(f"\n  Signal balance (t=80): YES={yes_preds} NO={no_preds} SKIP={skip_preds}")

        out_path = Path(f"models/m10_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": scaler,
                "model_type": "per_asset_m10_confluence",
                "feature_names": ALL_FEATURES,
                "asset": target_asset,
                "training_samples": len(df_train),
                "model_class": "XGBClassifier",
                "xgb_version": "v2_optimized",
            }, f)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
