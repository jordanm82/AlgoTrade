#!/usr/bin/env python3
"""Train per-asset M0 models with cross-asset confluence features.

Each asset gets its own LogReg model that sees:
  - 24 base features (same as unified model)
  - 5 cross-asset confluence features (from the other 3 assets)

Usage:
    ./venv/bin/python scripts/train_per_asset_models.py
"""
import pickle, sys, time
import numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from sklearn.linear_model import LogisticRegression
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
    # prev_result, prev_3_yes_pct, streak_length REMOVED — backwards-looking momentum trap
    "strike_delta", "strike_trend_3",
    "hour_sin", "hour_cos",
    "rsi_alignment", "atr_percentile", "bbw",
    # rsi_15m REMOVED from M0 — causes mean-reversion bias that buys every dip.
    # Kept in M10 where short-term RSI is valid for 5-min exit decisions.
]

CONFLUENCE_FEATURES = [
    "alt_rsi_avg",
    "alt_rsi_1h_avg",
    "alt_momentum_align",
    "alt_distance_avg",
]

# Regime detection — multi-hour context the model needs to distinguish
# "dip in uptrend" (buy) from "continuation in downtrend" (sell)
REGIME_FEATURES = [
    "return_4h",        # price change over last 4 hours (% * 100)
    "return_12h",       # price change over last 12 hours
    "price_vs_sma_1h",  # price position relative to 1h SMA (in ATR units)
    "lower_lows_4h",    # count of 4h candles making lower lows (0-3)
    "trend_strength",   # 4h close vs 4h SMA — signed trend (positive=bull, negative=bear)
]

ALL_FEATURES = BASE_FEATURES + CONFLUENCE_FEATURES + REGIME_FEATURES


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser()
    _parser.add_argument("--synthetic", action="store_true", help="Include synthetic labels from historical data")
    _args = _parser.parse_args()

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET MODELS WITH CROSS-ASSET CONFLUENCE")
    if _args.synthetic:
        print("*** INCLUDING SYNTHETIC LABELS FROM HISTORICAL DATA ***")
    print("=" * 80)

    # Fetch all data
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

    # Build settlement lookup by close_time
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

    # Train per-asset models
    print("\n[4/4] Training per-asset models...")
    all_assets = list(SERIES.keys())

    for target_asset in all_assets:
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {target_asset}")
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

            price = get_avg_price(cb_5m, bs_5m, ws_n, "open")
            if not price: continue
            prev_c = df_15m[df_15m.index < ws_n]
            if len(prev_c) < 20: continue
            prev = prev_c.iloc[-1]
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0: continue

            # Kalshi lookback
            prev_results, prev_strikes = [], []
            for lb in range(1, 4):
                if mi - lb >= 0:
                    pm = target_markets[mi - lb]
                    pr = pm.get("result", "")
                    ps = float(pm.get("floor_strike") or 0)
                    if pr: prev_results.append(1 if pr == "yes" else 0)
                    if ps: prev_strikes.append(ps)
            streak = 0
            if prev_results:
                lr = prev_results[0]
                for r in prev_results:
                    if r == lr: streak += 1
                    else: break
                streak = streak if lr == 1 else -streak
            sd = (strike - prev_strikes[0]) / atr if prev_strikes and atr > 0 else 0
            if len(prev_strikes) >= 2 and atr > 0:
                ds = [strike - prev_strikes[0]]
                for k in range(len(prev_strikes) - 1):
                    ds.append(prev_strikes[k] - prev_strikes[k + 1])
                st3 = sum(d / atr for d in ds) / len(ds)
            else:
                st3 = sd
            kx = {"prev_result": prev_results[0] if prev_results else 0.5,
                  "prev_3_yes_pct": sum(prev_results) / len(prev_results) if prev_results else 0.5,
                  "streak_length": streak, "strike_delta": sd, "strike_trend_3": st3}
            pa = atr_pctile[atr_pctile.index < ws_n]
            apv = float(pa.iloc[-1]) if len(pa) > 0 else 0.5

            feat = build_features(prev, df_1h, df_4h, ws_n, (price - strike) / atr,
                                  kalshi_extra=kx, atr_pctile_val=apv)
            if not feat: continue

            # === CONFLUENCE ===
            alt_rsi_15m, alt_rsi_1h, alt_momentum = [], [], []
            alt_prev_results, alt_distances = [], []

            for alt in alt_assets:
                alt_15m = ind_data[alt]["15m"]
                alt_1h_df = ind_data[alt]["1h"]
                alt_prev = alt_15m[alt_15m.index < ws_n]
                if len(alt_prev) >= 2:
                    ar = alt_prev.iloc[-1]
                    rsi_val = float(ar.get("rsi", 50))
                    alt_rsi_15m.append(rsi_val)
                    alt_momentum.append(1 if rsi_val >= 50 else -1)
                if alt_1h_df is not None:
                    alt_1h_prev = alt_1h_df[alt_1h_df.index <= ws_n]
                    if len(alt_1h_prev) >= 2:
                        alt_rsi_1h.append(float(alt_1h_prev.iloc[-1].get("rsi", 50)))
                if ct in settlement_by_time[alt]:
                    alt_prev_results.append(settlement_by_time[alt][ct])
                # Alt distance
                alt_price = get_avg_price(five_m[alt]["coinbase"], five_m[alt]["bitstamp"], ws_n, "open")
                if alt_price and len(alt_prev) >= 2:
                    alt_atr = float(alt_prev.iloc[-1].get("atr", 0))
                    for alt_mk in kalshi_markets[alt]:
                        if alt_mk.get("close_time") == ct:
                            alt_strike = float(alt_mk.get("floor_strike") or 0)
                            if alt_strike and alt_atr > 0:
                                alt_distances.append((alt_price - alt_strike) / alt_atr)
                            break

            feat["alt_rsi_avg"] = sum(alt_rsi_15m) / len(alt_rsi_15m) if alt_rsi_15m else 50
            feat["alt_rsi_1h_avg"] = sum(alt_rsi_1h) / len(alt_rsi_1h) if alt_rsi_1h else 50
            feat["alt_momentum_align"] = sum(alt_momentum) if alt_momentum else 0
            target_prev = prev_results[0] if prev_results else 0.5
            all_prev = [target_prev] + alt_prev_results
            feat["prev_result_consensus"] = sum(1 for r in all_prev if r == 1) / len(all_prev) if len(all_prev) >= 2 else 0.5
            feat["alt_distance_avg"] = sum(alt_distances) / len(alt_distances) if alt_distances else 0

            # === REGIME FEATURES — multi-hour context ===
            # 4h return: how much price moved in last 4 hours
            hr4_candles = df_15m[df_15m.index < ws_n]
            if len(hr4_candles) >= 16:  # 16 x 15m = 4 hours
                c_now = float(hr4_candles.iloc[-1]["close"])
                c_4h = float(hr4_candles.iloc[-16]["close"])
                feat["return_4h"] = (c_now - c_4h) / c_4h * 100
            else:
                feat["return_4h"] = 0

            # 12h return
            if len(hr4_candles) >= 48:  # 48 x 15m = 12 hours
                c_12h = float(hr4_candles.iloc[-48]["close"])
                feat["return_12h"] = (float(hr4_candles.iloc[-1]["close"]) - c_12h) / c_12h * 100
            else:
                feat["return_12h"] = 0

            # Price vs 1h SMA (in ATR units) — are we above or below the hourly average?
            if df_1h is not None:
                h1_filt = df_1h[df_1h.index <= ws_n]
                if len(h1_filt) >= 20 and atr > 0:
                    h1_sma = float(h1_filt["close"].rolling(20).mean().iloc[-1])
                    h1_close = float(h1_filt.iloc[-1]["close"])
                    feat["price_vs_sma_1h"] = (h1_close - h1_sma) / atr
                else:
                    feat["price_vs_sma_1h"] = 0
            else:
                feat["price_vs_sma_1h"] = 0

            # Lower lows on 4h chart — how many of last 3 4h candles made lower lows?
            if df_4h is not None:
                h4_filt = df_4h[df_4h.index <= ws_n]
                if len(h4_filt) >= 4:
                    ll_count = 0
                    for i in range(-3, 0):
                        if float(h4_filt.iloc[i]["low"]) < float(h4_filt.iloc[i-1]["low"]):
                            ll_count += 1
                    feat["lower_lows_4h"] = ll_count
                else:
                    feat["lower_lows_4h"] = 0
            else:
                feat["lower_lows_4h"] = 0

            # Trend strength: 4h close vs 4h SMA (signed, in ATR units)
            if df_4h is not None:
                h4_filt2 = df_4h[df_4h.index <= ws_n]
                if len(h4_filt2) >= 10 and atr > 0:
                    h4_sma = float(h4_filt2["close"].rolling(10).mean().iloc[-1])
                    h4_close = float(h4_filt2.iloc[-1]["close"])
                    feat["trend_strength"] = (h4_close - h4_sma) / atr
                else:
                    feat["trend_strength"] = 0
            else:
                feat["trend_strength"] = 0

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue
            rows.append({**feat, "label": label, "ts": close_dt})

        # Merge synthetic data if available
        if _args.synthetic:
            syn_path = Path("data/store/synthetic_labels.pkl")
            if syn_path.exists():
                syn_data = pickle.load(open(syn_path, "rb"))
                if target_asset in syn_data:
                    syn_asset = syn_data[target_asset]
                    syn_samples = syn_asset["samples"]
                    syn_15m = syn_asset["15m"]
                    syn_1h = syn_asset["1h"]
                    syn_4h = syn_asset["4h"]

                    syn_atr_s = syn_15m["atr"].dropna()
                    syn_ar20 = syn_atr_s.rolling(20)
                    syn_atr_p = ((syn_atr_s - syn_ar20.min()) / (syn_ar20.max() - syn_ar20.min())).fillna(0.5)

                    syn_count = 0
                    for s in syn_samples:
                        idx = s["_15m_idx"]
                        if idx < 20 or idx >= len(syn_15m) - 1:
                            continue
                        prev = syn_15m.iloc[idx]
                        atr_syn = float(prev.get("atr", 0))
                        if pd.isna(atr_syn) or atr_syn <= 0:
                            continue
                        ws_syn = syn_15m.index[idx + 1]  # window start

                        # Build features same as Kalshi path
                        kx_syn = {"strike_delta": 0, "strike_trend_3": 0}

                        # ATR percentile
                        pa_syn = syn_atr_p[syn_atr_p.index < ws_syn]
                        apv_syn = float(pa_syn.iloc[-1]) if len(pa_syn) > 0 else 0.5

                        # Regime features
                        prev_c_syn = syn_15m[syn_15m.index < ws_syn]
                        if len(prev_c_syn) >= 16:
                            kx_syn["return_4h"] = (float(prev_c_syn.iloc[-1]["close"]) - float(prev_c_syn.iloc[-16]["close"])) / float(prev_c_syn.iloc[-16]["close"]) * 100
                        else:
                            kx_syn["return_4h"] = 0
                        if len(prev_c_syn) >= 48:
                            kx_syn["return_12h"] = (float(prev_c_syn.iloc[-1]["close"]) - float(prev_c_syn.iloc[-48]["close"])) / float(prev_c_syn.iloc[-48]["close"]) * 100
                        else:
                            kx_syn["return_12h"] = 0
                        if syn_1h is not None:
                            h1f = syn_1h[syn_1h.index <= ws_syn]
                            if len(h1f) >= 20 and atr_syn > 0:
                                kx_syn["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr_syn
                            else:
                                kx_syn["price_vs_sma_1h"] = 0
                        else:
                            kx_syn["price_vs_sma_1h"] = 0
                        if syn_4h is not None:
                            h4f = syn_4h[syn_4h.index <= ws_syn]
                            if len(h4f) >= 4:
                                kx_syn["lower_lows_4h"] = sum(1 for i in range(-3, 0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i-1]["low"]))
                            else:
                                kx_syn["lower_lows_4h"] = 0
                            if len(h4f) >= 10 and atr_syn > 0:
                                kx_syn["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr_syn
                            else:
                                kx_syn["trend_strength"] = 0
                        else:
                            kx_syn["lower_lows_4h"] = 0
                            kx_syn["trend_strength"] = 0

                        # Confluence defaults for synthetic (no cross-asset data)
                        kx_syn["alt_rsi_avg"] = 50
                        kx_syn["alt_rsi_1h_avg"] = 50
                        kx_syn["alt_momentum_align"] = 0
                        kx_syn["alt_distance_avg"] = 0

                        feat_syn = build_features(prev, syn_1h, syn_4h, ws_syn,
                                                  s["distance_from_strike"],
                                                  kalshi_extra=kx_syn, atr_pctile_val=apv_syn)
                        if not feat_syn:
                            continue
                        if any(pd.isna(v) or np.isinf(v) for v in feat_syn.values()):
                            continue
                        # Ensure timezone-naive timestamp for sorting compatibility
                        syn_ts = s["ts"]
                        if hasattr(syn_ts, 'tzinfo') and syn_ts.tzinfo is None:
                            syn_ts = syn_ts.replace(tzinfo=timezone.utc)
                        elif not hasattr(syn_ts, 'tzinfo'):
                            syn_ts = pd.Timestamp(syn_ts, tz='UTC')
                        rows.append({**feat_syn, "label": s["label"], "ts": syn_ts})
                        syn_count += 1

                    print(f"  + {syn_count} synthetic samples added")

        df_all = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        print(f"  Total samples: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

        split = int(len(df_all) * 0.7)
        df_train = df_all.iloc[:split]
        df_test = df_all.iloc[split:]
        y_train = df_train["label"].values
        y_test = df_test["label"].values

        print(f"  Train: {len(df_train)} ({df_train['ts'].iloc[0].strftime('%m/%d')}-{df_train['ts'].iloc[-1].strftime('%m/%d')})")
        print(f"  Test:  {len(df_test)} ({df_test['ts'].iloc[0].strftime('%m/%d')}-{df_test['ts'].iloc[-1].strftime('%m/%d')})")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[ALL_FEATURES].values)
        model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        model.fit(X_train, y_train)

        # Top features
        print(f"\n  Top 10 features:")
        for name, coef in sorted(zip(ALL_FEATURES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:10]:
            marker = " [CONFLUENCE]" if name in CONFLUENCE_FEATURES else ""
            print(f"    {name:<24}: {coef:>+8.4f}{marker}")

        # Test
        probs = model.predict_proba(scaler.transform(df_test[ALL_FEATURES].values))[:, 1]
        print(f"\n  {'Thresh':>6} | {'Bets':>5} {'WR':>6} {'Y':>5} {'N':>5} {'Y_WR':>5} {'N_WR':>5} | {'P&L':>7}")
        print(f"  {'-' * 60}")
        for t in [55, 57, 60, 63, 65, 67, 70]:
            w, l, yw, yl, nw, nl = 0, 0, 0, 0, 0, 0
            for i in range(len(y_test)):
                p = int(probs[i] * 100)
                if p >= t: side = "YES"
                elif p <= (100 - t): side = "NO"
                else: continue
                won = (side == "YES" and y_test[i] == 1) or (side == "NO" and y_test[i] == 0)
                if won:
                    w += 1; (yw if side == "YES" else nw).__class__  # just for counting
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

        # Save
        out_path = Path(f"models/m0_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "knn": model,
                "scaler": scaler,
                "model_type": "per_asset_confluence",
                "feature_names": ALL_FEATURES,
                "asset": target_asset,
                "training_samples": len(df_train),
                "base_rate": float(y_train.mean()),
            }, f)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
