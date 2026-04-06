#!/usr/bin/env python3
"""Train per-asset M10 exit models with cross-asset confluence features.

Same as per-asset M0 but distance computed at minute 10.

Usage:
    ./venv/bin/python scripts/train_per_asset_m10.py
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
    "strike_delta", "strike_trend_3",
    "hour_sin", "hour_cos",
    "rsi_alignment", "atr_percentile", "rsi_15m", "bbw",
]

CONFLUENCE_FEATURES = [
    "alt_rsi_avg", "alt_rsi_1h_avg", "alt_momentum_align",
    "alt_distance_avg",
]

ALL_FEATURES = BASE_FEATURES + CONFLUENCE_FEATURES


def main():
    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET M10 MODELS WITH CROSS-ASSET CONFLUENCE")
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

    print("\n[4/4] Training per-asset M10 models...")
    all_assets = list(SERIES.keys())

    for target_asset in all_assets:
        print(f"\n{'=' * 80}")
        print(f"M10 TRAINING: {target_asset}")
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

            # Distance at minute 10
            feat = build_features(prev, df_1h, df_4h, ws_n, (price - strike) / atr,
                                  kalshi_extra=kx, atr_pctile_val=apv)
            if not feat: continue

            # Confluence
            alt_rsi_15m, alt_rsi_1h, alt_momentum = [], [], []
            alt_prev_results, alt_distances = [], []
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
                if ct in settlement_by_time[alt]:
                    alt_prev_results.append(settlement_by_time[alt][ct])
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
            target_prev = prev_results[0] if prev_results else 0.5
            all_prev = [target_prev] + alt_prev_results
            feat["prev_result_consensus"] = sum(1 for r in all_prev if r == 1) / len(all_prev) if len(all_prev) >= 2 else 0.5
            feat["alt_distance_avg"] = sum(alt_distances) / len(alt_distances) if alt_distances else 0

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue
            rows.append({**feat, "label": label, "ts": close_dt})

        df_all = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
        print(f"  Samples: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

        split = int(len(df_all) * 0.7)
        df_train = df_all.iloc[:split]
        df_test = df_all.iloc[split:]
        y_train = df_train["label"].values
        y_test = df_test["label"].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[ALL_FEATURES].values)
        model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        model.fit(X_train, y_train)

        # Top features
        print(f"\n  Top 5 features:")
        for name, coef in sorted(zip(ALL_FEATURES, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:5]:
            marker = " [C]" if name in CONFLUENCE_FEATURES else ""
            print(f"    {name:<24}: {coef:>+8.4f}{marker}")

        # Threshold sweep for M10 exit decision
        probs = model.predict_proba(scaler.transform(df_test[ALL_FEATURES].values))[:, 1]
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

        out_path = Path(f"models/m10_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump({
                "model": model,
                "scaler": scaler,
                "model_type": "per_asset_m10_confluence",
                "feature_names": ALL_FEATURES,
                "asset": target_asset,
                "training_samples": len(df_train),
            }, f)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
