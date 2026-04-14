#!/usr/bin/env python3
"""Backtest matching EXACT live trading flow with per-asset models.

Uses per-asset M0 + M10 models with cross-asset confluence and 5m intra-window features.
Caches fetched data to disk for fast re-runs.

Usage:
    ./venv/bin/python scripts/backtest_kalshi_labels.py [--days 60]
    ./venv/bin/python scripts/backtest_kalshi_labels.py --days 60 --no-cache
"""
import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID

SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

STARTING_BALANCE = 100.0
RISK_PCT = 0.05
MAX_CONTRACTS = 100
ENTRY_CENTS = 60
MAX_PER_WINDOW = 3
M10_EXIT_CENTS = 10
DAILY_LOSS_CAP = 0.20

# Per-asset thresholds (must match daemon)
M0_THRESHOLDS = {"BTC": 57, "ETH": 60, "SOL": 60, "XRP": 57}
M10_THRESHOLDS = {"BTC": 60, "ETH": 60, "SOL": 55, "XRP": 60}

CACHE_DIR = Path("data/store/backtest_cache")


def fetch_5m_history(symbol, exchange_name, days):
    constructors = {
        "coinbase": lambda: ccxt.coinbase({"enableRateLimit": True}),
        "bitstamp": lambda: ccxt.bitstamp({"enableRateLimit": True}),
    }
    ex = constructors[exchange_name]()
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    candle_ms = 300000
    while since < now_ms:
        try:
            candles = ex.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            if not candles:
                since += 1000 * candle_ms; time.sleep(0.3); continue
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.index = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop(columns=["timestamp"])
            all_frames.append(df)
            since = int(candles[-1][0]) + candle_ms
            time.sleep(0.3)
        except Exception:
            since += 500 * candle_ms; time.sleep(1)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames)[~pd.concat(all_frames).index.duplicated()].sort_index()


def fetch_candles(fetcher, symbol, timeframe, days):
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    batch_size = 300
    tf_ms = {"15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms; time.sleep(0.5); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.5)
        except Exception:
            since += batch_size * candle_ms; time.sleep(2)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames)[~pd.concat(all_frames).index.duplicated()].sort_index()


def get_avg_price(cb_5m, bs_5m, target_time, field="open"):
    prices = []
    for df_5m in [cb_5m, bs_5m]:
        if df_5m.empty: continue
        mask = (df_5m.index >= target_time) & (df_5m.index < target_time + timedelta(minutes=5))
        if mask.sum() > 0:
            prices.append(float(df_5m[mask].iloc[0][field]))
        else:
            before = df_5m[df_5m.index <= target_time]
            if len(before) > 0 and (target_time - before.index[-1]).total_seconds() < 600:
                prices.append(float(before.iloc[-1]["close"]))
    if not prices: return None
    return sum(prices) / len(prices)


def build_features(prev, df_1h, df_4h, ws_naive, distance, *,
                    kalshi_extra=None, atr_pctile_val=0.5):
    """Build feature vector from previous 15m candle + 1h/4h + extras."""
    sma_val = float(prev.get("sma_20", 0))
    adx_val = float(prev.get("adx", 20))
    close_val = float(prev.get("close", 0))
    ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
    pve = float(prev.get("price_vs_ema", 0))
    hr = float(prev.get("hourly_return", 0))
    if pd.isna(pve) or np.isinf(pve): pve = 0
    if pd.isna(hr) or np.isinf(hr): hr = 0

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
        # Use '<' not '<=': the 1h candle indexed at ws_naive is the IN-PROGRESS
        # candle at decision time (its full hour of data wasn't available yet
        # in a live scenario). Including it would leak future data.
        m1h = df_1h[df_1h.index < ws_naive]
        if len(m1h) >= 20:
            feat["rsi_1h"] = float(m1h.iloc[-1].get("rsi", 50))
            feat["macd_1h"] = float(m1h.iloc[-1].get("macd_hist", 0))
    if df_4h is not None:
        m4h = df_4h[df_4h.index < ws_naive]
        if len(m4h) >= 10:
            feat["rsi_4h"] = float(m4h.iloc[-1].get("rsi", 50))

    # Strike/time/technical
    kx = kalshi_extra or {}
    feat["strike_delta"] = kx.get("strike_delta", 0.0)
    feat["strike_trend_3"] = kx.get("strike_trend_3", 0.0)
    hour = ws_naive.hour if hasattr(ws_naive, 'hour') else 12
    feat["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    feat["rsi_alignment"] = (1 if feat["rsi_1h"] >= 50 else -1) * (1 if feat["rsi_4h"] >= 50 else -1)
    feat["atr_percentile"] = atr_pctile_val
    feat["rsi_15m"] = float(prev.get("rsi", 50))
    bb_upper = float(prev.get("bb_upper", 0))
    bb_lower = float(prev.get("bb_lower", 0))
    bb_mid = float(prev.get("sma_20", 0))
    feat["bbw"] = ((bb_upper - bb_lower) / bb_mid * 100) if bb_mid > 0 else 0

    # Confluence
    feat["alt_rsi_avg"] = kx.get("alt_rsi_avg", 50)
    feat["alt_rsi_1h_avg"] = kx.get("alt_rsi_1h_avg", 50)
    feat["alt_momentum_align"] = kx.get("alt_momentum_align", 0)
    feat["alt_distance_avg"] = kx.get("alt_distance_avg", 0)

    # Regime features
    feat["return_4h"] = kx.get("return_4h", 0)
    feat["return_12h"] = kx.get("return_12h", 0)
    feat["price_vs_sma_1h"] = kx.get("price_vs_sma_1h", 0)
    feat["lower_lows_4h"] = kx.get("lower_lows_4h", 0)
    feat["trend_strength"] = kx.get("trend_strength", 0)

    # Interaction features
    feat["pve_x_trend"] = feat["price_vs_ema"] * feat["trend_strength"]
    feat["pve_x_return12h"] = feat["price_vs_ema"] * feat["return_12h"]
    feat["slope_x_trend"] = feat.get("ema_slope", 0) * feat["trend_strength"]
    feat["slope_x_return12h"] = feat.get("ema_slope", 0) * feat["return_12h"]
    # RSI-centered × trend interactions
    feat["rsi1h_x_r12h"] = (feat.get("rsi_1h", 50) - 50) * feat["return_12h"]
    feat["rsi4h_x_r12h"] = (feat.get("rsi_4h", 50) - 50) * feat["return_12h"]
    feat["rsi1h_x_r4h"] = (feat.get("rsi_1h", 50) - 50) * feat.get("return_4h", 0)
    feat["dist_x_r12h"] = feat.get("distance_from_strike", 0) * feat["return_12h"]

    if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
        return None
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=0, help="Override M0 threshold (0=per-asset)")
    parser.add_argument("--m10-threshold", type=int, default=0, help="Override M10 threshold (0=per-asset)")
    parser.add_argument("--exit-min", type=int, default=10)
    parser.add_argument("--exit-max", type=int, default=25)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    # Load per-asset models
    m0_models, m10_models = {}, {}
    for ac in ["btc", "eth", "sol", "xrp"]:
        with open(f"models/m0_{ac}.pkl", "rb") as f:
            d = pickle.load(f)
            m0_models[ac.upper()] = (d["knn"], d["scaler"], d["feature_names"])
        with open(f"models/m10_{ac}.pkl", "rb") as f:
            d = pickle.load(f)
            m10_models[ac.upper()] = (d.get("model") or d.get("knn"), d["scaler"], d["feature_names"])

    print("=" * 80)
    print(f"BACKTEST — Per-Asset Models ({args.days} days)")
    print(f"M0: per-asset with confluence | M10: per-asset with 5m intra-window")
    print(f"Entry: {ENTRY_CENTS}c | Exit: random {args.exit_min}-{args.exit_max}c")
    print("=" * 80)

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Check cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"bt_{args.days}d"
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists() and not args.no_cache:
        print("\nLoading cached data...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        kalshi_markets = cached["kalshi_markets"]
        five_m = cached["five_m"]
        ind_data = cached["ind_data"]
        settlement_by_time = cached["settlement_by_time"]
        for a in SERIES: print(f"  {a}: {len(kalshi_markets[a])} markets (cached)")
    else:
        print("\nFetching Kalshi settlements...")
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
            filtered = [m for m in markets if m.get("close_time") and
                datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")) >= cutoff]
            kalshi_markets[asset] = sorted(filtered, key=lambda x: x.get("close_time", ""))
            print(f"  {asset}: {len(filtered)} markets")

        settlement_by_time = {}
        for asset in SERIES:
            settlement_by_time[asset] = {}
            for mk in kalshi_markets[asset]:
                ct = mk.get("close_time", "")
                r = mk.get("result", "")
                if ct and r: settlement_by_time[asset][ct] = 1 if r == "yes" else 0

        print("\nFetching 5m candles...")
        five_m = {}
        for asset, sym in ASSETS_SYMS.items():
            five_m[asset] = {}
            for ex in ["coinbase", "bitstamp"]:
                print(f"  {asset} {ex}...", end=" ", flush=True)
                df = fetch_5m_history(sym, ex, args.days + 5)
                five_m[asset][ex] = df
                print(f"{len(df)} candles")

        print("\nFetching indicators...")
        ind_data = {}
        for asset, sym in ASSETS_SYMS.items():
            df_15m = add_indicators(fetch_candles(fetcher, sym, "15m", args.days + 30))
            df_1h = add_indicators(fetch_candles(fetcher, sym, "1h", args.days + 30))
            df_4h = add_indicators(fetch_candles(fetcher, sym, "4h", args.days + 30))
            pct = df_15m["close"].pct_change()
            df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
            df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
            df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
            df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
            df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100
            ind_data[asset] = {"15m": df_15m, "1h": df_1h, "4h": df_4h}
            print(f"  {asset}: {len(df_15m)} 15m candles")

        # Save cache
        print(f"\nSaving cache to {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump({"kalshi_markets": kalshi_markets, "five_m": five_m,
                         "ind_data": ind_data, "settlement_by_time": settlement_by_time}, f)

    # Score
    print("\nScoring with per-asset models...")
    all_assets = list(SERIES.keys())
    signals = []

    for target in all_assets:
        alts = [a for a in all_assets if a != target]
        mkts = kalshi_markets[target]
        df15 = ind_data[target]["15m"]
        df1h = ind_data[target]["1h"]
        df4h = ind_data[target]["4h"]
        cb5 = five_m[target]["coinbase"]
        bs5 = five_m[target]["bitstamp"]
        atr_s = df15["atr"].dropna()
        ar20 = atr_s.rolling(20)
        atr_p = ((atr_s - ar20.min()) / (ar20.max() - ar20.min())).fillna(0.5)

        m0m, m0s, m0f = m0_models[target]
        m10m, m10s, m10f = m10_models[target]
        m0_thresh = args.threshold if args.threshold > 0 else M0_THRESHOLDS[target]
        m10_thresh = args.m10_threshold if args.m10_threshold > 0 else M10_THRESHOLDS[target]

        ct_count = 0
        for mi, mk in enumerate(mkts):
            strike = float(mk.get("floor_strike") or 0)
            result = mk.get("result", "")
            ct = mk.get("close_time", "")
            if not strike or not result or not ct: continue
            label = 1 if result == "yes" else 0
            cdt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            ws = cdt - timedelta(minutes=15)
            wsn = ws.replace(tzinfo=None)

            p0 = get_avg_price(cb5, bs5, wsn, "open")
            if not p0: continue
            pc = df15[df15.index < wsn]
            if len(pc) < 20: continue
            prev = pc.iloc[-1]
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0: continue

            # Strike lookback
            pstrikes = []
            for lb in range(1, 4):
                if mi - lb >= 0:
                    v = float(mkts[mi - lb].get("floor_strike") or 0)
                    if v: pstrikes.append(v)
            sd = (strike - pstrikes[0]) / atr if pstrikes and atr > 0 else 0
            if len(pstrikes) >= 2 and atr > 0:
                ds = [strike - pstrikes[0]]
                for k in range(len(pstrikes) - 1): ds.append(pstrikes[k] - pstrikes[k + 1])
                st3 = sum(d / atr for d in ds) / len(ds)
            else: st3 = sd

            kx = {"strike_delta": sd, "strike_trend_3": st3}
            pav = atr_p[atr_p.index < wsn]
            apv = float(pav.iloc[-1]) if len(pav) > 0 else 0.5

            # Confluence
            ar15, ar1h, amom, adist = [], [], [], []
            for alt in alts:
                a15 = ind_data[alt]["15m"]
                a1h_df = ind_data[alt]["1h"]
                a15f = a15[a15.index < wsn]
                if len(a15f) >= 2:
                    rv = float(a15f.iloc[-1].get("rsi", 50))
                    ar15.append(rv); amom.append(1 if rv >= 50 else -1)
                if a1h_df is not None:
                    a1f = a1h_df[a1h_df.index <= wsn]
                    if len(a1f) >= 2: ar1h.append(float(a1f.iloc[-1].get("rsi", 50)))
                altpx = get_avg_price(five_m[alt]["coinbase"], five_m[alt]["bitstamp"], wsn, "open")
                if altpx and len(a15f) >= 2:
                    aa = float(a15f.iloc[-1].get("atr", 0))
                    for amk in kalshi_markets[alt]:
                        if amk.get("close_time") == ct:
                            ast_v = float(amk.get("floor_strike") or 0)
                            if ast_v and aa > 0: adist.append((altpx - ast_v) / aa)
                            break
            kx["alt_rsi_avg"] = sum(ar15) / len(ar15) if ar15 else 50
            kx["alt_rsi_1h_avg"] = sum(ar1h) / len(ar1h) if ar1h else 50
            kx["alt_momentum_align"] = sum(amom) if amom else 0
            kx["alt_distance_avg"] = sum(adist) / len(adist) if adist else 0

            # Regime features
            if len(pc) >= 16:
                kx["return_4h"] = (float(pc.iloc[-1]["close"]) - float(pc.iloc[-16]["close"])) / float(pc.iloc[-16]["close"]) * 100
            else: kx["return_4h"] = 0
            if len(pc) >= 48:
                kx["return_12h"] = (float(pc.iloc[-1]["close"]) - float(pc.iloc[-48]["close"])) / float(pc.iloc[-48]["close"]) * 100
            else: kx["return_12h"] = 0
            if df1h is not None:
                # '<' drops the in-progress candle at wsn (see build_features)
                h1f = df1h[df1h.index < wsn]
                if len(h1f) >= 20 and atr > 0:
                    kx["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
                else: kx["price_vs_sma_1h"] = 0
            else: kx["price_vs_sma_1h"] = 0
            if df4h is not None:
                h4f = df4h[df4h.index < wsn]
                if len(h4f) >= 4:
                    kx["lower_lows_4h"] = sum(1 for i in range(-3,0) if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i-1]["low"]))
                else: kx["lower_lows_4h"] = 0
                if len(h4f) >= 10 and atr > 0:
                    kx["trend_strength"] = (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1])) / atr
                else: kx["trend_strength"] = 0
            else: kx["lower_lows_4h"] = 0; kx["trend_strength"] = 0

            # M0
            f0 = build_features(prev, df1h, df4h, wsn, (p0 - strike) / atr,
                                kalshi_extra=kx, atr_pctile_val=apv)
            if not f0: continue
            X0 = np.array([f0.get(f, 0) for f in m0f]).reshape(1, -1)
            # XGBoost: predict on raw features; LogReg: predict on scaled
            is_tree = hasattr(m0m, 'get_booster') or 'XGB' in type(m0m).__name__
            X0_pred = X0 if is_tree else m0s.transform(X0)
            prob = float(m0m.predict_proba(X0_pred)[0][1])
            pct_v = int(prob * 100)
            if pct_v >= m0_thresh: side = "yes"
            elif pct_v <= (100 - m0_thresh): side = "no"
            else: continue

            # M10
            m10x = False
            exit_price = random.randint(args.exit_min, args.exit_max)
            min10 = wsn + timedelta(minutes=10)
            min5 = wsn + timedelta(minutes=5)
            p10 = get_avg_price(cb5, bs5, min10, "open")
            if p10:
                f10 = build_features(prev, df1h, df4h, wsn, (p10 - strike) / atr,
                                     kalshi_extra=kx, atr_pctile_val=apv)
                if f10:
                    # 5m intra-window features
                    c1p, c2p = [], []
                    for df_5m in [cb5, bs5]:
                        if df_5m.empty: continue
                        m1 = (df_5m.index >= wsn) & (df_5m.index < min5)
                        if m1.sum() > 0: c1p.append(df_5m[m1].iloc[0])
                        m2 = (df_5m.index >= min5) & (df_5m.index < min10)
                        if m2.sum() > 0: c2p.append(df_5m[m2].iloc[0])
                    if c1p and c2p and atr > 0:
                        c1o = sum(float(c["open"]) for c in c1p) / len(c1p)
                        c1c = sum(float(c["close"]) for c in c1p) / len(c1p)
                        c1h = sum(float(c["high"]) for c in c1p) / len(c1p)
                        c1l = sum(float(c["low"]) for c in c1p) / len(c1p)
                        c1v = sum(float(c["volume"]) for c in c1p) / len(c1p)
                        c2c = sum(float(c["close"]) for c in c2p) / len(c2p)
                        c2h = sum(float(c["high"]) for c in c2p) / len(c2p)
                        c2l = sum(float(c["low"]) for c in c2p) / len(c2p)
                        c2v = sum(float(c["volume"]) for c in c2p) / len(c2p)
                        f10["price_move_atr"] = (c2c - c1o) / atr
                        f10["candle1_range_atr"] = (c1h - c1l) / atr
                        f10["candle2_range_atr"] = (c2h - c2l) / atr
                        f10["momentum_shift"] = (c2c - c1c) / atr
                        f10["volume_acceleration"] = c2v / c1v if c1v > 0 else 1.0
                    else:
                        for k in ["price_move_atr", "candle1_range_atr", "candle2_range_atr", "momentum_shift"]:
                            f10[k] = 0
                        f10["volume_acceleration"] = 1.0

                    X10 = np.array([f10.get(f, 0) for f in m10f]).reshape(1, -1)
                    is_tree_m10 = hasattr(m10m, 'get_booster') or 'XGB' in type(m10m).__name__
                    X10_pred = X10 if is_tree_m10 else m10s.transform(X10)
                    mp = float(m10m.predict_proba(X10_pred)[0][1])
                    mpc = int(mp * 100)
                    ms = "yes" if mpc >= m10_thresh else "no" if mpc <= (100 - m10_thresh) else "skip"
                    if ms != "skip" and ms != side:
                        m10x = True

            won = (side == "yes" and label == 1) or (side == "no" and label == 0)
            outcome = "PL" if m10x else ("WIN" if won else "LOSS")
            signals.append({
                "ts": cdt, "asset": target, "side": side.upper(), "m0_prob": prob,
                "m10_exit": m10x, "exit_price": exit_price,
                "outcome": outcome, "won": won, "won_settlement": won,
            })
            ct_count += 1
        print(f"  {target}: {ct_count} signals")

    df_sig = pd.DataFrame(signals).sort_values("ts").reset_index(drop=True)
    total = len(df_sig)
    if total == 0:
        print("No signals generated"); return

    # Results
    wins = (df_sig["outcome"] == "WIN").sum()
    losses = (df_sig["outcome"] == "LOSS").sum()
    pls = (df_sig["outcome"] == "PL").sum()
    entry_wr = df_sig["won_settlement"].mean() * 100
    held_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"ENTRY-ONLY (M0, no M10 exit)")
    print(f"{'=' * 80}")
    print(f"  WR: {entry_wr:.1f}% ({df_sig['won_settlement'].sum()}W / {(~df_sig['won_settlement']).sum()}L)")
    yes_n = len(df_sig[df_sig["side"] == "YES"])
    no_n = len(df_sig[df_sig["side"] == "NO"])
    y_wr = df_sig[df_sig["side"] == "YES"]["won_settlement"].mean() * 100 if yes_n > 0 else 0
    n_wr = df_sig[df_sig["side"] == "NO"]["won_settlement"].mean() * 100 if no_n > 0 else 0
    print(f"  YES: {y_wr:.1f}% ({yes_n}) | NO: {n_wr:.1f}% ({no_n})")

    print(f"\n{'=' * 80}")
    print(f"WITH M10 EXIT")
    print(f"{'=' * 80}")
    print(f"  Held: {wins + losses} | Exits: {pls} | W:{wins} L:{losses} PL:{pls}")
    print(f"  Held WR: {held_wr:.1f}%")
    if pls > 0:
        ex = df_sig[df_sig["m10_exit"]]
        good = (~ex["won_settlement"]).sum()
        bad = ex["won_settlement"].sum()
        print(f"  M10: {good} good / {bad} bad ({good / (good + bad) * 100:.0f}% acc)" if (good + bad) > 0 else "")

    # P&L with daily loss cap
    balance = STARTING_BALANCE
    peak = balance
    max_dd = max_dd_dollars = max_dd_peak = max_dd_trough = 0
    bets_placed = bets_skipped = 0
    pnl_wins = pnl_losses = pnl_exits = 0
    current_day = None
    day_start = balance
    day_halted = False

    for _, group in df_sig.groupby(df_sig["ts"].dt.floor("15min")):
        window_bets = group.nlargest(MAX_PER_WINDOW, "m0_prob")
        for _, row in window_bets.iterrows():
            bet_day = row["ts"].strftime("%m/%d")
            if bet_day != current_day:
                current_day = bet_day; day_start = balance; day_halted = False
            if day_halted: bets_skipped += 1; continue
            if day_start > 0 and (day_start - balance) / day_start >= DAILY_LOSS_CAP:
                day_halted = True; bets_skipped += 1; continue

            risk = balance * RISK_PCT
            contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
            cost = contracts * (ENTRY_CENTS / 100)
            if cost > balance: continue
            bets_placed += 1

            if row["outcome"] == "WIN":
                p = contracts * ((100 - ENTRY_CENTS) / 100); balance += p; pnl_wins += p
            elif row["outcome"] == "LOSS":
                balance -= cost; pnl_losses += cost
            elif row["outcome"] == "PL":
                ep = int(row.get("exit_price", M10_EXIT_CENTS))
                pnl_per = (ep - ENTRY_CENTS) / 100
                exit_pnl = contracts * pnl_per
                balance += exit_pnl
                pnl_exits += abs(exit_pnl) if exit_pnl < 0 else -exit_pnl

            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            dd_d = peak - balance
            if dd > max_dd:
                max_dd = dd; max_dd_dollars = dd_d; max_dd_peak = peak; max_dd_trough = balance

    ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    net_pnl = pnl_wins - pnl_losses - pnl_exits
    print(f"\n{'=' * 80}")
    print(f"P&L: ${STARTING_BALANCE} start, {RISK_PCT*100:.0f}% risk, max {MAX_PER_WINDOW}/window")
    print(f"{'=' * 80}")
    print(f"  Bets: {bets_placed} (skipped {bets_skipped} from daily cap)")
    print(f"  Gross won:   ${pnl_wins:+,.2f}")
    print(f"  Gross lost:  ${-pnl_losses:+,.2f}")
    print(f"  Early exits: ${-pnl_exits:+,.2f}")
    print(f"  Net P&L:     ${net_pnl:+,.2f}")
    print(f"  Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
    print(f"  Max DD: {max_dd:.1f}% (${max_dd_dollars:,.2f} — ${max_dd_peak:,.2f} to ${max_dd_trough:,.2f})")

    # Daily
    def sig_pnl(row):
        if row["outcome"] == "WIN": return (100 - ENTRY_CENTS) / 100
        elif row["outcome"] == "LOSS": return -ENTRY_CENTS / 100
        else: return (row.get("exit_price", M10_EXIT_CENTS) - ENTRY_CENTS) / 100
    df_sig["flat_pnl"] = df_sig.apply(sig_pnl, axis=1)
    df_sig["date"] = df_sig["ts"].apply(lambda t: t.strftime("%m/%d"))
    daily = df_sig.groupby("date").agg(
        bets=("outcome", "count"),
        wins=("outcome", lambda x: (x == "WIN").sum()),
        losses=("outcome", lambda x: (x == "LOSS").sum()),
        exits=("outcome", lambda x: (x == "PL").sum()),
        pnl=("flat_pnl", "sum"),
    )
    daily["wr"] = daily["wins"] / (daily["wins"] + daily["losses"]) * 100

    print(f"\n{'Date':<8} {'Bets':>5} {'W':>3} {'L':>3} {'PL':>3} {'WR':>6} {'P&L':>8}")
    print("-" * 42)
    for date, row in daily.iterrows():
        held = int(row["wins"] + row["losses"])
        wr_str = f"{row['wr']:.0f}%" if held > 0 else "  --"
        print(f"{date:<8} {int(row['bets']):>5} {int(row['wins']):>3} {int(row['losses']):>3} "
              f"{int(row['exits']):>3} {wr_str:>6} ${row['pnl']:>+6.1f}")

    wd = (daily["pnl"] > 0).sum()
    ld = (daily["pnl"] < 0).sum()
    print(f"\n  Winning: {wd} | Losing: {ld}")

    # Per-asset
    print(f"\n  Per-asset:")
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        ad = df_sig[df_sig["asset"] == a]
        if len(ad) == 0: continue
        aw = (ad["outcome"] == "WIN").sum()
        al = (ad["outcome"] == "LOSS").sum()
        ap = (ad["outcome"] == "PL").sum()
        awr = ad["won_settlement"].mean() * 100
        ahwr = aw / (aw + al) * 100 if (aw + al) > 0 else 0
        print(f"    {a}: {len(ad)} sig, Entry:{awr:.0f}%, Held:{ahwr:.0f}%, W:{aw} L:{al} PL:{ap}")


if __name__ == "__main__":
    main()
