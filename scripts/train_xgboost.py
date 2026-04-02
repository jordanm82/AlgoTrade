#!/usr/bin/env python3
"""Train XGBoost model alongside LogReg for comparison.

Strict 3-way data split:
  TRAIN:    days 30-97 (oldest 70%) — model learns patterns
  VALIDATE: days 10-30 (15%) — tune hyperparameters, prevent overfit
  TEST:     days 0-10 (newest 15%) — final honest WR, never seen by model

Both models trained on identical data, evaluated on identical test set.
XGBoost saved separately — not used for live decisions until validated.

Usage:
    ./venv/bin/python scripts/train_xgboost.py
"""
import pickle
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import ccxt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

FEATURE_NAMES = [
    "macd_15m", "norm_return", "ema_slope", "roc_5",
    "macd_1h", "price_vs_ema", "hourly_return", "trend_direction",
    "vol_ratio", "adx", "rsi_1h", "rsi_4h", "distance_from_strike",
]


def fetch_kalshi_settlements(client):
    all_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for _ in range(30):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = client._get("/trade-api/v2/markets", params)
            batch = resp.get("markets", [])
            cursor = resp.get("cursor", "")
            markets.extend(batch)
            if not batch or not cursor:
                break
        all_markets[asset] = markets
        print(f"  {asset}: {len(markets)} settled markets")
    return all_markets


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


def build_samples(kalshi_markets, five_m_data, indicator_data):
    """Build feature/label samples from Kalshi settlements."""
    all_rows = []
    for asset in SERIES:
        markets = kalshi_markets.get(asset, [])
        df_15m = indicator_data[asset]["15m"]
        df_1h = indicator_data[asset]["1h"]
        df_4h = indicator_data[asset]["4h"]
        cb_5m = five_m_data[asset].get("coinbase", pd.DataFrame())
        bs_5m = five_m_data[asset].get("bitstamp", pd.DataFrame())

        for m in markets:
            strike = float(m.get("floor_strike") or 0)
            result = m.get("result", "")
            close_time = m.get("close_time", "")
            if not strike or not result or not close_time:
                continue

            label = 1 if result == "yes" else 0
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = close_dt - timedelta(minutes=15)
            ws_naive = window_start.replace(tzinfo=None)

            # Minute-0 price: 5m candle open averaged
            prices = []
            for df_5m in [cb_5m, bs_5m]:
                if df_5m.empty:
                    continue
                mask = (df_5m.index >= ws_naive) & (df_5m.index < ws_naive + timedelta(minutes=5))
                if mask.sum() > 0:
                    prices.append(float(df_5m[mask].iloc[0]["open"]))
                else:
                    before = df_5m[df_5m.index <= ws_naive]
                    if len(before) > 0 and (ws_naive - before.index[-1]).total_seconds() < 600:
                        prices.append(float(before.iloc[-1]["close"]))
            if not prices:
                continue
            price_at_min0 = sum(prices) / len(prices)

            prev = df_15m[df_15m.index < ws_naive]
            if len(prev) < 20:
                continue
            pr = prev.iloc[-1]
            atr = float(pr.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            distance = (price_at_min0 - strike) / atr
            sma_val = float(pr.get("sma_20", 0))
            adx_val = float(pr.get("adx", 20))
            close_val = float(pr.get("close", 0))
            ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
            pve = float(pr.get("price_vs_ema", 0))
            hr = float(pr.get("hourly_return", 0))
            if pd.isna(pve) or np.isinf(pve): pve = 0
            if pd.isna(hr) or np.isinf(hr): hr = 0

            feat = {
                "macd_15m": float(pr.get("macd_hist", 0)),
                "norm_return": float(pr.get("norm_return", 0)) if pd.notna(pr.get("norm_return")) else 0,
                "ema_slope": float(pr.get("ema_slope", 0)) if pd.notna(pr.get("ema_slope")) else 0,
                "roc_5": float(pr.get("roc_5", 0)),
                "macd_1h": 0.0,
                "price_vs_ema": pve,
                "hourly_return": hr,
                "trend_direction": adx_val * ts_sign,
                "vol_ratio": float(pr.get("vol_ratio", 1)) if pd.notna(pr.get("vol_ratio")) else 1,
                "adx": adx_val,
                "rsi_1h": 50.0,
                "rsi_4h": 50.0,
                "distance_from_strike": distance,
            }

            if df_1h is not None:
                m1h = df_1h.index <= ws_naive
                if m1h.sum() >= 20:
                    r1h = df_1h.loc[m1h].iloc[-1]
                    feat["rsi_1h"] = float(r1h.get("rsi", 50))
                    feat["macd_1h"] = float(r1h.get("macd_hist", 0))
            if df_4h is not None:
                m4h = df_4h.index <= ws_naive
                if m4h.sum() >= 10:
                    feat["rsi_4h"] = float(df_4h.loc[m4h].iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            all_rows.append({
                **feat, "label": label, "ts": close_dt, "asset": asset,
            })

    return pd.DataFrame(all_rows)


def evaluate(model, scaler, X, y, name, threshold=55):
    """Evaluate a model and return stats."""
    if scaler is not None:
        X_s = scaler.transform(X)
    else:
        X_s = X
    probs = model.predict_proba(X_s)[:, 1]

    w, l, s = 0, 0, 0
    yw, yl, nw, nl = 0, 0, 0, 0

    for i in range(len(y)):
        pct = int(probs[i] * 100)
        if pct >= threshold:
            side = "YES"
        elif pct <= (100 - threshold):
            side = "NO"
        else:
            s += 1; continue

        won = (side == "YES" and y[i] == 1) or (side == "NO" and y[i] == 0)
        if won:
            w += 1
            if side == "YES": yw += 1
            else: nw += 1
        else:
            l += 1
            if side == "YES": yl += 1
            else: nl += 1

    total = w + l
    wr = w / total * 100 if total > 0 else 0
    yn = yw + yl
    nn = nw + nl
    pnl = w * 0.50 - l * 0.50
    return {"name": name, "wr": wr, "w": w, "l": l, "skip": s, "yes": yn, "no": nn,
            "y_wr": yw / yn * 100 if yn > 0 else 0,
            "n_wr": nw / nn * 100 if nn > 0 else 0,
            "pnl": pnl}


def main():
    print("=" * 80)
    print("XGBoost vs LogReg — Strict 3-Way Split")
    print("TRAIN: oldest 70% | VALIDATE: 15% | TEST: newest 15%")
    print("Same data, same features, same labels (Kalshi settlements)")
    print("=" * 80)

    # Fetch data
    print("\n[1/4] Fetching Kalshi settlements...")
    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    kalshi_markets = fetch_kalshi_settlements(client)

    print("\n[2/4] Fetching 5m candle history (Coinbase + Bitstamp)...")
    fetcher = DataFetcher()
    five_m_data = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m_data[asset] = {}
        for ex_name in ["coinbase", "bitstamp"]:
            print(f"  {asset} {ex_name}...", end=" ", flush=True)
            df = fetch_5m_history(sym, ex_name, 100)
            five_m_data[asset][ex_name] = df
            print(f"{len(df)} candles")

    print("\n[3/4] Fetching indicator data...")
    indicator_data = {}
    for asset, sym in ASSETS_SYMS.items():
        df_15m = fetch_candles(fetcher, sym, "15m", 110)
        df_1h = fetch_candles(fetcher, sym, "1h", 110)
        df_4h = fetch_candles(fetcher, sym, "4h", 110)
        df = add_indicators(df_15m)
        pct = df["close"].pct_change()
        df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
        df["ema_slope"] = df["ema_12"].pct_change(3) * 100
        df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
        df["hourly_return"] = df["close"].pct_change(4) * 100
        indicator_data[asset] = {
            "15m": df,
            "1h": add_indicators(df_1h) if not df_1h.empty else None,
            "4h": add_indicators(df_4h) if not df_4h.empty else None,
        }
        print(f"  {asset}: {len(df)} 15m candles")

    print("\n[4/4] Building samples...")
    df_all = build_samples(kalshi_markets, five_m_data, indicator_data)
    df_all = df_all.sort_values("ts").reset_index(drop=True)
    print(f"Total: {len(df_all)} | Label YES: {df_all['label'].mean():.1%}")

    # === STRICT 3-way split ===
    n = len(df_all)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    df_train = df_all.iloc[:train_end]
    df_val = df_all.iloc[train_end:val_end]
    df_test = df_all.iloc[val_end:]

    print(f"\nSplit:")
    print(f"  TRAIN:    {len(df_train)} ({df_train['ts'].iloc[0].strftime('%m/%d')} - {df_train['ts'].iloc[-1].strftime('%m/%d')})")
    print(f"  VALIDATE: {len(df_val)} ({df_val['ts'].iloc[0].strftime('%m/%d')} - {df_val['ts'].iloc[-1].strftime('%m/%d')})")
    print(f"  TEST:     {len(df_test)} ({df_test['ts'].iloc[0].strftime('%m/%d')} - {df_test['ts'].iloc[-1].strftime('%m/%d')})")

    X_train = df_train[FEATURE_NAMES].values
    y_train = df_train["label"].values
    X_val = df_val[FEATURE_NAMES].values
    y_val = df_val["label"].values
    X_test = df_test[FEATURE_NAMES].values
    y_test = df_test["label"].values

    # === Train LogReg ===
    print(f"\n{'=' * 80}")
    print("LOGISTIC REGRESSION")
    print(f"{'=' * 80}")

    lr_scaler = StandardScaler()
    X_train_lr = lr_scaler.fit_transform(X_train)
    lr_model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    lr_model.fit(X_train_lr, y_train)

    lr_train = evaluate(lr_model, lr_scaler, X_train, y_train, "LR Train")
    lr_val = evaluate(lr_model, lr_scaler, X_val, y_val, "LR Validate")
    lr_test = evaluate(lr_model, lr_scaler, X_test, y_test, "LR Test")

    print(f"  Train:    {lr_train['wr']:.1f}% ({lr_train['w']}W/{lr_train['l']}L/{lr_train['skip']}S)")
    print(f"  Validate: {lr_val['wr']:.1f}% ({lr_val['w']}W/{lr_val['l']}L/{lr_val['skip']}S)")
    print(f"  Test:     {lr_test['wr']:.1f}% ({lr_test['w']}W/{lr_test['l']}L/{lr_test['skip']}S)")
    print(f"  Test P&L: ${lr_test['pnl']:+.0f} | YES:{lr_test['y_wr']:.0f}% NO:{lr_test['n_wr']:.0f}%")

    # === Train XGBoost ===
    print(f"\n{'=' * 80}")
    print("XGBOOST")
    print(f"{'=' * 80}")

    # Tune on validation set — try multiple configs
    best_xgb = None
    best_val_wr = 0
    best_config = ""

    configs = [
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 5, "n_estimators": 150, "learning_rate": 0.05},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.03},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.1},
    ]

    print(f"\nTuning on validation set ({len(df_val)} samples)...")
    print(f"{'Config':<45} {'Train':>6} {'Val':>6} {'Gap':>5}")
    print("-" * 65)

    for cfg in configs:
        xgb = XGBClassifier(
            **cfg,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
            random_state=42,
        )
        xgb.fit(X_train, y_train)

        train_r = evaluate(xgb, None, X_train, y_train, "train")
        val_r = evaluate(xgb, None, X_val, y_val, "val")
        gap = train_r["wr"] - val_r["wr"]
        overfit = "⚠" if gap > 5 else ""

        cfg_str = f"d={cfg['max_depth']} n={cfg['n_estimators']} lr={cfg['learning_rate']}"
        print(f"  {cfg_str:<43} {train_r['wr']:>5.1f}% {val_r['wr']:>5.1f}% {gap:>+4.1f}pp {overfit}")

        if val_r["wr"] > best_val_wr:
            best_val_wr = val_r["wr"]
            best_xgb = xgb
            best_config = cfg_str

    print(f"\nBest config: {best_config} (val {best_val_wr:.1f}%)")

    # Evaluate best XGBoost on TEST set (never seen)
    xgb_train = evaluate(best_xgb, None, X_train, y_train, "XGB Train")
    xgb_val = evaluate(best_xgb, None, X_val, y_val, "XGB Validate")
    xgb_test = evaluate(best_xgb, None, X_test, y_test, "XGB Test")

    print(f"\n  Train:    {xgb_train['wr']:.1f}% ({xgb_train['w']}W/{xgb_train['l']}L/{xgb_train['skip']}S)")
    print(f"  Validate: {xgb_val['wr']:.1f}% ({xgb_val['w']}W/{xgb_val['l']}L/{xgb_val['skip']}S)")
    print(f"  Test:     {xgb_test['wr']:.1f}% ({xgb_test['w']}W/{xgb_test['l']}L/{xgb_test['skip']}S)")
    print(f"  Test P&L: ${xgb_test['pnl']:+.0f} | YES:{xgb_test['y_wr']:.0f}% NO:{xgb_test['n_wr']:.0f}%")

    # Feature importance
    print(f"\nXGBoost feature importance:")
    importances = best_xgb.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 100)
        print(f"  {name:<22}: {imp:.3f} {bar}")

    # === Head-to-head ===
    print(f"\n{'=' * 80}")
    print("HEAD TO HEAD (TEST SET — never seen by either model)")
    print(f"{'=' * 80}")
    print(f"{'':>15} {'WR':>6} {'Bets':>5} {'P&L':>7} {'YES':>4} {'NO':>4} {'Y_WR':>5} {'N_WR':>5}")
    print("-" * 55)
    for r in [lr_test, xgb_test]:
        print(f"  {r['name']:<13} {r['wr']:>5.1f}% {r['w']+r['l']:>4} ${r['pnl']:>+5.0f} {r['yes']:>4} {r['no']:>4} {r['y_wr']:>4.0f}% {r['n_wr']:>4.0f}%")

    advantage = xgb_test["wr"] - lr_test["wr"]
    if advantage > 2:
        print(f"\n  XGBoost is {advantage:+.1f}pp better on unseen data ✓")
    elif advantage < -2:
        print(f"\n  LogReg is {-advantage:+.1f}pp better — XGBoost may be overfitting ✗")
    else:
        print(f"\n  Within 2pp — no significant difference ({advantage:+.1f}pp)")

    # Save XGBoost model separately (not replacing production LogReg)
    xgb_path = Path("models/xgboost_kalshi.pkl")
    xgb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xgb_path, "wb") as f:
        pickle.dump({
            "model": best_xgb,
            "model_type": "xgboost",
            "feature_names": FEATURE_NAMES,
            "config": best_config,
            "train_wr": xgb_train["wr"],
            "val_wr": xgb_val["wr"],
            "test_wr": xgb_test["wr"],
            "training_samples": len(df_train),
        }, f)
    print(f"\nXGBoost saved to {xgb_path} (NOT production — for comparison only)")
    print(f"Production LogReg unchanged at models/knn_kalshi.pkl")


if __name__ == "__main__":
    main()
