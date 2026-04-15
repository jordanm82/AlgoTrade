#!/usr/bin/env python3
"""Train per-asset M10 exit models with continuation de-bias controls.

Key parity/safety rules:
- M10 prices/features must use Coinbase+Bitstamp 5m candles (minute 0/5/10).
- If required intra-window candles are missing, skip the sample (never zero-fill).
- Higher-timeframe candles use strict '< window_start' filtering.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from exchange.kalshi import KalshiClient
from scripts.backtest_kalshi_labels import (
    fetch_5m_history,
    fetch_candles,
)
from strategy.m10_feature_builder import (
    build_common_feature_vector,
    compute_confluence_features,
    compute_m10_intra_from_exchange_dfs,
    get_avg_price_5m,
)

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
    "rsi1h_x_r12h", "rsi4h_x_r12h", "rsi1h_x_r4h", "dist_x_r12h",
]

ALL_FEATURES = (
    BASE_FEATURES
    + CONFLUENCE_FEATURES
    + INTRA_WINDOW_FEATURES
    + REGIME_FEATURES
    + INTERACTION_FEATURES
)

# Purge gap: 4 hours = 16 fifteen-minute windows
PURGE_GAP_SAMPLES = 16


def compute_time_decay_weights(timestamps: pd.Series, half_life_days: int = 60) -> np.ndarray:
    """Exponential decay weights — recent samples weighted more heavily."""
    latest = timestamps.max()
    days_ago = (latest - timestamps).dt.total_seconds() / 86400
    weights = np.exp(-np.log(2) * days_ago / half_life_days)
    return (weights / weights.mean()).values


def evaluate_threshold(probs: np.ndarray, y_true: np.ndarray, threshold: int) -> dict:
    """Evaluate model at confidence threshold. Returns (w/l/wr/pnl/balance stats)."""
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
            if side == "YES":
                yw += 1
            else:
                nw += 1
        else:
            l += 1
            if side == "YES":
                yl += 1
            else:
                nl += 1

    tot = w + l
    wr = (w / tot * 100) if tot > 0 else 0.0
    pnl = w * 0.40 - l * 0.60

    return {
        "w": w,
        "l": l,
        "tot": tot,
        "wr": wr,
        "pnl": pnl,
        "yw": yw,
        "yl": yl,
        "nw": nw,
        "nl": nl,
    }


def evaluate_guardrails(
    probs: np.ndarray,
    y_true: np.ndarray,
    feat_df: pd.DataFrame,
    *,
    threshold: int = 90,
    continuation_dist: float = 0.8,
    continuation_ret12: float = 0.5,
) -> dict:
    """Evaluate continuation-bias guardrails on a held-out test slice."""
    pcts = probs * 100.0
    sides = np.array(["SKIP"] * len(probs), dtype=object)
    sides[pcts >= threshold] = "YES"
    sides[pcts <= (100 - threshold)] = "NO"

    decided = sides != "SKIP"
    wins = ((sides == "YES") & (y_true == 1)) | ((sides == "NO") & (y_true == 0))
    losses = decided & (~wins)
    total = int(decided.sum())
    w = int(wins.sum())
    l = int(losses.sum())
    true_wr = (w / total * 100) if total > 0 else 0.0
    pnl = w * 0.40 - l * 0.60

    yes_mask = sides == "YES"
    no_mask = sides == "NO"
    y_tot = int(yes_mask.sum())
    n_tot = int(no_mask.sum())
    y_w = int((yes_mask & (y_true == 1)).sum())
    n_w = int((no_mask & (y_true == 0)).sum())
    y_wr = (y_w / y_tot * 100) if y_tot > 0 else 0.0
    n_wr = (n_w / n_tot * 100) if n_tot > 0 else 0.0

    if "distance_from_strike" in feat_df.columns and "return_12h" in feat_df.columns:
        dist = feat_df["distance_from_strike"].values
        ret12 = feat_df["return_12h"].values
        bull = (dist >= continuation_dist) & (ret12 >= continuation_ret12)
        bear = (dist <= -continuation_dist) & (ret12 <= -continuation_ret12)
    else:
        bull = np.zeros(len(probs), dtype=bool)
        bear = np.zeros(len(probs), dtype=bool)

    bull_n = int(bull.sum())
    bear_n = int(bear.sum())
    bull_no_rate = (float(((sides == "NO") & bull).sum()) / bull_n * 100) if bull_n > 0 else None
    bear_yes_rate = (float(((sides == "YES") & bear).sum()) / bear_n * 100) if bear_n > 0 else None

    bull_yes_decided = bull & (y_true == 1) & decided
    bull_yes_decided_n = int(bull_yes_decided.sum())
    bull_false_no = int(((sides == "NO") & bull_yes_decided).sum())
    bull_false_no_rate = (
        bull_false_no / bull_yes_decided_n * 100
        if bull_yes_decided_n > 0
        else None
    )

    return {
        "threshold": threshold,
        "bets": total,
        "wins": w,
        "losses": l,
        "true_wr": true_wr,
        "pnl": pnl,
        "yes_bets": y_tot,
        "no_bets": n_tot,
        "yes_wr": y_wr,
        "no_wr": n_wr,
        "bull_bucket_n": bull_n,
        "bear_bucket_n": bear_n,
        "bull_no_rate": bull_no_rate,
        "bear_yes_rate": bear_yes_rate,
        "bull_false_no_rate": bull_false_no_rate,
    }


def apply_continuation_weights(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    base_weights: np.ndarray,
    *,
    continuation_weight: float = 1.0,
    continuation_dist: float = 0.8,
    continuation_ret12: float = 0.5,
) -> np.ndarray:
    """Upweight continuation-consistent labels in strong-trend buckets."""
    if continuation_weight <= 1.0:
        return base_weights
    if "distance_from_strike" not in df_train.columns or "return_12h" not in df_train.columns:
        return base_weights

    w = base_weights.copy()
    dist = df_train["distance_from_strike"].values
    ret12 = df_train["return_12h"].values

    bull = (dist >= continuation_dist) & (ret12 >= continuation_ret12) & (y_train == 1)
    bear = (dist <= -continuation_dist) & (ret12 <= -continuation_ret12) & (y_train == 0)
    w[bull] *= continuation_weight
    w[bear] *= continuation_weight
    return w / w.mean()


def build_asset_dataset(
    *,
    target_asset: str,
    all_assets: list[str],
    kalshi_markets: dict,
    strike_by_time: dict,
    ind_data: dict,
    five_m: dict,
) -> pd.DataFrame:
    """Build strict point-in-time M10 dataset for one asset."""
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
        if not strike or not result or not ct:
            continue

        label = 1 if result == "yes" else 0
        close_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        ws = close_dt - timedelta(minutes=15)
        ws_n = ws.replace(tzinfo=None)

        min10 = ws_n + timedelta(minutes=10)
        price = get_avg_price_5m(cb_5m, bs_5m, min10, "open")
        if price is None:
            continue

        prev_c = df_15m[df_15m.index < ws_n]
        if len(prev_c) < 20:
            continue
        prev = prev_c.iloc[-1]
        atr = float(prev.get("atr", 0))
        if pd.isna(atr) or atr <= 0:
            continue

        prev_strikes = []
        for lb in range(1, 4):
            if mi - lb >= 0:
                ps = float(target_markets[mi - lb].get("floor_strike") or 0)
                if ps:
                    prev_strikes.append(ps)
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

        # Regime features at decision timestamp.
        hr4_candles = df_15m[df_15m.index < ws_n]
        if len(hr4_candles) >= 16:
            kx["return_4h"] = (
                (float(hr4_candles.iloc[-1]["close"]) - float(hr4_candles.iloc[-16]["close"]))
                / float(hr4_candles.iloc[-16]["close"]) * 100
            )
        else:
            kx["return_4h"] = 0

        if len(hr4_candles) >= 48:
            kx["return_12h"] = (
                (float(hr4_candles.iloc[-1]["close"]) - float(hr4_candles.iloc[-48]["close"]))
                / float(hr4_candles.iloc[-48]["close"]) * 100
            )
        else:
            kx["return_12h"] = 0

        if df_1h is not None:
            h1f = df_1h[df_1h.index < ws_n]
            if len(h1f) >= 20 and atr > 0:
                kx["price_vs_sma_1h"] = (
                    (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1]))
                    / atr
                )
            else:
                kx["price_vs_sma_1h"] = 0
        else:
            kx["price_vs_sma_1h"] = 0

        if df_4h is not None:
            h4f = df_4h[df_4h.index < ws_n]
            if len(h4f) >= 4:
                kx["lower_lows_4h"] = sum(
                    1
                    for i in range(-3, 0)
                    if float(h4f.iloc[i]["low"]) < float(h4f.iloc[i - 1]["low"])
                )
            else:
                kx["lower_lows_4h"] = 0

            if len(h4f) >= 10 and atr > 0:
                kx["trend_strength"] = (
                    (float(h4f.iloc[-1]["close"]) - float(h4f["close"].rolling(10).mean().iloc[-1]))
                    / atr
                )
            else:
                kx["trend_strength"] = 0
        else:
            kx["lower_lows_4h"] = 0
            kx["trend_strength"] = 0

        # Confluence at minute-10 anchor (shared builder path).
        kx.update(
            compute_confluence_features(
                alt_keys=alt_assets,
                ws_naive=ws_n,
                get_15m_df=lambda alt: ind_data[alt]["15m"],
                get_1h_df=lambda alt: ind_data[alt]["1h"],
                get_anchor_price=lambda alt: get_avg_price_5m(
                    five_m[alt]["coinbase"], five_m[alt]["bitstamp"], min10, "open"
                ),
                get_strike=lambda alt: strike_by_time.get(alt, {}).get(ct) or None,
            )
        )

        feat = build_common_feature_vector(
            prev,
            df_1h,
            df_4h,
            ws_n,
            (price - strike) / atr,
            kalshi_extra=kx,
            atr_pctile_val=apv,
        )
        if not feat:
            continue

        intra = compute_m10_intra_from_exchange_dfs(cb_5m, bs_5m, ws_n, atr)
        if intra is None:
            continue
        feat.update(intra)

        if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
            continue

        rows.append({**feat, "label": label, "ts": close_dt})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL,XRP")
    parser.add_argument("--continuation-weight", type=float, default=1.0)
    parser.add_argument("--continuation-dist", type=float, default=0.8)
    parser.add_argument("--continuation-ret12", type=float, default=0.5)
    parser.add_argument("--monotone-distance", action="store_true")
    parser.add_argument("--max-settlement-pages", type=int, default=30)
    parser.add_argument("--eval-threshold", type=int, default=90)
    parser.add_argument("--history-days", type=int, default=100)
    parser.add_argument("--report-json", type=str, default="")
    args = parser.parse_args()

    selected_assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    selected_assets = [a for a in selected_assets if a in SERIES]
    if not selected_assets:
        raise ValueError(
            f"No valid assets in --assets={args.assets}. Valid: {','.join(SERIES.keys())}"
        )

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()

    print("=" * 80)
    print("PER-ASSET M10 MODELS v3 — CONTINUATION-SAFE")
    print("  max_depth=4, lr=0.02, early_stop=80, subsample=0.7")
    print("  Purged walk-forward (4h gap) + time-decay weighting (60d half-life)")
    print(f"  Assets: {','.join(selected_assets)}")
    print(
        f"  Continuation weight: {args.continuation_weight:.2f} "
        f"| monotone distance: {args.monotone_distance}"
    )
    print("=" * 80)

    print("\n[1/4] Fetching settlements...")
    kalshi_markets = {}
    for asset in selected_assets:
        series = SERIES[asset]
        markets = []
        cursor = ""
        for _ in range(args.max_settlement_pages):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = client._get("/trade-api/v2/markets", params)
            batch = resp.get("markets", [])
            cursor = resp.get("cursor", "")
            markets.extend(batch)
            if not batch or not cursor:
                break
        kalshi_markets[asset] = sorted(markets, key=lambda x: x.get("close_time", ""))
        print(f"  {asset}: {len(markets)}")

    print("\n[2/4] Fetching 5m candles...")
    five_m = {}
    for asset in selected_assets:
        sym = ASSETS_SYMS[asset]
        five_m[asset] = {
            "coinbase": fetch_5m_history(sym, "coinbase", args.history_days),
            "bitstamp": fetch_5m_history(sym, "bitstamp", args.history_days),
        }
        print(f"  {asset} done")

    print("\n[3/4] Fetching indicators...")
    ind_data = {}
    for asset in selected_assets:
        sym = ASSETS_SYMS[asset]
        ind_days = max(110, args.history_days + 10)
        df_15m = add_indicators(fetch_candles(fetcher, sym, "15m", ind_days))
        df_1h = add_indicators(fetch_candles(fetcher, sym, "1h", ind_days))
        df_4h = add_indicators(fetch_candles(fetcher, sym, "4h", ind_days))
        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100
        ind_data[asset] = {"15m": df_15m, "1h": df_1h, "4h": df_4h}
        print(f"  {asset}: {len(df_15m)} 15m candles")

    print("\n[4/4] Training per-asset M10 models...")
    strike_by_time = {
        asset: {
            mk.get("close_time"): float(mk.get("floor_strike") or 0)
            for mk in kalshi_markets[asset]
            if mk.get("close_time")
        }
        for asset in selected_assets
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "assets": selected_assets,
            "continuation_weight": args.continuation_weight,
            "continuation_dist": args.continuation_dist,
            "continuation_ret12": args.continuation_ret12,
            "monotone_distance": args.monotone_distance,
            "eval_threshold": args.eval_threshold,
            "history_days": args.history_days,
        },
        "assets": {},
    }

    for target_asset in selected_assets:
        print(f"\n{'=' * 80}")
        print(f"M10 TRAINING: {target_asset} (XGBoost v3)")
        print(f"{'=' * 80}")

        df_all = build_asset_dataset(
            target_asset=target_asset,
            all_assets=selected_assets,
            kalshi_markets=kalshi_markets,
            strike_by_time=strike_by_time,
            ind_data=ind_data,
            five_m=five_m,
        )
        if df_all.empty:
            print("  No usable samples — skipping asset")
            report["assets"][target_asset] = {"error": "no_samples"}
            continue

        print(f"  Samples: {len(df_all)} | YES: {df_all['label'].mean():.1%}")

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

        if len(df_train) < 100 or len(df_val) < 30 or len(df_test) < 30:
            print("  Not enough split data after purge — skipping asset")
            report["assets"][target_asset] = {
                "error": "insufficient_split_data",
                "train": len(df_train),
                "val": len(df_val),
                "test": len(df_test),
            }
            continue

        X_train = df_train[ALL_FEATURES].values
        y_train = df_train["label"].values
        X_val = df_val[ALL_FEATURES].values
        y_val = df_val["label"].values
        X_test = df_test[ALL_FEATURES].values
        y_test = df_test["label"].values

        sample_weights = compute_time_decay_weights(df_train["ts"], half_life_days=60)
        sample_weights = apply_continuation_weights(
            df_train,
            y_train,
            sample_weights,
            continuation_weight=args.continuation_weight,
            continuation_dist=args.continuation_dist,
            continuation_ret12=args.continuation_ret12,
        )

        model_kwargs = dict(
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
        if args.monotone_distance and "distance_from_strike" in ALL_FEATURES:
            mono = [0] * len(ALL_FEATURES)
            mono[ALL_FEATURES.index("distance_from_strike")] = 1
            model_kwargs["monotone_constraints"] = tuple(mono)

        model = XGBClassifier(**model_kwargs)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
        print(
            f"  Train: {len(df_train)} | Gap: {PURGE_GAP_SAMPLES} | "
            f"Val: {len(df_val)} | Test: {len(df_test)}"
        )
        print(f"  Best iteration: {best_iter}")

        importances = dict(zip(ALL_FEATURES, model.feature_importances_))
        print("\n  Top 10 features:")
        for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
            marker = " [5m]" if name in INTRA_WINDOW_FEATURES else ""
            print(f"    {name:<24}: {imp:.4f}{marker}")

        probs = model.predict_proba(X_test)[:, 1]
        print("\n   Thresh |  Bets     WR |     P&L")
        print("  " + "-" * 35)
        for t in [75, 80, 85, 90, 95]:
            r = evaluate_threshold(probs, y_test, t)
            if r["tot"] < 10:
                continue
            print(f"  {t:>7} | {r['tot']:>5} {r['wr']:>6.1f}% | ${r['pnl']:>+7.1f}")

        guard = evaluate_guardrails(
            probs,
            y_test,
            df_test[ALL_FEATURES],
            threshold=args.eval_threshold,
            continuation_dist=args.continuation_dist,
            continuation_ret12=args.continuation_ret12,
        )

        yes_preds = int((probs >= args.eval_threshold / 100.0).sum())
        no_preds = int((probs <= (100 - args.eval_threshold) / 100.0).sum())
        skip_preds = len(probs) - yes_preds - no_preds
        print(
            f"\n  Signal balance (t={args.eval_threshold}): "
            f"YES={yes_preds} NO={no_preds} SKIP={skip_preds}"
        )
        print(
            "  Guardrails: "
            f"true_wr={guard['true_wr']:.1f}% "
            f"bull_false_no_rate={guard['bull_false_no_rate']} "
            f"bull_no_rate={guard['bull_no_rate']}"
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        scaler.feature_names_in_ = np.array(ALL_FEATURES)

        out_path = Path(f"models/m10_{target_asset.lower()}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "model_type": "per_asset_m10_confluence",
                    "feature_names": ALL_FEATURES,
                    "asset": target_asset,
                    "training_samples": len(df_train),
                    "model_class": "XGBClassifier",
                    "xgb_version": "v3_continuation_safe",
                    "continuation_weight": args.continuation_weight,
                    "continuation_dist": args.continuation_dist,
                    "continuation_ret12": args.continuation_ret12,
                    "monotone_distance": bool(args.monotone_distance),
                },
                f,
            )
        print(f"\n  Saved to {out_path}")

        report["assets"][target_asset] = {
            "samples": int(len(df_all)),
            "train": int(len(df_train)),
            "val": int(len(df_val)),
            "test": int(len(df_test)),
            "best_iteration": int(best_iter),
            f"guardrails_t{args.eval_threshold}": guard,
            "top_features": [
                {"name": k, "importance": float(v)}
                for k, v in sorted(importances.items(), key=lambda x: -x[1])[:10]
            ],
        }

    if args.report_json:
        rp = Path(args.report_json)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written: {rp}")


if __name__ == "__main__":
    main()
