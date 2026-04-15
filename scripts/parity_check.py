#!/usr/bin/env python3
"""Parity diagnostics for current per-asset M0 features.

Reads live feature logs, rebuilds features with the backtest-style code path,
and compares only the features actually used by each asset's current model.

Usage:
    ./venv/bin/python scripts/parity_check.py --last 20
    ./venv/bin/python scripts/parity_check.py --asset BTC --last 15 --tolerance 1e-4
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from scripts.backtest_kalshi_labels import build_features

ASSETS_SYMS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
}


def _load_m0_feature_names() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        path = Path(f"models/m0_{asset.lower()}.pkl")
        if not path.exists():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        names = data.get("feature_names")
        if isinstance(names, (list, tuple)) and names:
            out[asset] = list(names)
    return out


def _infer_asset(entry: dict) -> str | None:
    raw_asset = str(entry.get("asset", "")).upper()
    if raw_asset in ASSETS_SYMS:
        return raw_asset

    close = entry.get("close")
    try:
        close_val = float(close)
    except (TypeError, ValueError):
        return None

    if close_val > 10000:
        return "BTC"
    if close_val > 500:
        return "ETH"
    if close_val > 10:
        return "SOL"
    if close_val > 0:
        return "XRP"
    return None


def _floor_window_start(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        ts = ts.replace(tzinfo=None)
    return ts.replace(minute=ts.minute - (ts.minute % 15), second=0, microsecond=0)


def _safe_float(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if np.isnan(x) or np.isinf(x):
        return default
    return x


def _read_feature_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows

    with open(path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ts_raw = row.get("ts")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue

            asset = _infer_asset(row)
            if asset is None:
                continue

            row["_asset"] = asset
            row["_ts"] = ts
            row["_ws"] = _floor_window_start(ts)
            rows.append(row)

    return rows


def _prepare_indicators(fetcher: DataFetcher) -> dict[str, dict[str, pd.DataFrame | None]]:
    out: dict[str, dict[str, pd.DataFrame | None]] = {}

    for asset, symbol in ASSETS_SYMS.items():
        df_15m = add_indicators(fetcher.ohlcv(symbol, "15m", limit=240))
        raw_1h = fetcher.ohlcv(symbol, "1h", limit=140)
        raw_4h = fetcher.ohlcv(symbol, "4h", limit=100)

        df_1h = add_indicators(raw_1h.iloc[:-1]) if raw_1h is not None and len(raw_1h) > 1 else None
        df_4h = add_indicators(raw_4h.iloc[:-1]) if raw_4h is not None and len(raw_4h) > 1 else None

        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100

        out[asset] = {"15m": df_15m, "1h": df_1h, "4h": df_4h}

    return out


def _atr_percentile(df_15m: pd.DataFrame, ws: datetime) -> float:
    atr_s = df_15m["atr"].dropna()
    if atr_s.empty:
        return 0.5
    r20 = atr_s.rolling(20)
    atr_p = ((atr_s - r20.min()) / (r20.max() - r20.min())).fillna(0.5)
    hist = atr_p[atr_p.index < ws]
    if len(hist) == 0:
        return 0.5
    return float(hist.iloc[-1])


def _compute_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame | None,
                    df_4h: pd.DataFrame | None, ws: datetime, atr: float) -> dict:
    out: dict[str, float] = {}

    prev_c = df_15m[df_15m.index < ws]
    if len(prev_c) >= 16:
        out["return_4h"] = (
            (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-16]["close"]))
            / float(prev_c.iloc[-16]["close"]) * 100
        )
    else:
        out["return_4h"] = 0.0

    if len(prev_c) >= 48:
        out["return_12h"] = (
            (float(prev_c.iloc[-1]["close"]) - float(prev_c.iloc[-48]["close"]))
            / float(prev_c.iloc[-48]["close"]) * 100
        )
    else:
        out["return_12h"] = 0.0

    if df_1h is not None:
        h1 = df_1h[df_1h.index < ws]
        if len(h1) >= 20 and atr > 0:
            out["price_vs_sma_1h"] = (
                float(h1.iloc[-1]["close"]) - float(h1["close"].rolling(20).mean().iloc[-1])
            ) / atr
        else:
            out["price_vs_sma_1h"] = 0.0
    else:
        out["price_vs_sma_1h"] = 0.0

    if df_4h is not None:
        h4 = df_4h[df_4h.index < ws]
        if len(h4) >= 4:
            out["lower_lows_4h"] = float(
                sum(1 for i in range(-3, 0) if float(h4.iloc[i]["low"]) < float(h4.iloc[i - 1]["low"]))
            )
        else:
            out["lower_lows_4h"] = 0.0
        if len(h4) >= 10 and atr > 0:
            out["trend_strength"] = (
                float(h4.iloc[-1]["close"]) - float(h4["close"].rolling(10).mean().iloc[-1])
            ) / atr
        else:
            out["trend_strength"] = 0.0
    else:
        out["lower_lows_4h"] = 0.0
        out["trend_strength"] = 0.0

    return out


def _compute_confluence(asset: str, ws: datetime, ind_data: dict[str, dict], entry: dict) -> dict:
    alt_assets = [a for a in ASSETS_SYMS if a != asset]
    alt_rsi_15m: list[float] = []
    alt_rsi_1h: list[float] = []
    alt_momentum: list[int] = []

    for alt in alt_assets:
        alt_15m = ind_data[alt]["15m"]
        alt_1h = ind_data[alt]["1h"]

        alt_15m_f = alt_15m[alt_15m.index < ws]
        if len(alt_15m_f) >= 2:
            rsi_v = _safe_float(alt_15m_f.iloc[-1].get("rsi", 50), 50.0)
            alt_rsi_15m.append(rsi_v)
            alt_momentum.append(1 if rsi_v >= 50 else -1)

        if alt_1h is not None:
            alt_1h_f = alt_1h[alt_1h.index < ws]
            if len(alt_1h_f) >= 2:
                alt_rsi_1h.append(_safe_float(alt_1h_f.iloc[-1].get("rsi", 50), 50.0))

    out = {
        "alt_rsi_avg": sum(alt_rsi_15m) / len(alt_rsi_15m) if alt_rsi_15m else 50.0,
        "alt_rsi_1h_avg": sum(alt_rsi_1h) / len(alt_rsi_1h) if alt_rsi_1h else 50.0,
        "alt_momentum_align": float(sum(alt_momentum)) if alt_momentum else 0.0,
        # Needs Kalshi alt strikes to recompute exactly; use logged value if present.
        "alt_distance_avg": _safe_float(entry.get("alt_distance_avg"), 0.0),
    }
    return out


def _recompute_entry_features(entry: dict, ind_data: dict[str, dict]) -> tuple[dict | None, str | None]:
    asset = entry["_asset"]
    ws = entry["_ws"]

    df_15m = ind_data[asset]["15m"]
    df_1h = ind_data[asset]["1h"]
    df_4h = ind_data[asset]["4h"]

    prev_c = df_15m[df_15m.index < ws]
    if len(prev_c) < 20:
        return None, "not enough 15m history"

    prev = prev_c.iloc[-1]
    atr = _safe_float(prev.get("atr"), None)
    if atr is None or atr <= 0:
        return None, "invalid ATR"

    distance = _safe_float(entry.get("distance_from_strike"), None)
    if distance is None:
        return None, "missing distance_from_strike in live log"

    kx = {
        "strike_delta": _safe_float(entry.get("strike_delta"), 0.0),
        "strike_trend_3": _safe_float(entry.get("strike_trend_3"), 0.0),
    }
    kx.update(_compute_confluence(asset, ws, ind_data, entry))
    kx.update(_compute_regime(df_15m, df_1h, df_4h, ws, atr))

    feat = build_features(
        prev,
        df_1h,
        df_4h,
        ws,
        distance,
        kalshi_extra=kx,
        atr_pctile_val=_atr_percentile(df_15m, ws),
    )
    if not feat:
        return None, "build_features returned None"

    return feat, None


def _compare_features(entry: dict, recomputed: dict, model_features: list[str],
                      tolerance: float) -> list[dict]:
    diffs: list[dict] = []
    for name in model_features:
        live_v = entry.get(name)
        rec_v = recomputed.get(name)

        live_f = _safe_float(live_v, None)
        rec_f = _safe_float(rec_v, None)

        if live_f is None or rec_f is None:
            diffs.append({
                "feature": name,
                "live": live_v,
                "recomputed": rec_v,
                "diff": None,
                "reason": "missing_or_non_numeric",
            })
            continue

        diff = abs(live_f - rec_f)
        if diff > tolerance:
            diffs.append({
                "feature": name,
                "live": live_f,
                "recomputed": rec_f,
                "diff": diff,
                "reason": "value_mismatch",
            })
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="Check the last N deduped predictions")
    parser.add_argument("--asset", type=str, default="", help="Optional asset filter: BTC/ETH/SOL/XRP")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Absolute diff tolerance")
    parser.add_argument("--show", type=int, default=12, help="Max mismatches to print per prediction")
    args = parser.parse_args()

    model_features = _load_m0_feature_names()
    if not model_features:
        print("No per-asset M0 models found in models/m0_*.pkl")
        return

    asset_filter = args.asset.upper().strip()
    if asset_filter and asset_filter not in ASSETS_SYMS:
        print(f"Invalid --asset '{args.asset}'. Expected one of: {', '.join(ASSETS_SYMS)}")
        return

    rows = _read_feature_log(Path("data/store/feature_log.jsonl"))
    if asset_filter:
        rows = [r for r in rows if r["_asset"] == asset_filter]

    # Keep only rows for assets with loaded model feature names.
    rows = [r for r in rows if r["_asset"] in model_features]
    if not rows:
        print("No matching feature_log entries found.")
        return

    # Deduplicate by (asset, window_start); keep latest log line for that window.
    dedup: dict[tuple[str, datetime], dict] = {}
    for row in rows:
        key = (row["_asset"], row["_ws"])
        dedup[key] = row

    selected = sorted(dedup.values(), key=lambda r: r["_ts"])[-args.last:]
    if not selected:
        print("No entries selected after dedupe/filter.")
        return

    fetcher = DataFetcher()
    ind_data = _prepare_indicators(fetcher)

    print("=" * 100)
    print("PARITY CHECK (M0, model-feature aware)")
    print(f"Entries: {len(selected)} | Tolerance: {args.tolerance:g}")
    if asset_filter:
        print(f"Asset filter: {asset_filter}")
    print("=" * 100)

    checked = 0
    with_mismatch = 0
    skipped = 0
    mismatch_by_feature: dict[str, list[float]] = defaultdict(list)

    for row in selected:
        asset = row["_asset"]
        feats = model_features.get(asset)
        if not feats:
            skipped += 1
            continue

        recomputed, err = _recompute_entry_features(row, ind_data)
        checked += 1

        ts = row["_ts"].isoformat()
        prob = _safe_float(row.get("prob"), None)
        prob_txt = f"{prob:.4f}" if prob is not None else "n/a"

        if recomputed is None:
            skipped += 1
            print(f"[{ts}] {asset} prob={prob_txt} -> SKIP ({err})")
            continue

        diffs = _compare_features(row, recomputed, feats, args.tolerance)
        if not diffs:
            print(f"[{ts}] {asset} prob={prob_txt} -> OK ({len(feats)} model features)")
            continue

        with_mismatch += 1
        diffs_sorted = sorted(diffs, key=lambda d: d["diff"] if d["diff"] is not None else float("inf"), reverse=True)
        print("-" * 100)
        print(f"[{ts}] {asset} prob={prob_txt} -> {len(diffs)} mismatches ({len(feats)} checked)")
        for d in diffs_sorted[: max(1, args.show)]:
            if d["diff"] is None:
                print(
                    f"  {d['feature']:<24} live={d['live']} recomputed={d['recomputed']} diff=NA ({d['reason']})"
                )
            else:
                mismatch_by_feature[d["feature"]].append(float(d["diff"]))
                print(
                    f"  {d['feature']:<24} live={d['live']:+.8f} recomputed={d['recomputed']:+.8f} diff={d['diff']:.8f}"
                )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Checked entries: {checked}")
    print(f"Entries with mismatches: {with_mismatch}")
    print(f"Skipped entries: {skipped}")

    if mismatch_by_feature:
        print("\nTop mismatched features:")
        ranked = sorted(
            mismatch_by_feature.items(),
            key=lambda kv: (len(kv[1]), np.mean(kv[1]) if kv[1] else 0.0),
            reverse=True,
        )
        for feat, vals in ranked[:15]:
            print(f"  {feat:<24} count={len(vals):>3} mean_abs_diff={np.mean(vals):.8f} max_abs_diff={np.max(vals):.8f}")
    else:
        print("No numeric mismatches above tolerance.")


if __name__ == "__main__":
    main()
