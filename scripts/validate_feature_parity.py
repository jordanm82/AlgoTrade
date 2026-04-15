#!/usr/bin/env python3
"""Validate parity diagnostics for current per-asset M0/M10 feature sets.

What this validates:
1) M0 live-vs-recomputed parity on recent feature_log windows, using each
   asset model's exact `feature_names`.
2) M10 feature-set coverage: for each asset model, verify every model feature
   is produced and finite in the current reference feature builder.

Usage:
    ./venv/bin/python scripts/validate_feature_parity.py
    ./venv/bin/python scripts/validate_feature_parity.py --last 30 --tolerance 1e-4
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
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


def _safe_float(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if np.isnan(x) or np.isinf(x):
        return default
    return x


def _load_feature_names(prefix: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for asset in ASSETS_SYMS:
        path = Path(f"models/{prefix}_{asset.lower()}.pkl")
        if not path.exists():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        names = data.get("feature_names")
        if isinstance(names, (list, tuple)) and names:
            out[asset] = list(names)
    return out


def _infer_asset(entry: dict) -> str | None:
    raw = str(entry.get("asset", "")).upper()
    if raw in ASSETS_SYMS:
        return raw

    close = _safe_float(entry.get("close"), None)
    if close is None:
        return None
    if close > 10000:
        return "BTC"
    if close > 500:
        return "ETH"
    if close > 10:
        return "SOL"
    if close > 0:
        return "XRP"
    return None


def _floor_ws(ts: datetime) -> datetime:
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        ts = ts.replace(tzinfo=None)
    return ts.replace(minute=ts.minute - (ts.minute % 15), second=0, microsecond=0)


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
            row["_ws"] = _floor_ws(ts)
            rows.append(row)

    return rows


def _prepare_indicators(fetcher: DataFetcher) -> dict[str, dict[str, pd.DataFrame | None]]:
    out: dict[str, dict[str, pd.DataFrame | None]] = {}

    for asset, symbol in ASSETS_SYMS.items():
        df_15m = add_indicators(fetcher.ohlcv(symbol, "15m", limit=260))
        raw_1h = fetcher.ohlcv(symbol, "1h", limit=160)
        raw_4h = fetcher.ohlcv(symbol, "4h", limit=110)

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


def _compute_confluence(asset: str, ws: datetime, ind_data: dict[str, dict], entry: dict | None) -> dict:
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
        # Exact recompute requires Kalshi alt strikes; use logged if available, else 0.
        "alt_distance_avg": _safe_float((entry or {}).get("alt_distance_avg"), 0.0),
    }
    return out


def _build_m0_features(entry: dict, ind_data: dict[str, dict]) -> tuple[dict | None, str | None]:
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
        return None, "missing distance_from_strike"

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


def _build_m10_features(asset: str, ws: datetime, ind_data: dict[str, dict],
                        fetcher: DataFetcher) -> tuple[dict | None, str | None]:
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

    kx = {
        "strike_delta": 0.0,
        "strike_trend_3": 0.0,
        "alt_distance_avg": 0.0,
    }
    kx.update(_compute_confluence(asset, ws, ind_data, entry=None))
    kx.update(_compute_regime(df_15m, df_1h, df_4h, ws, atr))

    feat = build_features(
        prev,
        df_1h,
        df_4h,
        ws,
        0.0,
        kalshi_extra=kx,
        atr_pctile_val=_atr_percentile(df_15m, ws),
    )
    if not feat:
        return None, "build_features returned None"

    # M10-only intra-window features
    feat["price_move_atr"] = 0.0
    feat["candle1_range_atr"] = 0.0
    feat["candle2_range_atr"] = 0.0
    feat["momentum_shift"] = 0.0
    feat["volume_acceleration"] = 1.0

    # Try to compute live-style intra-window values from 1m candles.
    try:
        df_1m = fetcher.ohlcv(ASSETS_SYMS[asset], "1m", limit=60)
        if df_1m is not None and len(df_1m) >= 10:
            min5 = ws + timedelta(minutes=5)
            min10 = ws + timedelta(minutes=10)
            c1 = df_1m[(df_1m.index >= ws) & (df_1m.index < min5)]
            c2 = df_1m[(df_1m.index >= min5) & (df_1m.index < min10)]
            if len(c1) >= 3 and len(c2) >= 3 and atr > 0:
                c1_open = float(c1.iloc[0]["open"])
                c1_close = float(c1.iloc[-1]["close"])
                c1_high = float(c1["high"].max())
                c1_low = float(c1["low"].min())
                c1_vol = float(c1["volume"].sum())

                c2_close = float(c2.iloc[-1]["close"])
                c2_high = float(c2["high"].max())
                c2_low = float(c2["low"].min())
                c2_vol = float(c2["volume"].sum())

                feat["price_move_atr"] = (c2_close - c1_open) / atr
                feat["candle1_range_atr"] = (c1_high - c1_low) / atr
                feat["candle2_range_atr"] = (c2_high - c2_low) / atr
                feat["momentum_shift"] = (c2_close - c1_close) / atr
                feat["volume_acceleration"] = c2_vol / c1_vol if c1_vol > 0 else 1.0
    except Exception:
        pass

    return feat, None


def _compare_feature_dict(live_entry: dict, recomputed: dict, features: list[str],
                          tolerance: float) -> list[dict]:
    diffs = []
    for name in features:
        lv = live_entry.get(name)
        rv = recomputed.get(name)
        lvf = _safe_float(lv, None)
        rvf = _safe_float(rv, None)

        if lvf is None or rvf is None:
            diffs.append({
                "feature": name,
                "live": lv,
                "recomputed": rv,
                "diff": None,
                "reason": "missing_or_non_numeric",
            })
            continue

        diff = abs(lvf - rvf)
        if diff > tolerance:
            diffs.append({
                "feature": name,
                "live": lvf,
                "recomputed": rvf,
                "diff": diff,
                "reason": "value_mismatch",
            })
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="Use last N deduped M0 predictions from feature_log")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Absolute diff tolerance")
    parser.add_argument("--show", type=int, default=10, help="Max mismatch lines to print per asset")
    args = parser.parse_args()

    m0_features = _load_feature_names("m0")
    m10_features = _load_feature_names("m10")

    print("=" * 100)
    print("FEATURE PARITY VALIDATION (per-asset M0/M10)")
    print("=" * 100)
    print(f"Loaded M0 models: {len(m0_features)} | Loaded M10 models: {len(m10_features)}")
    for asset in ASSETS_SYMS:
        m0n = len(m0_features.get(asset, []))
        m10n = len(m10_features.get(asset, []))
        print(f"  {asset}: M0={m0n:>2} features | M10={m10n:>2} features")

    rows = _read_feature_log(Path("data/store/feature_log.jsonl"))
    rows = [r for r in rows if r["_asset"] in m0_features]

    dedup: dict[tuple[str, datetime], dict] = {}
    for row in rows:
        dedup[(row["_asset"], row["_ws"])] = row
    selected = sorted(dedup.values(), key=lambda r: r["_ts"])[-args.last:]

    fetcher = DataFetcher()
    ind_data = _prepare_indicators(fetcher)

    print("\n" + "=" * 100)
    print("M0 LIVE VS RECOMPUTED")
    print("=" * 100)

    total_checked = 0
    total_skipped = 0
    total_mismatch_entries = 0
    per_asset_mismatches: dict[str, list[dict]] = defaultdict(list)

    for row in selected:
        asset = row["_asset"]
        feats = m0_features.get(asset, [])
        if not feats:
            continue

        rec, err = _build_m0_features(row, ind_data)
        total_checked += 1
        if rec is None:
            total_skipped += 1
            per_asset_mismatches[asset].append({
                "feature": "<entry>",
                "live": "n/a",
                "recomputed": "n/a",
                "diff": None,
                "reason": err,
            })
            continue

        diffs = _compare_feature_dict(row, rec, feats, args.tolerance)
        if diffs:
            total_mismatch_entries += 1
            per_asset_mismatches[asset].extend(diffs)

    print(f"Entries checked: {total_checked}")
    print(f"Entries with mismatches: {total_mismatch_entries}")
    print(f"Entries skipped: {total_skipped}")

    for asset in ASSETS_SYMS:
        diffs = per_asset_mismatches.get(asset, [])
        if not diffs:
            print(f"\n[{asset}] no mismatches above tolerance")
            continue

        print(f"\n[{asset}] mismatches: {len(diffs)}")
        sorted_diffs = sorted(
            diffs,
            key=lambda d: d["diff"] if d["diff"] is not None else float("inf"),
            reverse=True,
        )
        for d in sorted_diffs[: max(1, args.show)]:
            if d["diff"] is None:
                print(
                    f"  {d['feature']:<24} live={d['live']} recomputed={d['recomputed']} diff=NA ({d['reason']})"
                )
            else:
                print(
                    f"  {d['feature']:<24} live={d['live']:+.8f} recomputed={d['recomputed']:+.8f} diff={d['diff']:.8f}"
                )

    print("\n" + "=" * 100)
    print("M10 FEATURE COVERAGE (model feature_names)")
    print("=" * 100)

    ws_now = _floor_ws(datetime.now(timezone.utc))

    for asset in ASSETS_SYMS:
        feats = m10_features.get(asset, [])
        if not feats:
            print(f"[{asset}] no M10 model found")
            continue

        rec, err = _build_m10_features(asset, ws_now, ind_data, fetcher)
        if rec is None:
            print(f"[{asset}] unable to build reference M10 features: {err}")
            continue

        issues = []
        for name in feats:
            val = rec.get(name)
            vf = _safe_float(val, None)
            if vf is None:
                issues.append({
                    "feature": name,
                    "live": "n/a",
                    "recomputed": val,
                    "diff": "n/a",
                    "reason": "missing_or_non_numeric_reference",
                })

        if not issues:
            print(f"[{asset}] OK ({len(feats)} / {len(feats)} model features available and finite)")
        else:
            print(f"[{asset}] FAIL ({len(issues)} problematic features)")
            for issue in issues[: max(1, args.show)]:
                print(
                    f"  {issue['feature']:<24} live={issue['live']} recomputed={issue['recomputed']} "
                    f"diff={issue['diff']} ({issue['reason']})"
                )


if __name__ == "__main__":
    main()
