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
from config.production import MAX_CONCURRENT_KALSHI_BETS, KALSHI_RISK_PER_BET_PCT
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID
from strategy.m10_feature_builder import (
    build_common_feature_vector,
    compute_confluence_features,
    compute_m10_intra_from_exchange_dfs,
    filter_completed_candles,
    get_avg_price_5m,
)

SERIES = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M", "XRP": "KXXRP15M"}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

STARTING_BALANCE = 100.0
RISK_PCT = KALSHI_RISK_PER_BET_PCT
MAX_CONTRACTS = 100
ENTRY_CENTS = 60
DEFAULT_MAX_PER_WINDOW = MAX_CONCURRENT_KALSHI_BETS
M10_EXIT_CENTS = 10
DEFAULT_DAILY_LOSS_CAP = 0.0

# Per-asset thresholds (must match daemon)
M0_THRESHOLDS = {"BTC": 57, "ETH": 57, "SOL": 57, "XRP": 57}
M10_THRESHOLDS = {"BTC": 85, "ETH": 85, "SOL": 85, "XRP": 85}

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
    # Backwards-compatible wrapper for callers and existing training imports.
    return get_avg_price_5m(cb_5m, bs_5m, target_time, field)


def get_m10_exit_price_cents(*, historical_bid_cents=None, floor_cents=M10_EXIT_CENTS):
    """Conservative floor pricing assumption."""
    if historical_bid_cents is None:
        return int(floor_cents)
    return max(int(floor_cents), int(historical_bid_cents))


def estimate_m10_bid_cents(*, side: str, m10_yes_prob: float, floor_cents: int, spread_cents: int = 5):
    """Estimate an actionable bid from model probability (bounds-mode upper estimate)."""
    fair_yes = float(m10_yes_prob) * 100.0
    fair_side = fair_yes if side == "yes" else (100.0 - fair_yes)
    est_bid = int(round(fair_side - float(spread_cents)))
    est_bid = max(int(floor_cents), min(99, est_bid))
    return est_bid


def build_features(prev, df_1h, df_4h, ws_naive, distance, *,
                    kalshi_extra=None, atr_pctile_val=0.5):
    # Backwards-compatible wrapper for callers and existing training imports.
    return build_common_feature_vector(
        prev,
        df_1h,
        df_4h,
        ws_naive,
        distance,
        kalshi_extra=kalshi_extra,
        atr_pctile_val=atr_pctile_val,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=0, help="Override M0 threshold (0=per-asset)")
    parser.add_argument("--m10-threshold", type=int, default=0, help="Override M10 threshold (0=per-asset)")
    parser.add_argument(
        "--m10-exit-mode",
        choices=["floor", "bounds"],
        default="bounds",
        help="floor=always floor price on M10 exits; bounds=estimate bid and track floor bound",
    )
    parser.add_argument(
        "--m10-est-spread",
        type=int,
        default=5,
        help="Bid spread discount in cents for bounds-mode estimated M10 exits",
    )
    parser.add_argument("--m10-exit-floor", type=int, default=M10_EXIT_CENTS,
                        help="Minimum M10 exit price floor in cents")
    parser.add_argument(
        "--max-per-window",
        type=int,
        default=DEFAULT_MAX_PER_WINDOW,
        help=(
            "Max bets per 15m window. "
            f"Default matches live MAX_CONCURRENT_KALSHI_BETS ({DEFAULT_MAX_PER_WINDOW})."
        ),
    )
    parser.add_argument("--daily-loss-cap", type=float, default=DEFAULT_DAILY_LOSS_CAP,
                        help="Optional per-day loss cap (fraction, e.g. 0.2). 0 disables cap.")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    max_per_window = max(1, int(args.max_per_window))

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
    print(
        f"Entry: {ENTRY_CENTS}c (assumed fill) | Max/window: {max_per_window} | "
        f"M10 exit mode: {args.m10_exit_mode} "
        f"(floor={args.m10_exit_floor}c, est_spread={args.m10_est_spread}c)"
    )
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
    strike_by_time = {
        asset: {
            mk.get("close_time"): float(mk.get("floor_strike") or 0)
            for mk in kalshi_markets[asset]
            if mk.get("close_time")
        }
        for asset in all_assets
    }
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

            # Confluence at minute-0 anchor (shared builder path).
            kx.update(
                compute_confluence_features(
                    alt_keys=alts,
                    ws_naive=wsn,
                    get_15m_df=lambda alt: ind_data[alt]["15m"],
                    get_1h_df=lambda alt: ind_data[alt]["1h"],
                    get_anchor_price=lambda alt: get_avg_price(
                        five_m[alt]["coinbase"], five_m[alt]["bitstamp"], wsn, "open"
                    ),
                    get_strike=lambda alt: strike_by_time.get(alt, {}).get(ct) or None,
                )
            )

            # Regime features
            if len(pc) >= 16:
                kx["return_4h"] = (float(pc.iloc[-1]["close"]) - float(pc.iloc[-16]["close"])) / float(pc.iloc[-16]["close"]) * 100
            else: kx["return_4h"] = 0
            if len(pc) >= 48:
                kx["return_12h"] = (float(pc.iloc[-1]["close"]) - float(pc.iloc[-48]["close"])) / float(pc.iloc[-48]["close"]) * 100
            else: kx["return_12h"] = 0
            if df1h is not None:
                h1f = filter_completed_candles(df1h, wsn, "1h")
                if len(h1f) >= 20 and atr > 0:
                    kx["price_vs_sma_1h"] = (float(h1f.iloc[-1]["close"]) - float(h1f["close"].rolling(20).mean().iloc[-1])) / atr
                else: kx["price_vs_sma_1h"] = 0
            else: kx["price_vs_sma_1h"] = 0
            if df4h is not None:
                h4f = filter_completed_candles(df4h, wsn, "4h")
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
            if any(f not in f0 for f in m0f):
                continue
            X0 = np.array([f0[f] for f in m0f], dtype=float).reshape(1, -1)
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
            exit_price = None
            exit_price_floor = None
            exit_price_est = None
            m10_prob = None
            m10_quality = None
            min10 = wsn + timedelta(minutes=10)
            p10 = get_avg_price(cb5, bs5, min10, "open")
            if p10 is not None:
                # Recompute confluence at minute-10 anchor for M10 parity.
                kx_m10 = dict(kx)
                kx_m10.update(
                    compute_confluence_features(
                        alt_keys=alts,
                        ws_naive=wsn,
                        get_15m_df=lambda alt: ind_data[alt]["15m"],
                        get_1h_df=lambda alt: ind_data[alt]["1h"],
                        get_anchor_price=lambda alt: get_avg_price(
                            five_m[alt]["coinbase"], five_m[alt]["bitstamp"], min10, "open"
                        ),
                        get_strike=lambda alt: strike_by_time.get(alt, {}).get(ct) or None,
                    )
                )

                f10 = build_features(
                    prev,
                    df1h,
                    df4h,
                    wsn,
                    (p10 - strike) / atr,
                    kalshi_extra=kx_m10,
                    atr_pctile_val=apv,
                )
                if f10:
                    intra = compute_m10_intra_from_exchange_dfs(cb5, bs5, wsn, atr)
                    if intra is None:
                        # Strict no-fallback: missing intra-window 5m features => no M10 action.
                        f10 = None
                    else:
                        f10.update(intra)

                    if f10 is not None:
                        if any(f not in f10 for f in m10f):
                            f10 = None
                    if f10 is not None:
                        X10 = np.array([f10[f] for f in m10f], dtype=float).reshape(1, -1)
                        is_tree_m10 = hasattr(m10m, 'get_booster') or 'XGB' in type(m10m).__name__
                        X10_pred = X10 if is_tree_m10 else m10s.transform(X10)
                        mp = float(m10m.predict_proba(X10_pred)[0][1])
                        m10_prob = mp
                        mpc = int(mp * 100)
                        ms = "yes" if mpc >= m10_thresh else "no" if mpc <= (100 - m10_thresh) else "skip"
                        if ms != "skip" and ms != side:
                            m10x = True
                            exit_price_floor = get_m10_exit_price_cents(
                                historical_bid_cents=None,
                                floor_cents=args.m10_exit_floor,
                            )
                            exit_price_est = estimate_m10_bid_cents(
                                side=side,
                                m10_yes_prob=mp,
                                floor_cents=args.m10_exit_floor,
                                spread_cents=args.m10_est_spread,
                            )
                            exit_price = (
                                exit_price_est if args.m10_exit_mode == "bounds"
                                else exit_price_floor
                            )
                            m10_quality = "GOOD" if not ((side == "yes" and label == 1) or (side == "no" and label == 0)) else "BAD"

            won = (side == "yes" and label == 1) or (side == "no" and label == 0)
            if m10x:
                pnl_per = (int(exit_price) - ENTRY_CENTS) if exit_price is not None else 0
                if pnl_per > 0:
                    outcome = "WIN_EXIT"
                elif pnl_per < 0:
                    outcome = "LOSS_EXIT"
                else:
                    outcome = "FLAT_EXIT"
            else:
                outcome = "WIN" if won else "LOSS"
            signals.append({
                "ts": cdt, "asset": target, "side": side.upper(), "m0_prob": prob,
                "m10_exit": m10x, "exit_price": exit_price,
                "exit_price_floor": exit_price_floor,
                "exit_price_est": exit_price_est,
                "m10_prob": m10_prob,
                "m10_quality": m10_quality,
                "outcome": outcome, "won": won, "won_settlement": won,
            })
            ct_count += 1
        print(f"  {target}: {ct_count} signals")

    df_sig = pd.DataFrame(signals).sort_values("ts").reset_index(drop=True)
    total = len(df_sig)
    if total == 0:
        print("No signals generated"); return

    # Results
    wins = df_sig["outcome"].isin(["WIN", "WIN_EXIT"]).sum()
    losses = df_sig["outcome"].isin(["LOSS", "LOSS_EXIT"]).sum()
    flats = (df_sig["outcome"] == "FLAT_EXIT").sum()
    realized_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    entry_wr = df_sig["won_settlement"].mean() * 100
    held_wins = (df_sig["outcome"] == "WIN").sum()
    held_losses = (df_sig["outcome"] == "LOSS").sum()
    held_wr = held_wins / (held_wins + held_losses) * 100 if (held_wins + held_losses) > 0 else 0

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
    print(f"  Realized outcomes: W:{wins} L:{losses} F:{flats} | Total: {wins + losses + flats}")
    print(f"  Realized WR (W/(W+L)): {realized_wr:.1f}%")
    print(f"  Held WR: {held_wr:.1f}%")
    if int(df_sig["m10_exit"].sum()) > 0:
        ex = df_sig[df_sig["m10_exit"]]
        good = (~ex["won_settlement"]).sum()
        bad = ex["won_settlement"].sum()
        exit_win = (ex["outcome"] == "WIN_EXIT").sum()
        exit_loss = (ex["outcome"] == "LOSS_EXIT").sum()
        exit_flat = (ex["outcome"] == "FLAT_EXIT").sum()
        if (good + bad) > 0:
            print(f"  M10 quality vs settlement: {good} good / {bad} bad ({good / (good + bad) * 100:.0f}% acc)")
        print(f"  M10 realized exits: W:{exit_win} L:{exit_loss} F:{exit_flat}")
        if args.m10_exit_mode == "bounds":
            ex_floor = ex["exit_price_floor"].dropna().astype(float)
            ex_est = ex["exit_price_est"].dropna().astype(float)
            if len(ex_floor) > 0 and len(ex_est) > 0:
                print(
                    f"  M10 exit price bounds: floor avg={ex_floor.mean():.1f}c | "
                    f"est avg={ex_est.mean():.1f}c"
                )

    # P&L with daily loss cap
    balance = STARTING_BALANCE
    peak = balance
    max_dd = max_dd_dollars = max_dd_peak = max_dd_trough = 0
    bets_placed = bets_skipped = 0
    pnl_wins = pnl_losses = pnl_exit_net = 0
    executed_rows: list[dict] = []
    current_day = None
    day_start = balance
    day_halted = False
    loss_cap = max(0.0, float(args.daily_loss_cap))
    loss_cap_enabled = loss_cap > 0

    for _, group in df_sig.groupby(df_sig["ts"].dt.floor("15min")):
        window_bets = group.nlargest(max_per_window, "m0_prob")
        for _, row in window_bets.iterrows():
            if loss_cap_enabled:
                bet_day = row["ts"].strftime("%m/%d")
                if bet_day != current_day:
                    current_day = bet_day; day_start = balance; day_halted = False
                if day_halted: bets_skipped += 1; continue
                if day_start > 0 and (day_start - balance) / day_start >= loss_cap:
                    day_halted = True; bets_skipped += 1; continue

            risk = balance * RISK_PCT
            contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
            cost = contracts * (ENTRY_CENTS / 100)
            if cost > balance: continue
            bets_placed += 1
            executed_rows.append(row.to_dict())

            if row["outcome"] == "WIN":
                p = contracts * ((100 - ENTRY_CENTS) / 100); balance += p; pnl_wins += p
            elif row["outcome"] == "LOSS":
                balance -= cost; pnl_losses += cost
            elif row["m10_exit"]:
                ep = row.get("exit_price")
                if pd.isna(ep):
                    raise RuntimeError("M10 exit signal missing exit_price")
                ep = int(ep)
                pnl_per = (ep - ENTRY_CENTS) / 100
                exit_pnl = contracts * pnl_per
                balance += exit_pnl
                pnl_exit_net += exit_pnl

            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            dd_d = peak - balance
            if dd > max_dd:
                max_dd = dd; max_dd_dollars = dd_d; max_dd_peak = peak; max_dd_trough = balance

    floor_balance = None
    floor_net_pnl = None
    if args.m10_exit_mode == "bounds":
        floor_balance = STARTING_BALANCE
        floor_pnl_wins = floor_pnl_losses = floor_pnl_exit = 0.0
        floor_current_day = None
        floor_day_start = floor_balance
        floor_day_halted = False
        for _, group in df_sig.groupby(df_sig["ts"].dt.floor("15min")):
            window_bets = group.nlargest(max_per_window, "m0_prob")
            for _, row in window_bets.iterrows():
                if loss_cap_enabled:
                    bet_day = row["ts"].strftime("%m/%d")
                    if bet_day != floor_current_day:
                        floor_current_day = bet_day
                        floor_day_start = floor_balance
                        floor_day_halted = False
                    if floor_day_halted:
                        continue
                    if floor_day_start > 0 and (floor_day_start - floor_balance) / floor_day_start >= loss_cap:
                        floor_day_halted = True
                        continue

                risk = floor_balance * RISK_PCT
                contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
                cost = contracts * (ENTRY_CENTS / 100)
                if cost > floor_balance:
                    continue

                if row["outcome"] == "WIN":
                    p = contracts * ((100 - ENTRY_CENTS) / 100)
                    floor_balance += p
                    floor_pnl_wins += p
                elif row["outcome"] == "LOSS":
                    floor_balance -= cost
                    floor_pnl_losses += cost
                elif row["m10_exit"]:
                    ep_floor = row.get("exit_price_floor")
                    ep = int(ep_floor) if not pd.isna(ep_floor) else int(row["exit_price"])
                    exit_pnl = contracts * ((ep - ENTRY_CENTS) / 100)
                    floor_balance += exit_pnl
                    floor_pnl_exit += exit_pnl
        floor_net_pnl = floor_pnl_wins - floor_pnl_losses + floor_pnl_exit

    ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    net_pnl = pnl_wins - pnl_losses + pnl_exit_net
    print(f"\n{'=' * 80}")
    print(f"P&L: ${STARTING_BALANCE} start, {RISK_PCT*100:.0f}% risk, max {max_per_window}/window")
    print(f"{'=' * 80}")
    if loss_cap_enabled:
        print(f"  Bets: {bets_placed} (skipped {bets_skipped} from daily cap @ {loss_cap:.0%})")
    else:
        print(f"  Bets: {bets_placed} (daily cap disabled)")
    print(f"  Gross won:   ${pnl_wins:+,.2f}")
    print(f"  Gross lost:  ${-pnl_losses:+,.2f}")
    print(f"  Early exits: ${pnl_exit_net:+,.2f}")
    print(f"  Net P&L:     ${net_pnl:+,.2f}")
    print(f"  Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
    if floor_balance is not None and floor_net_pnl is not None:
        floor_ret = (floor_balance - STARTING_BALANCE) / STARTING_BALANCE * 100
        print(
            f"  Bounds (floor vs est): End ${floor_balance:,.2f} ({floor_ret:+,.1f}%) "
            f"→ ${balance:,.2f} ({ret:+,.1f}%)"
        )
    print(f"  Max DD: {max_dd:.1f}% (${max_dd_dollars:,.2f} — ${max_dd_peak:,.2f} to ${max_dd_trough:,.2f})")

    df_exec = pd.DataFrame(executed_rows)
    if df_exec.empty:
        print("\nNo executed bets after window-cap/daily-cap filters.")
        return

    # Daily (executed bets only)
    def sig_pnl(row):
        if row["outcome"] == "WIN":
            return (100 - ENTRY_CENTS) / 100
        if row["outcome"] == "LOSS":
            return -ENTRY_CENTS / 100
        ep = row.get("exit_price")
        if pd.isna(ep):
            raise RuntimeError("M10 exit signal missing exit_price in daily P&L")
        return (float(ep) - ENTRY_CENTS) / 100
    df_exec["flat_pnl"] = df_exec.apply(sig_pnl, axis=1)
    df_exec["date"] = df_exec["ts"].apply(lambda t: t.strftime("%m/%d"))
    daily = df_exec.groupby("date").agg(
        bets=("outcome", "count"),
        wins=("outcome", lambda x: x.isin(["WIN", "WIN_EXIT"]).sum()),
        losses=("outcome", lambda x: x.isin(["LOSS", "LOSS_EXIT"]).sum()),
        flats=("outcome", lambda x: (x == "FLAT_EXIT").sum()),
        exits=("m10_exit", "sum"),
        pnl=("flat_pnl", "sum"),
    )
    daily["true_wr"] = daily["wins"] / (daily["wins"] + daily["losses"]) * 100

    print(f"\n{'Date':<8} {'Bets':>5} {'W':>3} {'L':>3} {'F':>3} {'MX':>3} {'TWR':>6} {'P&L':>8}")
    print("-" * 42)
    for date, row in daily.iterrows():
        denom = int(row["wins"] + row["losses"])
        wr_str = f"{row['true_wr']:.0f}%" if denom > 0 else "  --"
        print(
            f"{date:<8} {int(row['bets']):>5} {int(row['wins']):>3} {int(row['losses']):>3} "
            f"{int(row['flats']):>3} {int(row['exits']):>3} {wr_str:>6} ${row['pnl']:>+6.1f}"
        )

    wd = (daily["pnl"] > 0).sum()
    ld = (daily["pnl"] < 0).sum()
    print(f"\n  Winning: {wd} | Losing: {ld}")

    # Per-asset
    print(f"\n  Per-asset:")
    for a in ["BTC", "ETH", "SOL", "XRP"]:
        ad = df_exec[df_exec["asset"] == a]
        if len(ad) == 0: continue
        aw = ad["outcome"].isin(["WIN", "WIN_EXIT"]).sum()
        al = ad["outcome"].isin(["LOSS", "LOSS_EXIT"]).sum()
        af = (ad["outcome"] == "FLAT_EXIT").sum()
        amx = int(ad["m10_exit"].sum())
        awr = ad["won_settlement"].mean() * 100
        ahwr = aw / (aw + al) * 100 if (aw + al) > 0 else 0
        held_w = (ad["outcome"] == "WIN").sum()
        held_l = (ad["outcome"] == "LOSS").sum()
        held_wr = held_w / (held_w + held_l) * 100 if (held_w + held_l) > 0 else 0
        print(
            f"    {a}: {len(ad)} sig, Entry:{awr:.0f}%, Real:{ahwr:.0f}%, "
            f"Held:{held_wr:.0f}%, W:{aw} L:{al} F:{af} M10X:{amx}"
        )


if __name__ == "__main__":
    main()
