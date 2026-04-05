#!/usr/bin/env python3
"""Backtest matching EXACT live trading flow.

Live flow replicated:
  1. Minute 0: Previous 15m candle indicators + 5m open (BRTI proxy) → M0 entry
  2. Minute 10: Same indicators + price at minute 10 → M10 hold/exit
  3. Settlement: Kalshi result (yes/no)

Data sources (must match live):
  - Labels: Kalshi settled market result (yes/no)
  - Strike: Kalshi floor_strike
  - Price at min 0: Coinbase + Bitstamp 5m candle OPEN averaged
  - Price at min 10: Coinbase + Bitstamp 5m candle at min 10
  - Indicators: Coinbase 15m/1h/4h (previous completed candle)

Usage:
    ./venv/bin/python scripts/backtest_kalshi_labels.py [--days 30]
"""
import argparse
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
from config.settings import KALSHI_KEY_FILE, KALSHI_API_KEY_ID

SERIES = {
    "BTC": "KXBTC15M",
    "ETH": "KXETH15M",
    "SOL": "KXSOL15M",
    "XRP": "KXXRP15M",
}
ASSETS_SYMS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"}

STARTING_BALANCE = 100.0
RISK_PCT = 0.05
MAX_CONTRACTS = 100
ENTRY_CENTS = 60
MAX_PER_WINDOW = 3
M10_EXIT_CENTS = 10  # floor sell price on early exit


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
    """Get averaged price from Coinbase + Bitstamp 5m candles at target time."""
    prices = []
    for df_5m in [cb_5m, bs_5m]:
        if df_5m.empty:
            continue
        mask = (df_5m.index >= target_time) & (df_5m.index < target_time + timedelta(minutes=5))
        if mask.sum() > 0:
            prices.append(float(df_5m[mask].iloc[0][field]))
        else:
            before = df_5m[df_5m.index <= target_time]
            if len(before) > 0 and (target_time - before.index[-1]).total_seconds() < 600:
                prices.append(float(before.iloc[-1]["close"]))
    if not prices:
        return None
    return sum(prices) / len(prices)


def build_features(prev, df_1h, df_4h, ws_naive, distance, *,
                    kalshi_extra=None, atr_pctile_val=0.5):
    """Build feature vector from previous 15m candle + 1h/4h + Kalshi context."""
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
        m1h = df_1h.index <= ws_naive
        if m1h.sum() >= 20:
            feat["rsi_1h"] = float(df_1h.loc[m1h].iloc[-1].get("rsi", 50))
            feat["macd_1h"] = float(df_1h.loc[m1h].iloc[-1].get("macd_hist", 0))
    if df_4h is not None:
        m4h = df_4h.index <= ws_naive
        if m4h.sum() >= 10:
            feat["rsi_4h"] = float(df_4h.loc[m4h].iloc[-1].get("rsi", 50))

    # Kalshi-specific features
    kx = kalshi_extra or {}
    feat["prev_result"] = kx.get("prev_result", 0.5)
    feat["prev_3_yes_pct"] = kx.get("prev_3_yes_pct", 0.5)
    feat["streak_length"] = kx.get("streak_length", 0)
    feat["strike_delta"] = kx.get("strike_delta", 0.0)
    feat["strike_trend_3"] = kx.get("strike_trend_3", 0.0)

    # Time features
    hour = ws_naive.hour if hasattr(ws_naive, 'hour') else 12
    feat["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feat["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))

    # Technical additions
    feat["rsi_alignment"] = (
        (1 if feat["rsi_1h"] >= 50 else -1) *
        (1 if feat["rsi_4h"] >= 50 else -1)
    )
    feat["atr_percentile"] = atr_pctile_val
    feat["rsi_15m"] = float(prev.get("rsi", 50))

    # Bollinger Band Width — regime detection
    bb_upper = float(prev.get("bb_upper", 0))
    bb_lower = float(prev.get("bb_lower", 0))
    bb_mid = float(prev.get("sma_20", 0))
    feat["bbw"] = ((bb_upper - bb_lower) / bb_mid * 100) if bb_mid > 0 else 0

    if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
        return None
    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--threshold", type=int, default=55, help="Confidence threshold (55=default)")
    parser.add_argument("--exit-min", type=int, default=None, help="Min early exit price (random range)")
    parser.add_argument("--exit-max", type=int, default=None, help="Max early exit price (random range)")
    parser.add_argument("--m10-threshold", type=int, default=55, help="M10 exit confidence threshold")
    args = parser.parse_args()

    THRESHOLD = args.threshold
    M10_THRESHOLD = args.m10_threshold
    random_exit = args.exit_min is not None and args.exit_max is not None

    # Load M0 (entry) model
    with open("models/knn_kalshi.pkl", "rb") as f:
        m0_data = pickle.load(f)
    m0_model = m0_data["knn"]
    m0_scaler = m0_data["scaler"]
    FEATURES = m0_data["feature_names"]

    # Load M10 (confirmation) model — may have different feature set (e.g. includes rsi_15m)
    m10_model = None
    m10_scaler = None
    m10_path = Path("models/m10_kalshi.pkl")
    if m10_path.exists():
        with open(m10_path, "rb") as f:
            m10_data = pickle.load(f)
        m10_model = m10_data.get("knn") or m10_data.get("model")
        m10_scaler = m10_data["scaler"]
        print("M10 model loaded — will simulate minute-10 exits")
    else:
        print("WARNING: No M10 model found — backtest will NOT simulate early exits")

    print("=" * 80)
    print(f"BACKTEST — Live Flow Simulation ({args.days} days)")
    print(f"M0: {m0_data.get('model_type')} | Labels: {m0_data.get('labels', '?')}")
    print(f"Entry: minute 0 (5m open) | M10 exit: minute 10 (5m price)")
    exit_desc = f"random {args.exit_min}-{args.exit_max}c" if random_exit else f"floor {M10_EXIT_CENTS}c"
    print(f"Max entry: {ENTRY_CENTS}c | M10 exit: {exit_desc} | Threshold: {THRESHOLD}")
    print("=" * 80)

    client = KalshiClient(api_key_id=KALSHI_API_KEY_ID, private_key_path=str(KALSHI_KEY_FILE))
    fetcher = DataFetcher()
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    # Fetch Kalshi settlements
    print("\nFetching Kalshi settlements...")
    kalshi_markets = {}
    for asset, series in SERIES.items():
        markets = []
        cursor = ""
        for page in range(30):
            params = {"series_ticker": series, "status": "settled", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            resp = client._get("/trade-api/v2/markets", params)
            batch = resp.get("markets", [])
            cursor = resp.get("cursor", "")
            markets.extend(batch)
            if not batch or not cursor:
                break
        filtered = [mk for mk in markets
                    if mk.get("close_time") and
                    datetime.fromisoformat(mk["close_time"].replace("Z", "+00:00")) >= cutoff]
        kalshi_markets[asset] = filtered
        print(f"  {asset}: {len(filtered)} markets in last {args.days} days")

    # Fetch 5m data from both exchanges
    print("\nFetching 5m candles (Coinbase + Bitstamp)...")
    five_m = {}
    for asset, sym in ASSETS_SYMS.items():
        five_m[asset] = {}
        for ex in ["coinbase", "bitstamp"]:
            print(f"  {asset} {ex}...", end=" ", flush=True)
            df = fetch_5m_history(sym, ex, args.days + 5)
            five_m[asset][ex] = df
            print(f"{len(df)} candles")

    # Fetch indicator data
    print("\nFetching indicator data...")
    ind_data = {}
    for asset, sym in ASSETS_SYMS.items():
        df_15m = add_indicators(fetch_candles(fetcher, sym, "15m", args.days + 30))
        df_1h = fetch_candles(fetcher, sym, "1h", args.days + 30)
        df_4h = fetch_candles(fetcher, sym, "4h", args.days + 30)
        df_1h_ind = add_indicators(df_1h) if not df_1h.empty else None
        df_4h_ind = add_indicators(df_4h) if not df_4h.empty else None

        pct = df_15m["close"].pct_change()
        df_15m["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df_15m["vol_ratio"] = df_15m["volume"] / df_15m["vol_sma_20"]
        df_15m["ema_slope"] = df_15m["ema_12"].pct_change(3) * 100
        df_15m["price_vs_ema"] = (df_15m["close"] - df_15m["sma_20"]) / df_15m["atr"].replace(0, np.nan)
        df_15m["hourly_return"] = df_15m["close"].pct_change(4) * 100

        ind_data[asset] = {"15m": df_15m, "1h": df_1h_ind, "4h": df_4h_ind}
        print(f"  {asset}: {len(df_15m)} 15m candles")

    # Run backtest
    print("\nScoring with M0 entry + M10 exit simulation...")
    all_signals = []

    for asset in SERIES:
        markets = kalshi_markets.get(asset, [])
        df_15m = ind_data[asset]["15m"]
        df_1h = ind_data[asset]["1h"]
        df_4h = ind_data[asset]["4h"]
        cb_5m = five_m[asset].get("coinbase", pd.DataFrame())
        bs_5m = five_m[asset].get("bitstamp", pd.DataFrame())

        # Sort chronologically for lookback
        markets = sorted(markets, key=lambda x: x.get("close_time", ""))

        # Pre-compute ATR percentile
        atr_s = df_15m["atr"].dropna()
        atr_r20 = atr_s.rolling(20)
        atr_pctile = ((atr_s - atr_r20.min()) / (atr_r20.max() - atr_r20.min())).fillna(0.5)

        count = 0
        for mi, mk in enumerate(markets):
            strike = float(mk.get("floor_strike") or 0)
            result = mk.get("result", "")
            close_time = mk.get("close_time", "")
            if not strike or not result or not close_time:
                continue

            label = 1 if result == "yes" else 0
            close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            window_start = close_dt - timedelta(minutes=15)
            ws_naive = window_start.replace(tzinfo=None)

            # --- MINUTE 0: Entry decision ---
            price_at_min0 = get_avg_price(cb_5m, bs_5m, ws_naive, field="open")
            if price_at_min0 is None:
                continue

            prev_candles = df_15m[df_15m.index < ws_naive]
            if len(prev_candles) < 20:
                continue
            prev = prev_candles.iloc[-1]
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0:
                continue

            m0_distance = (price_at_min0 - strike) / atr

            # Kalshi lookback features
            prev_results = []
            prev_strikes = []
            for lb in range(1, 4):
                if mi - lb >= 0:
                    pm = markets[mi - lb]
                    pr = pm.get("result", "")
                    ps = float(pm.get("floor_strike") or 0)
                    if pr:
                        prev_results.append(1 if pr == "yes" else 0)
                    if ps:
                        prev_strikes.append(ps)

            streak = 0
            if prev_results:
                last_r = prev_results[0]
                for r in prev_results:
                    if r == last_r:
                        streak += 1
                    else:
                        break
                streak = streak if last_r == 1 else -streak

            s_delta = (strike - prev_strikes[0]) / atr if prev_strikes and atr > 0 else 0.0
            if len(prev_strikes) >= 2 and atr > 0:
                deltas = [strike - prev_strikes[0]]
                for k in range(len(prev_strikes) - 1):
                    deltas.append(prev_strikes[k] - prev_strikes[k + 1])
                s_trend = sum(d / atr for d in deltas) / len(deltas)
            else:
                s_trend = s_delta

            kalshi_extra = {
                "prev_result": prev_results[0] if prev_results else 0.5,
                "prev_3_yes_pct": sum(prev_results) / len(prev_results) if prev_results else 0.5,
                "streak_length": streak,
                "strike_delta": s_delta,
                "strike_trend_3": s_trend,
            }

            # ATR percentile for this candle
            prev_atr_p = atr_pctile[atr_pctile.index < ws_naive]
            atr_p_val = float(prev_atr_p.iloc[-1]) if len(prev_atr_p) > 0 else 0.5

            feat = build_features(prev, df_1h, df_4h, ws_naive, m0_distance,
                                  kalshi_extra=kalshi_extra, atr_pctile_val=atr_p_val)
            if feat is None:
                continue

            X = np.array([feat[f] for f in FEATURES]).reshape(1, -1)
            m0_prob = float(m0_model.predict_proba(m0_scaler.transform(X))[0][1])
            m0_pct = int(m0_prob * 100)

            if m0_pct >= THRESHOLD:
                side = "yes"
            elif m0_pct <= (100 - THRESHOLD):
                side = "no"
            else:
                continue  # SKIP — no signal

            # --- MINUTE 10: M10 confirmation ---
            m10_exit = False
            m10_prob_val = None
            exit_price_cents = M10_EXIT_CENTS  # floor
            if m10_model is not None:
                min10_time = ws_naive + timedelta(minutes=10)
                price_at_min10 = get_avg_price(cb_5m, bs_5m, min10_time, field="open")
                if price_at_min10 is not None:
                    m10_distance = (price_at_min10 - strike) / atr
                    m10_feat = build_features(prev, df_1h, df_4h, ws_naive, m10_distance,
                                              kalshi_extra=kalshi_extra, atr_pctile_val=atr_p_val)
                    if m10_feat is not None:
                        m10_features = m10_data.get("feature_names", FEATURES)
                        X10 = np.array([m10_feat[f] for f in m10_features]).reshape(1, -1)
                        m10_prob_val = float(m10_model.predict_proba(m10_scaler.transform(X10))[0][1])
                        m10_pct = int(m10_prob_val * 100)
                        m10_side = "yes" if m10_pct >= M10_THRESHOLD else "no" if m10_pct <= (100 - M10_THRESHOLD) else "skip"
                        # M10 disagrees with our bet → exit
                        if m10_side != "skip" and m10_side != side:
                            m10_exit = True
                            if random_exit:
                                # Random exit price in specified range
                                import random
                                exit_price_cents = random.randint(args.exit_min, args.exit_max)
                            else:
                                # Estimate exit price from contract value at minute 10
                                if side == "yes":
                                    exit_price_cents = max(M10_EXIT_CENTS, int(m10_prob_val * 100))
                                else:
                                    exit_price_cents = max(M10_EXIT_CENTS, int((1 - m10_prob_val) * 100))

            # --- Result ---
            won_settlement = (side == "yes" and label == 1) or (side == "no" and label == 0)

            if m10_exit:
                outcome = "PL"  # early exit
                won = False
            else:
                outcome = "WIN" if won_settlement else "LOSS"
                won = won_settlement

            all_signals.append({
                "ts": close_dt,
                "asset": asset,
                "side": side.upper(),
                "m0_prob": m0_prob,
                "m10_prob": m10_prob_val,
                "m10_exit": m10_exit,
                "exit_price": exit_price_cents,
                "outcome": outcome,
                "won": won,
                "won_settlement": won_settlement,
                "label": label,
                "distance_m0": m0_distance,
                "strike": strike,
                "settled": float(mk.get("expiration_value", 0)),
            })
            count += 1
        print(f"  {asset}: {count} signals")

    df_sig = pd.DataFrame(all_signals).sort_values("ts").reset_index(drop=True)
    total = len(df_sig)
    yes_n = len(df_sig[df_sig["side"] == "YES"])
    no_n = len(df_sig[df_sig["side"] == "NO"])
    print(f"\nTotal signals: {total} ({yes_n}Y / {no_n}N)")

    # --- Results: Entry-only WR (no M10) ---
    entry_wr = df_sig["won_settlement"].mean() * 100
    y_wr = df_sig[df_sig["side"] == "YES"]["won_settlement"].mean() * 100 if yes_n > 0 else 0
    n_wr = df_sig[df_sig["side"] == "NO"]["won_settlement"].mean() * 100 if no_n > 0 else 0

    print(f"\n{'=' * 80}")
    print("ENTRY-ONLY (M0 model, no M10 exit)")
    print(f"{'=' * 80}")
    print(f"  WR:   {entry_wr:.1f}% ({df_sig['won_settlement'].sum()}W / {(~df_sig['won_settlement']).sum()}L)")
    print(f"  YES:  {y_wr:.1f}% | NO: {n_wr:.1f}%")
    print(f"  Y:N ratio: {yes_n/no_n:.1f}:1" if no_n > 0 else "  Y:N ratio: inf")

    # --- Results: With M10 exit ---
    if m10_model is not None:
        n_exits = df_sig["m10_exit"].sum()
        n_held = total - n_exits
        wins = (df_sig["outcome"] == "WIN").sum()
        losses = (df_sig["outcome"] == "LOSS").sum()
        pls = (df_sig["outcome"] == "PL").sum()

        # Of the M10 exits, how many WOULD have won if held?
        exits_df = df_sig[df_sig["m10_exit"]]
        good_exits = (~exits_df["won_settlement"]).sum()  # would have lost → good exit
        bad_exits = exits_df["won_settlement"].sum()  # would have won → bad exit

        print(f"\n{'=' * 80}")
        print("WITH M10 EXIT (live trading simulation)")
        print(f"{'=' * 80}")
        print(f"  Held to settlement: {n_held} | M10 early exits: {n_exits}")
        print(f"  W:{wins} L:{losses} PL:{pls}")
        if n_held > 0:
            held_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            print(f"  Held WR: {held_wr:.1f}% (of {wins + losses} held bets)")
        print(f"  M10 exits: {good_exits} GOOD (saved from loss) / {bad_exits} BAD (missed win)")
        if n_exits > 0:
            print(f"  M10 accuracy: {good_exits / n_exits * 100:.0f}%")

    # --- P&L simulation ---
    print(f"\n{'=' * 80}")
    print(f"P&L: ${STARTING_BALANCE} start, {RISK_PCT*100:.0f}% per bet, max {MAX_PER_WINDOW}/window")
    print(f"Entry @ {ENTRY_CENTS}c | M10 exit floor @ {M10_EXIT_CENTS}c")
    print(f"{'=' * 80}")

    DAILY_LOSS_CAP = 0.20  # stop trading if daily loss exceeds 20% of start-of-day balance

    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0
    bets_placed = 0
    bets_skipped_cap = 0
    pnl_wins = 0
    pnl_losses = 0
    pnl_exits = 0

    # Track daily state for loss cap
    current_day = None
    day_start_balance = balance
    day_halted = False

    for _, group in df_sig.groupby(df_sig["ts"].dt.floor("15min")):
        window_bets = group.nlargest(MAX_PER_WINDOW, "m0_prob")
        for _, row in window_bets.iterrows():
            # Check if new day
            bet_day = row["ts"].strftime("%m/%d")
            if bet_day != current_day:
                current_day = bet_day
                day_start_balance = balance
                day_halted = False

            # Daily loss cap: skip if already down 20% today
            if day_halted:
                bets_skipped_cap += 1
                continue
            day_loss_pct = (day_start_balance - balance) / day_start_balance if day_start_balance > 0 else 0
            if day_loss_pct >= DAILY_LOSS_CAP:
                day_halted = True
                bets_skipped_cap += 1
                continue

            risk = balance * RISK_PCT
            contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
            cost = contracts * (ENTRY_CENTS / 100)
            if cost > balance:
                continue
            bets_placed += 1

            if row["outcome"] == "WIN":
                profit = contracts * ((100 - ENTRY_CENTS) / 100)
                balance += profit
                pnl_wins += profit
            elif row["outcome"] == "LOSS":
                balance -= cost
                pnl_losses += cost
            elif row["outcome"] == "PL":
                # M10 exit: sell at estimated contract value (floor = 10c)
                exit_px = int(row.get("exit_price", M10_EXIT_CENTS))
                pnl_per = exit_px - ENTRY_CENTS  # negative (selling below entry)
                exit_pnl = contracts * (pnl_per / 100)
                balance += exit_pnl
                pnl_exits += abs(exit_pnl) if exit_pnl < 0 else -exit_pnl

            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            dd_dollars = peak - balance
            if dd > max_dd:
                max_dd = dd
                max_dd_dollars = dd_dollars
                max_dd_peak = peak
                max_dd_trough = balance

    ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    net_pnl = pnl_wins - pnl_losses - pnl_exits
    print(f"  Bets placed: {bets_placed} (skipped {bets_skipped_cap} from daily loss cap)")
    print(f"  Gross won:    ${pnl_wins:+,.2f}")
    print(f"  Gross lost:   ${-pnl_losses:+,.2f}")
    print(f"  Early exits:  ${-pnl_exits:+,.2f}")
    print(f"  Net P&L:      ${net_pnl:+,.2f}")
    print(f"  Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
    print(f"  Max drawdown: {max_dd:.1f}% (${max_dd_dollars:,.2f} — from ${max_dd_peak:,.2f} to ${max_dd_trough:,.2f})")
    print(f"  Daily loss cap: {DAILY_LOSS_CAP*100:.0f}%")

    # Daily breakdown
    print(f"\n{'=' * 80}")
    print("DAILY BREAKDOWN")
    print(f"{'=' * 80}")
    df_sig["date"] = df_sig["ts"].apply(lambda t: t.strftime("%m/%d"))
    # Compute per-signal flat P&L for daily aggregation
    def signal_pnl(row):
        if row["outcome"] == "WIN":
            return (100 - ENTRY_CENTS) / 100
        elif row["outcome"] == "LOSS":
            return -ENTRY_CENTS / 100
        else:  # PL
            return (row.get("exit_price", M10_EXIT_CENTS) - ENTRY_CENTS) / 100
    df_sig["flat_pnl"] = df_sig.apply(signal_pnl, axis=1)

    daily = df_sig.groupby("date").agg(
        bets=("outcome", "count"),
        wins=("outcome", lambda x: (x == "WIN").sum()),
        losses=("outcome", lambda x: (x == "LOSS").sum()),
        exits=("outcome", lambda x: (x == "PL").sum()),
        pnl=("flat_pnl", "sum"),
    )
    daily["wr"] = daily["wins"] / (daily["wins"] + daily["losses"]) * 100

    print(f"{'Date':<8} {'Bets':>5} {'W':>3} {'L':>3} {'PL':>3} {'WR':>6} {'P&L':>8}")
    print("-" * 42)
    for date, row in daily.iterrows():
        held = int(row["wins"] + row["losses"])
        wr_str = f"{row['wr']:.0f}%" if held > 0 else "  --"
        print(f"{date:<8} {int(row['bets']):>5} {int(row['wins']):>3} {int(row['losses']):>3} "
              f"{int(row['exits']):>3} {wr_str:>6} ${row['pnl']:>+6.1f}")

    winning_days = (daily["pnl"] > 0).sum()
    losing_days = (daily["pnl"] < 0).sum()
    be_days = (daily["pnl"] == 0).sum()
    print(f"\n  Winning days: {winning_days} | Losing days: {losing_days} | Break-even: {be_days}")

    # --- Losing day analysis ---
    if losing_days > 0:
        print(f"\n{'=' * 80}")
        print("LOSING DAY ANALYSIS")
        print(f"{'=' * 80}")

        loss_dates = daily[daily["pnl"] < 0].index.tolist()
        # Check if consecutive
        for i, date in enumerate(loss_dates):
            adj = []
            if i > 0 and loss_dates[i-1] == daily.index[daily.index.get_loc(date) - 1]:
                adj.append(loss_dates[i-1])
            if i < len(loss_dates) - 1 and loss_dates[i+1] == daily.index[daily.index.get_loc(date) + 1]:
                adj.append(loss_dates[i+1])
            streak = "consecutive" if adj else "isolated"

            row = daily.loc[date]
            print(f"\n  {date}: {streak} | P&L ${row['pnl']:+.1f} | "
                  f"W:{int(row['wins'])} L:{int(row['losses'])} PL:{int(row['exits'])} "
                  f"WR:{row['wr']:.0f}%")

        # Per-asset and side breakdown on losing days
        df_sig["date"] = df_sig["ts"].apply(lambda t: t.strftime("%m/%d"))
        loss_signals = df_sig[df_sig["date"].isin(loss_dates)]

        print(f"\n  Aggregate stats on {len(loss_dates)} losing days:")
        print(f"  Total signals: {len(loss_signals)}")

        # Side breakdown
        for side_val in ["YES", "NO"]:
            s = loss_signals[loss_signals["side"] == side_val]
            if len(s) == 0:
                continue
            w = (s["outcome"] == "WIN").sum()
            l = (s["outcome"] == "LOSS").sum()
            pl = (s["outcome"] == "PL").sum()
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"    {side_val}: {len(s)} signals | W:{w} L:{l} PL:{pl} | Held WR:{wr:.0f}%")

        # Asset breakdown
        for asset in sorted(loss_signals["asset"].unique()):
            a = loss_signals[loss_signals["asset"] == asset]
            w = (a["outcome"] == "WIN").sum()
            l = (a["outcome"] == "LOSS").sum()
            pl = (a["outcome"] == "PL").sum()
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            avg_dist = a["distance_m0"].mean()
            print(f"    {asset}: {len(a)} signals | W:{w} L:{l} PL:{pl} | "
                  f"Held WR:{wr:.0f}% | Avg dist:{avg_dist:+.2f}")

        # Hour distribution on losing days vs all days
        loss_hours = loss_signals["ts"].apply(lambda t: t.hour)
        all_hours = df_sig["ts"].apply(lambda t: t.hour)
        print(f"\n  Hour distribution (losing days vs all):")
        for h in sorted(loss_hours.unique()):
            loss_ct = (loss_hours == h).sum()
            all_ct = (all_hours == h).sum()
            loss_pct = loss_ct / len(loss_signals) * 100
            all_pct = all_ct / len(df_sig) * 100
            if abs(loss_pct - all_pct) > 2:  # only show notable differences
                print(f"    Hour {h:02d}: {loss_pct:.0f}% of losing-day signals vs {all_pct:.0f}% overall"
                      f" {'← OVER-REPRESENTED' if loss_pct > all_pct + 3 else ''}")


if __name__ == "__main__":
    main()
