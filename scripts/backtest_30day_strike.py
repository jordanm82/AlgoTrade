#!/usr/bin/env python3
"""30-day backtest of the strike-relative model with compounding.

Flat 5% sizing, max 100 contracts, max 3 per window.
Tests on the most recent 30 days of data using the saved model.

Usage:
    ./venv/bin/python scripts/backtest_30day_strike.py
"""
import sys, time, pickle
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from data.fetcher import DataFetcher
from data.indicators import add_indicators

ASSETS = {'BTC': 'BTC/USD', 'ETH': 'ETH/USD', 'SOL': 'SOL/USD', 'XRP': 'XRP/USD'}
STARTING_BALANCE = 100.0
RISK_PCT = 0.05
MAX_CONTRACTS = 100
ENTRY_CENTS = 50
MAX_PER_WINDOW = 3


def fetch_candles(fetcher, symbol, timeframe, days):
    all_frames = []
    now_ms = int(time.time() * 1000)
    since = now_ms - days * 86400 * 1000
    batch_size = 300
    tf_ms = {"5m": 300000, "15m": 900000, "1h": 3600000, "4h": 14400000}
    candle_ms = tf_ms.get(timeframe, 900000)
    while since < now_ms:
        try:
            df = fetcher.ohlcv(symbol, timeframe, limit=batch_size, since=since)
            if df is None or df.empty:
                since += batch_size * candle_ms; time.sleep(0.3); continue
            all_frames.append(df)
            since = int(df.index[-1].timestamp() * 1000) + candle_ms
            time.sleep(0.3)
        except Exception:
            since += batch_size * candle_ms; time.sleep(1)
    if not all_frames:
        return pd.DataFrame()
    combined = pd.concat(all_frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def main():
    with open("models/knn_kalshi.pkl", "rb") as f:
        m = pickle.load(f)
    model = m["knn"]
    scaler_m = m["scaler"]
    FEATURES = m["feature_names"]

    print("=" * 80)
    print(f"30-DAY BACKTEST — Strike-Relative Model ({m['model_type']})")
    print(f"Features: {len(FEATURES)} | Sizing: flat 5% of balance, max 100 contracts")
    print("=" * 80)

    fetcher = DataFetcher()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    all_signals = []

    print("\nFetching data...")
    for name, sym in ASSETS.items():
        print(f"  {name}...", end=" ", flush=True)
        df_15m_raw = fetch_candles(fetcher, sym, "15m", 60)
        df_5m = fetch_candles(fetcher, sym, "5m", 35)
        df_1h_raw = fetch_candles(fetcher, sym, "1h", 60)
        df_4h_raw = fetch_candles(fetcher, sym, "4h", 60)

        if df_15m_raw.empty or df_5m.empty:
            print("SKIP"); continue

        df = add_indicators(df_15m_raw)
        df_1h_ind = add_indicators(df_1h_raw) if df_1h_raw is not None and not df_1h_raw.empty else None
        df_4h_ind = add_indicators(df_4h_raw) if df_4h_raw is not None and not df_4h_raw.empty else None

        pct = df["close"].pct_change()
        df["norm_return"] = (pct - pct.rolling(20).mean()) / pct.rolling(20).std()
        df["vol_ratio"] = df["volume"] / df["vol_sma_20"]
        df["ema_slope"] = df["ema_12"].pct_change(3) * 100
        df["price_vs_ema"] = (df["close"] - df["sma_20"]) / df["atr"].replace(0, np.nan)
        df["hourly_return"] = df["close"].pct_change(4) * 100

        count = 0
        for i in range(50, len(df) - 1):
            t = df.index[i]
            if t < cutoff:
                continue

            window = df.iloc[i]
            prev = df.iloc[i - 1]

            strike = float(window["open"])
            if strike <= 0: continue
            atr = float(prev.get("atr", 0))
            if pd.isna(atr) or atr <= 0: continue

            min5_end = t + pd.Timedelta(minutes=5)
            mask_5m = (df_5m.index >= t) & (df_5m.index < min5_end)
            if mask_5m.sum() > 0:
                price_at_min5 = float(df_5m[mask_5m].iloc[-1]["close"])
            else:
                before = df_5m[df_5m.index <= min5_end]
                if len(before) > 0 and (min5_end - before.index[-1]).total_seconds() < 600:
                    price_at_min5 = float(before.iloc[-1]["close"])
                else:
                    continue

            distance = (price_at_min5 - strike) / atr
            window_close = float(window["close"])
            label = 1 if window_close >= strike else 0

            r = prev
            sma_val = float(r.get("sma_20", 0))
            adx_val = float(r.get("adx", 20))
            close_val = float(r.get("close", 0))
            ts_sign = (1 if close_val >= sma_val else -1) if sma_val > 0 else 0
            pve = float(r.get("price_vs_ema", 0))
            hr = float(r.get("hourly_return", 0))
            if pd.isna(pve) or np.isinf(pve): pve = 0
            if pd.isna(hr) or np.isinf(hr): hr = 0

            feat = {
                "macd_15m": float(r.get("macd_hist", 0)),
                "norm_return": float(r.get("norm_return", 0)) if pd.notna(r.get("norm_return")) else 0,
                "ema_slope": float(r.get("ema_slope", 0)) if pd.notna(r.get("ema_slope")) else 0,
                "roc_5": float(r.get("roc_5", 0)),
                "macd_1h": 0.0,
                "price_vs_ema": pve,
                "hourly_return": hr,
                "trend_direction": adx_val * ts_sign,
                "vol_ratio": float(r.get("vol_ratio", 1)) if pd.notna(r.get("vol_ratio")) else 1,
                "adx": adx_val,
                "rsi_1h": 50.0,
                "rsi_4h": 50.0,
                "distance_from_strike": distance,
            }

            if df_1h_ind is not None:
                m1h = df_1h_ind.index <= t
                if m1h.sum() >= 20:
                    r1h = df_1h_ind.loc[m1h].iloc[-1]
                    feat["rsi_1h"] = float(r1h.get("rsi", 50))
                    feat["macd_1h"] = float(r1h.get("macd_hist", 0))
            if df_4h_ind is not None:
                m4h = df_4h_ind.index <= t
                if m4h.sum() >= 10:
                    feat["rsi_4h"] = float(df_4h_ind.loc[m4h].iloc[-1].get("rsi", 50))

            if any(pd.isna(v) or np.isinf(v) for v in feat.values()):
                continue

            X = np.array([feat[f] for f in FEATURES]).reshape(1, -1)
            prob = float(model.predict_proba(scaler_m.transform(X))[0][1])
            pct_val = int(prob * 100)

            if pct_val >= 55:
                side = "YES"
            elif pct_val <= 45:
                side = "NO"
            else:
                continue

            won = (side == "YES" and label == 1) or (side == "NO" and label == 0)
            all_signals.append({"ts": t, "asset": name, "side": side, "prob": prob,
                                "won": won, "label": label, "distance": distance})
            count += 1

        print(f"{count} signals")

    df_sig = pd.DataFrame(all_signals).sort_values("ts").reset_index(drop=True)
    yes_n = len(df_sig[df_sig["side"] == "YES"])
    no_n = len(df_sig[df_sig["side"] == "NO"])
    print(f"\nTotal signals: {len(df_sig)} ({yes_n}Y / {no_n}N)")

    # === Flat P&L ===
    wr = df_sig["won"].mean() * 100
    flat_pnl = df_sig["won"].sum() * 0.50 - (~df_sig["won"]).sum() * 0.50
    y_wr = df_sig[df_sig["side"] == "YES"]["won"].mean() * 100 if yes_n > 0 else 0
    n_wr = df_sig[df_sig["side"] == "NO"]["won"].mean() * 100 if no_n > 0 else 0

    print(f"\n{'=' * 80}")
    print("FLAT P&L (50c per contract)")
    print(f"{'=' * 80}")
    print(f"  WR:   {wr:.1f}% ({df_sig['won'].sum()}W / {(~df_sig['won']).sum()}L)")
    print(f"  YES:  {y_wr:.1f}% | NO: {n_wr:.1f}%")
    print(f"  Y:N ratio: {yes_n/no_n:.1f}:1" if no_n > 0 else "  Y:N ratio: inf")
    print(f"  P&L:  ${flat_pnl:+.2f}")

    # === Per-day breakdown ===
    print(f"\n{'=' * 80}")
    print("DAILY BREAKDOWN")
    print(f"{'=' * 80}")
    df_sig["date"] = df_sig["ts"].apply(lambda t: pd.Timestamp(t).strftime("%m/%d"))
    daily = df_sig.groupby("date").agg(
        bets=("won", "count"),
        wins=("won", "sum"),
    )
    daily["wr"] = daily["wins"] / daily["bets"] * 100
    daily["pnl"] = daily["wins"] * 0.50 - (daily["bets"] - daily["wins"]) * 0.50

    print(f"{'Date':<8} {'Bets':>5} {'WR':>6} {'P&L':>8}")
    print("-" * 30)
    for date, row in daily.iterrows():
        print(f"{date:<8} {int(row['bets']):>5} {row['wr']:>5.0f}% ${row['pnl']:>+6.1f}")

    winning_days = (daily["pnl"] > 0).sum()
    losing_days = (daily["pnl"] < 0).sum()
    even_days = (daily["pnl"] == 0).sum()
    print(f"\n  Winning days: {winning_days} | Losing days: {losing_days} | Even: {even_days}")

    # === Compounding simulation ===
    print(f"\n{'=' * 80}")
    print(f"COMPOUNDING: $100 start, 5% per bet, max 100 contracts, max 3/window")
    print(f"{'=' * 80}")

    df_sig["window"] = df_sig["ts"].apply(lambda t: pd.Timestamp(t).floor("15min"))
    balance = STARTING_BALANCE
    peak = balance
    max_dd = 0
    bets_placed = 0
    cap_hits = 0

    weekly_equity = [(0, balance)]
    week_count = 0

    for window_ts, group in df_sig.groupby("window"):
        window_bets = group.nlargest(MAX_PER_WINDOW, "prob") if len(group) > MAX_PER_WINDOW else group

        for _, row in window_bets.iterrows():
            risk = balance * RISK_PCT
            contracts = max(1, min(int(risk / (ENTRY_CENTS / 100)), MAX_CONTRACTS))
            cost = contracts * (ENTRY_CENTS / 100)

            if cost > balance:
                continue
            if contracts == MAX_CONTRACTS:
                cap_hits += 1

            bets_placed += 1
            if row["won"]:
                balance += contracts * ((100 - ENTRY_CENTS) / 100)
            else:
                balance -= cost

            if balance > peak: peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd: max_dd = dd

        if bets_placed > 0 and bets_placed % max(1, len(df_sig) // 6) < MAX_PER_WINDOW:
            weekly_equity.append((bets_placed, balance))

    weekly_equity.append((bets_placed, balance))
    ret = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    print(f"  Bets placed: {bets_placed}")
    print(f"  Start: ${STARTING_BALANCE:.0f} → End: ${balance:,.2f} ({ret:+,.1f}%)")
    print(f"  Max drawdown: {max_dd:.1f}%")
    if cap_hits > 0:
        print(f"  Hit contract cap: {cap_hits} times")

    seen = set()
    curve_parts = []
    for _, eq in weekly_equity:
        key = f"${eq:,.0f}"
        if key not in seen:
            seen.add(key)
            curve_parts.append(key)
    print(f"  Equity: {' → '.join(curve_parts)}")


if __name__ == "__main__":
    main()
