#!/usr/bin/env python3
"""Live test: fetch real-time leading indicators for BTC/ETH/SOL/XRP and score them.

Compares the lagging-only score (from OHLCV) vs the enhanced score
(lagging + order book + trade flow + cross-asset momentum).

Usage:
    source venv/bin/activate
    python live_test_enhanced.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.fetcher import DataFetcher
from data.indicators import add_indicators
from data.market_data import get_all_signals, get_cross_asset_momentum, TRACKED_PAIRS
from strategy.strategies.kalshi_predictor import KalshiPredictor


ASSETS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "XRP": "XRP/USDT",
}


def main():
    print("=" * 75)
    print("ENHANCED KALSHI PREDICTOR - LIVE SIGNAL TEST")
    print("=" * 75)

    fetcher = DataFetcher()
    predictor = KalshiPredictor()

    # Fetch cross-asset momentum first (used for all alts)
    print("\nFetching cross-asset momentum ...")
    try:
        cross = get_cross_asset_momentum()
        btc_dir = cross.get("market_direction", 0)
        btc_leads = cross.get("btc_leads", 0)
        print(f"  BTC 24h change: {btc_dir:+.2f}%")
        print(f"  BTC vs alts divergence: {btc_leads:+.2f}%")
        for sym in TRACKED_PAIRS:
            info = cross.get(sym, {})
            print(f"  {sym}: ${info.get('price', 0):,.2f}  ({info.get('change_pct', 0):+.2f}%)")
    except Exception as e:
        print(f"  Error fetching cross-asset data: {e}")
        cross = {}

    print()

    for asset_name, symbol in ASSETS.items():
        print("-" * 75)
        print(f"  {asset_name} ({symbol})")
        print("-" * 75)

        # 1. Fetch OHLCV + indicators (lagging)
        try:
            df_raw = fetcher.ohlcv(symbol, "15m", limit=100)
            df = add_indicators(df_raw)
            price = float(df.iloc[-1]["close"])
            rsi = float(df.iloc[-1].get("rsi", 50))
            print(f"  Price: ${price:,.2f}  |  RSI: {rsi:.1f}")
        except Exception as e:
            print(f"  Error fetching OHLCV: {e}")
            continue

        # 2. Score with lagging only
        signal_lagging = predictor.score(df)
        if signal_lagging:
            print(f"  Lagging-only:  {signal_lagging.direction} {signal_lagging.confidence}%")
        else:
            print(f"  Lagging-only:  NEUTRAL (no signal)")

        # 3. Fetch leading indicators
        try:
            md = get_all_signals(symbol)
            ob = md["order_book"]
            tf = md["trade_flow"]
            print(f"  Order Book:    imbalance={ob['imbalance']:+.4f}  "
                  f"spread={ob['spread_pct']:.4f}%  wall_ratio={ob['wall_ratio']:.1f}")
            print(f"  Trade Flow:    net_flow={tf['net_flow']:+.4f}  "
                  f"buy_ratio={tf['buy_ratio']:.2f}  large_bias={tf['large_trade_bias']:+.4f}")
        except Exception as e:
            print(f"  Error fetching leading data: {e}")
            md = {}

        # 4. Score with full enhanced data
        if md:
            md["cross_asset"] = {
                "market_direction": cross.get("market_direction", 0),
            }
            signal_enhanced = predictor.score(df, market_data=md)
            if signal_enhanced:
                print(f"  Enhanced:      {signal_enhanced.direction} {signal_enhanced.confidence}%")
                # Show component breakdown
                comps = signal_enhanced.components
                parts = []
                for name in ["rsi", "bb", "macd", "volume", "momentum", "rsi_trend",
                             "order_book", "trade_flow", "large_trade", "spread", "cross_asset"]:
                    c = comps.get(name, {})
                    up = c.get("up", c.get("score", 0))
                    down = c.get("down", 0)
                    if up > 0:
                        parts.append(f"{name}:+{up}UP")
                    if down > 0:
                        parts.append(f"{name}:+{down}DN")
                if parts:
                    print(f"  Components:    {', '.join(parts)}")

                # Compare
                lag_conf = signal_lagging.confidence if signal_lagging else 0
                lag_dir = signal_lagging.direction if signal_lagging else "NEUTRAL"
                enh_conf = signal_enhanced.confidence
                enh_dir = signal_enhanced.direction
                if lag_dir == enh_dir:
                    delta = enh_conf - lag_conf
                    print(f"  Delta:         {delta:+d} pts (leading indicators {'boost' if delta > 0 else 'dampen'} signal)")
                else:
                    print(f"  Delta:         DIRECTION CHANGED ({lag_dir} -> {enh_dir})")
            else:
                print(f"  Enhanced:      NEUTRAL (no signal)")
        print()
        time.sleep(0.5)

    print("=" * 75)
    print("Live test complete.")


if __name__ == "__main__":
    main()
