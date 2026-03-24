"""Convenience script: python run_backtest.py <symbol> <timeframe> <limit>
Fetches data from Binance, adds indicators, runs SMA crossover backtest, prints results."""
import sys
import json
from data.fetcher import DataFetcher
from data.indicators import add_indicators
from strategy.backtest import run_backtest
import pandas as pd


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    print(f"Fetching {limit} {timeframe} candles for {symbol}...")
    fetcher = DataFetcher()
    df = fetcher.ohlcv(symbol, timeframe, limit=limit)
    df = add_indicators(df)
    print(f"Got {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    results = run_backtest(
        df=df,
        buy_condition=lambda row: (
            pd.notna(row.get("sma_20")) and pd.notna(row.get("sma_50"))
            and row["sma_20"] > row["sma_50"]
            and pd.notna(row.get("rsi")) and row["rsi"] < 60
        ),
        sell_condition=lambda row: (
            pd.notna(row.get("sma_20")) and pd.notna(row.get("sma_50"))
            and row["sma_20"] < row["sma_50"]
        ),
        cash=10000,
        commission=0.001,
    )

    # Remove non-serializable stats object
    del results["stats"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
