# strategy/backtest.py
import pandas as pd
from backtesting import Strategy as BtStrategy
from backtesting.lib import FractionalBacktest
from typing import Callable


def run_backtest(
    df: pd.DataFrame,
    buy_condition: Callable,
    sell_condition: Callable,
    cash: float = 10000,
    commission: float = 0.001,
    size: float = 0.1,
) -> dict:
    """Run a backtest on OHLCV data with buy/sell conditions.

    buy_condition: callable(row) -> bool
    sell_condition: callable(row) -> bool
    Returns dict with equity_final, max_drawdown, num_trades, sharpe, win_rate.
    """
    # backtesting.py expects capitalized column names
    bt_df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    # Keep only required + indicator columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    indicator_cols = [c for c in bt_df.columns if c not in required]

    # Store conditions in closure
    _buy = buy_condition
    _sell = sell_condition
    _indicator_cols = indicator_cols
    _size = size

    class DynamicStrategy(BtStrategy):
        def init(self):
            pass

        def next(self):
            row = {}
            current_df = self.data.df
            for col in _indicator_cols:
                row[col] = current_df[col].iloc[-1]
            row["close"] = self.data.Close[-1]

            if not self.position and _buy(pd.Series(row)):
                self.buy(size=_size)
            elif self.position and _sell(pd.Series(row)):
                self.position.close()

    bt = FractionalBacktest(
        bt_df, DynamicStrategy, cash=cash, commission=commission,
    )
    stats = bt.run()

    return {
        "equity_final": float(stats["Equity Final [$]"]),
        "equity_peak": float(stats["Equity Peak [$]"]),
        "max_drawdown": float(stats["Max. Drawdown [%]"]),
        "num_trades": int(stats["# Trades"]),
        "sharpe": float(stats["Sharpe Ratio"]) if pd.notna(stats["Sharpe Ratio"]) else 0.0,
        "win_rate": float(stats["Win Rate [%]"]) if pd.notna(stats["Win Rate [%]"]) else 0.0,
        "return_pct": float(stats["Return [%]"]),
        "stats": stats,
    }
