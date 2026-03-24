"""Backtester with compounding equity, long/short, leverage, and stop-losses."""
import pandas as pd
import numpy as np
from typing import Callable


def compound_backtest(
    df: pd.DataFrame,
    buy_fn: Callable | None = None,
    sell_fn: Callable | None = None,
    short_fn: Callable | None = None,
    cover_fn: Callable | None = None,
    initial_equity: float = 1000.0,
    size_pct: float = 0.10,
    leverage: int = 1,
    commission: float = 0.001,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    warmup: int = 50,
) -> dict:
    """Backtest with compounding position sizing.

    Args:
        df: OHLCV + indicator DataFrame
        buy_fn: callable(current_row, prev_row) -> bool for long entry
        sell_fn: callable(current_row, prev_row) -> bool for long exit
        short_fn: callable(current_row, prev_row) -> bool for short entry
        cover_fn: callable(current_row, prev_row) -> bool for short exit
        initial_equity: starting capital
        size_pct: fraction of current equity per trade
        leverage: leverage multiplier (1-3)
        commission: fee per trade side (0.001 = 0.1%)
        stop_loss_pct: optional hard stop as fraction (0.02 = 2%)
        take_profit_pct: optional take profit as fraction (0.03 = 3%)
        warmup: bars to skip for indicator warmup
    """
    equity = initial_equity
    position = None
    trades = []
    equity_curve = []

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else row
        price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        # Track unrealized equity
        if position is not None:
            if position["side"] == "long":
                unrealized = (price - position["entry"]) / position["entry"] * position["size_usd"] * leverage
            else:
                unrealized = (position["entry"] - price) / position["entry"] * position["size_usd"] * leverage
            equity_curve.append(equity + unrealized)
        else:
            equity_curve.append(equity)

        if position is not None:
            entry = position["entry"]
            side = position["side"]
            closed = False

            # Check stop loss
            if stop_loss_pct is not None and not closed:
                if side == "long" and low <= entry * (1 - stop_loss_pct):
                    exit_price = entry * (1 - stop_loss_pct)
                    pnl = (exit_price - entry) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, exit_price, pnl - cost, equity, "stop_loss"))
                    position = None
                    closed = True
                elif side == "short" and high >= entry * (1 + stop_loss_pct):
                    exit_price = entry * (1 + stop_loss_pct)
                    pnl = (entry - exit_price) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, exit_price, pnl - cost, equity, "stop_loss"))
                    position = None
                    closed = True

            # Check take profit
            if take_profit_pct is not None and not closed:
                if side == "long" and high >= entry * (1 + take_profit_pct):
                    exit_price = entry * (1 + take_profit_pct)
                    pnl = (exit_price - entry) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, exit_price, pnl - cost, equity, "take_profit"))
                    position = None
                    closed = True
                elif side == "short" and low <= entry * (1 - take_profit_pct):
                    exit_price = entry * (1 - take_profit_pct)
                    pnl = (entry - exit_price) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, exit_price, pnl - cost, equity, "take_profit"))
                    position = None
                    closed = True

            # Check signal-based exits
            if not closed:
                if side == "long" and sell_fn and sell_fn(row, prev):
                    pnl = (price - entry) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, price, pnl - cost, equity, "signal"))
                    position = None
                elif side == "short" and cover_fn and cover_fn(row, prev):
                    pnl = (entry - price) / entry * position["size_usd"] * leverage
                    cost = position["size_usd"] * commission
                    equity += pnl - cost
                    trades.append(_trade_record(position, price, pnl - cost, equity, "signal"))
                    position = None

        # Open new positions (only if flat)
        if position is None:
            if buy_fn and buy_fn(row, prev):
                size_usd = equity * size_pct
                cost = size_usd * commission
                equity -= cost
                position = {"side": "long", "entry": price, "size_usd": size_usd, "units": size_usd / price}
            elif short_fn and short_fn(row, prev):
                size_usd = equity * size_pct
                cost = size_usd * commission
                equity -= cost
                position = {"side": "short", "entry": price, "size_usd": size_usd, "units": size_usd / price}

    # Close any open position at end
    if position is not None:
        price = float(df.iloc[-1]["close"])
        if position["side"] == "long":
            pnl = (price - position["entry"]) / position["entry"] * position["size_usd"] * leverage
        else:
            pnl = (position["entry"] - price) / position["entry"] * position["size_usd"] * leverage
        cost = position["size_usd"] * commission
        equity += pnl - cost
        trades.append(_trade_record(position, price, pnl - cost, equity, "end_of_data"))

    if not trades:
        return {
            "num_trades": 0, "win_rate": 0, "total_return_pct": 0,
            "final_equity": initial_equity, "max_drawdown_pct": 0,
            "avg_return_pct": 0, "profit_factor": 0,
            "trades": [], "equity_curve": [initial_equity],
        }

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    win_rate = len(wins) / len(tdf) * 100

    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    ec = pd.Series(equity_curve if equity_curve else [initial_equity])
    peak = ec.expanding().max()
    max_dd = ((ec - peak) / peak * 100).min()

    return {
        "num_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "total_return_pct": round((equity - initial_equity) / initial_equity * 100, 2),
        "final_equity": round(equity, 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "avg_return_pct": round(float(tdf["return_pct"].mean()), 2),
        "profit_factor": round(profit_factor, 2),
        "trades": trades,
        "equity_curve": equity_curve,
    }


def _trade_record(position: dict, exit_price: float, pnl: float, equity: float, reason: str) -> dict:
    entry = position["entry"]
    side = position["side"]
    if side == "long":
        ret = (exit_price - entry) / entry * 100
    else:
        ret = (entry - exit_price) / entry * 100
    return {
        "side": side, "entry": entry, "exit": exit_price,
        "size_usd": position["size_usd"], "pnl": round(pnl, 4),
        "return_pct": round(ret, 2), "equity_after": round(equity, 2),
        "reason": reason,
    }
