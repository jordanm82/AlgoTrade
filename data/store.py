# data/store.py
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


class DataStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _ohlcv_path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "-").replace(":", "-")
        return self.base_dir / f"ohlcv_{safe}_{timeframe}.parquet"

    def save_ohlcv(self, symbol: str, timeframe: str, df: pd.DataFrame):
        df.to_parquet(self._ohlcv_path(symbol, timeframe))

    def load_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._ohlcv_path(symbol, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _trades_path(self) -> Path:
        return self.base_dir / "trades.csv"

    def append_trade(self, trade: dict):
        path = self._trades_path()
        df = pd.DataFrame([trade])
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def load_trades(self) -> pd.DataFrame:
        path = self._trades_path()
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def save_snapshot(self, snapshot: dict):
        snapshots_dir = self.base_dir / "snapshots"
        snapshots_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = snapshots_dir / f"snapshot_{ts}.json"
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    def load_latest_snapshot(self) -> dict | None:
        snapshots_dir = self.base_dir / "snapshots"
        if not snapshots_dir.exists():
            return None
        files = sorted(snapshots_dir.glob("snapshot_*.json"))
        if not files:
            return None
        with open(files[-1]) as f:
            return json.load(f)
