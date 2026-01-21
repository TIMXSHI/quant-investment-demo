from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class DailyBarsPartitionLayout:
    """
    Your current parquet layout:
      {data_dir}/symbol={SYMBOL}/year=YYYY/month=MM/day=DD/bars.parquet
    """
    filename: str = "bars.parquet"


def _iter_dates(start: datetime.date, end: datetime.date) -> Iterable[datetime.date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _expected_partition_file(
    sym_dir: Path,
    d: datetime.date,
    layout: DailyBarsPartitionLayout,
) -> Path:
    return (
        sym_dir
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}"
        / layout.filename
    )


def load_symbol_daily_range(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: Path | str = Path("data/raw/polygon/1day"),
    layout: DailyBarsPartitionLayout = DailyBarsPartitionLayout(),
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load daily bars for a symbol between [start_date, end_date] (inclusive)
    by reading only existing day partitions.

    Args:
        symbol: e.g. "AAPL"
        start_date/end_date: "YYYY-MM-DD"
        data_dir: root path containing symbol=... partitions
        layout: partition file layout
        strict: if True, raise FileNotFoundError when nothing is found

    Returns:
        DataFrame indexed by datetime (date), sorted ascending.
        Expected columns include: open, high, low, close, volume, vwap, transactions (if present).
    """
    data_dir = Path(data_dir)
    sym_dir = data_dir / f"symbol={symbol}"

    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    files: List[Path] = []
    for d in _iter_dates(start, end):
        f = _expected_partition_file(sym_dir, d, layout)
        if f.exists():
            files.append(f)

    if not files:
        if strict:
            raise FileNotFoundError(f"No {layout.filename} found for {symbol} in {start_date} ~ {end_date}")
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # normalize date/index
    if "date" not in df.columns:
        raise ValueError("Input parquet must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )

    # numeric safety
    for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # slice range (inclusive)
    df = df.loc[start_date:end_date].copy()
    return df
