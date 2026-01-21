from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple, List

import pandas as pd

from quantdemo.research.features import load_symbol_daily_range
from quantdemo.research.indicators import add_indicators


@dataclass(frozen=True)
class MacdScanParams:
    ma_fast: int = 20
    ma_slow: int = 50
    vol_ma: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    lookback_days: int = 120  # 只读最近N天，加速扫描


def list_symbols_from_raw(data_dir: Path) -> List[str]:
    """
    data_dir example: data/raw/polygon/1day
    expects children like: symbol=AAPL, symbol=TSLA ...
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    symbols = []
    for p in data_dir.glob("symbol=*"):
        if p.is_dir():
            symbols.append(p.name.split("symbol=")[-1])
    symbols.sort()
    return symbols


def scan_macd_crosses_on_date(
    target_date: date,
    data_dir: Path,
    params: MacdScanParams = MacdScanParams(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (golden_cross_df, dead_cross_df)
    Each df columns:
      symbol, date, close, MACD, SIGNAL, HIST
    """
    symbols = list_symbols_from_raw(data_dir)

    start_date = (target_date - timedelta(days=params.lookback_days)).isoformat()
    end_date = target_date.isoformat()

    golden_rows = []
    dead_rows = []

    for sym in symbols:
        try:
            df = load_symbol_daily_range(sym, start_date, end_date, data_dir=data_dir, strict=False)
            if df is None or df.empty:
                continue

            df = add_indicators(
                df,
                ma_fast=params.ma_fast,
                ma_slow=params.ma_slow,
                vol_ma=params.vol_ma,
                macd_fast=params.macd_fast,
                macd_slow=params.macd_slow,
                macd_signal=params.macd_signal,
            )

            # 必须确保 target_date 这天存在（交易日才有）
            if pd.Timestamp(end_date) not in df.index:
                continue

            # 取 target_date 和前一个交易日（df 内的上一行）
            loc = df.index.get_loc(pd.Timestamp(end_date))
            if isinstance(loc, slice) or loc == 0:
                continue

            today = df.iloc[loc]
            prev = df.iloc[loc - 1]

            macd_t, sig_t = float(today["MACD"]), float(today["SIGNAL"])
            macd_p, sig_p = float(prev["MACD"]), float(prev["SIGNAL"])

            row = {
                "symbol": sym,
                "date": end_date,
                "close": float(today["close"]),
                "MACD": macd_t,
                "SIGNAL": sig_t,
                "HIST": float(today["HIST"]),
            }

            # 金叉
            if (macd_t > sig_t) and (macd_p <= sig_p):
                golden_rows.append(row)

            # 死叉
            if (macd_t < sig_t) and (macd_p >= sig_p):
                dead_rows.append(row)

        except Exception:
            # 扫描时遇到坏数据/缺列/损坏 parquet：跳过该 symbol
            continue

    golden_df = pd.DataFrame(golden_rows).sort_values(["HIST"], ascending=False) if golden_rows else pd.DataFrame(
        columns=["symbol", "date", "close", "MACD", "SIGNAL", "HIST"]
    )
    dead_df = pd.DataFrame(dead_rows).sort_values(["HIST"], ascending=True) if dead_rows else pd.DataFrame(
        columns=["symbol", "date", "close", "MACD", "SIGNAL", "HIST"]
    )

    return golden_df, dead_df
