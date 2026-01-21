from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# -------------------------
# Import project modules
# -------------------------
SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.insert(0, str(SRC_DIR))

from quantdemo.strategy.macd_scanner import MacdScanParams  # noqa: E402
from quantdemo.research.indicators import add_indicators  # noqa: E402


# -------------------------
# Helpers
# -------------------------
def is_business_day(d: date) -> bool:
    return d.weekday() < 5


def last_business_day(d: date) -> date:
    while not is_business_day(d):
        d = d - timedelta(days=1)
    return d


def parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def next_month(d: date) -> date:
    y, m = d.year, d.month
    if m == 12:
        return date(y + 1, 1, 1)
    return date(y, m + 1, 1)


def iter_month_starts(start: date, end: date) -> Iterable[date]:
    m = month_start(start)
    end_m = month_start(end)
    while m <= end_m:
        yield m
        m = next_month(m)


def get_repo_root() -> Path:
    # src/scripts/*.py -> parents[2] => repo root
    return Path(__file__).resolve().parents[2]


def fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def list_symbols_from_curated(curated_root: Path) -> List[str]:
    """
    curated_root example:
      data/curated/polygon/1day
        symbol=AAPL/year=2025/month=07/...
    """
    if not curated_root.exists():
        raise FileNotFoundError(f"Curated root not found: {curated_root}")

    syms: List[str] = []
    for p in curated_root.glob("symbol=*"):
        if p.is_dir():
            syms.append(p.name.split("=", 1)[1])
    return sorted(set(syms))


def find_month_file(curated_root: Path, symbol: str, y: int, m: int) -> Optional[Path]:
    """
    Each symbol each month is in one folder:
      symbol=XXX/year=YYYY/month=MM/*.parquet
    We'll pick the first parquet in it (you can standardize later).
    """
    month_dir = curated_root / f"symbol={symbol}" / f"year={y:04d}" / f"month={m:02d}"
    if not month_dir.exists():
        return None
    files = sorted(month_dir.glob("*.parquet"))
    if not files:
        return None
    return files[0]


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tolerant column normalization:
    - ensure 'date' exists
    - map Open/High/Low/Close/Volume to lower-case if needed
    """
    if "date" not in df.columns:
        for c in ["Date", "datetime", "time", "timestamp"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

    # Standardize OHLCV names
    colmap = {}
    if "Close" in df.columns and "close" not in df.columns:
        colmap["Close"] = "close"
    if "Open" in df.columns and "open" not in df.columns:
        colmap["Open"] = "open"
    if "High" in df.columns and "high" not in df.columns:
        colmap["High"] = "high"
    if "Low" in df.columns and "low" not in df.columns:
        colmap["Low"] = "low"
    if "Volume" in df.columns and "volume" not in df.columns:
        colmap["Volume"] = "volume"
    if colmap:
        df = df.rename(columns=colmap)

    # Normalize date to python date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    return df


def add_ema_columns(df: pd.DataFrame, spans: List[int], price_col: str = "close") -> pd.DataFrame:
    """
    Add EMA columns: EMA_{span}
    Assumes df index is DatetimeIndex.
    """
    if price_col not in df.columns:
        return df

    s = pd.to_numeric(df[price_col], errors="coerce")
    for span in spans:
        df[f"EMA_{span}"] = s.ewm(span=int(span), adjust=False).mean()
    return df


def load_symbol_range_from_curated(
    curated_root: Path,
    symbol: str,
    start: date,
    end: date,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Load enough data to compute indicators in [start..end].
    We read monthly files covering [start-lookback .. end], concat once.
    """
    need_start = start - timedelta(days=lookback_days + 10)  # small buffer
    need_end = end

    parts: List[pd.DataFrame] = []
    for m0 in iter_month_starts(need_start, need_end):
        f = find_month_file(curated_root, symbol, m0.year, m0.month)
        if f is None:
            continue

        df = pd.read_parquet(f)
        df = normalize_ohlcv_columns(df)

        if "date" not in df.columns:
            continue

        df = df.sort_values("date").reset_index(drop=True)
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, ignore_index=True)

    # drop duplicate dates (keep last)
    full = full.drop_duplicates(subset=["date"], keep="last")
    full = full.sort_values("date").reset_index(drop=True)

    # DatetimeIndex for indicator functions (your add_indicators likely expects df.index usage)
    full.index = pd.to_datetime(full["date"])

    # keep only needed window (including lookback buffer)
    full = full[(full["date"] >= need_start) & (full["date"] <= need_end)]
    return full


# -------------------------
# Core build
# -------------------------
def build_macd_daily_table_from_curated(
    curated_root: Path,
    out_file: Path,
    start_date: date,
    end_date: date,
    params: MacdScanParams,
    ema_spans: List[int],
    symbols: Optional[List[str]] = None,
    show_progress: bool = True,
    print_every: int = 50,
) -> None:
    """
    Build a long-table parquet:
      date, symbol, close,
      MACD, SIGNAL, HIST, macd_diff, golden_cross, dead_cross,
      EMA_{span}...

    using curated monthly files.
    """
    if not is_business_day(end_date):
        end_date = last_business_day(end_date)

    if symbols is None:
        symbols = list_symbols_from_curated(curated_root)
    if not symbols:
        raise ValueError(f"No symbols found under curated root: {curated_root}")

    try:
        from tqdm import tqdm  # type: ignore

        iterator = tqdm(symbols, desc="Symbols", unit="sym") if show_progress else symbols
    except Exception:
        iterator = symbols

    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    t0 = time.time()
    processed = 0
    ok = 0
    empty = 0
    err = 0

    print(f"[INFO] Curated root: {curated_root.resolve()}")
    print(f"[INFO] Output file : {out_file.resolve()}")
    print(f"[INFO] Date range  : {start_date} -> {end_date}")
    print(f"[INFO] Symbols     : {len(symbols):,}")
    print(f"[INFO] Lookback    : {params.lookback_days} days")
    print(f"[INFO] EMA spans   : {ema_spans}")
    print()

    for sym in iterator:
        processed += 1
        try:
            df = load_symbol_range_from_curated(
                curated_root=curated_root,
                symbol=sym,
                start=start_date,
                end=end_date,
                lookback_days=params.lookback_days,
            )

            if df.empty or "close" not in df.columns:
                empty += 1
                continue

            # Your existing indicators (MACD + MA + VOL_MA etc)
            df = add_indicators(
                df,
                ma_fast=params.ma_fast,
                ma_slow=params.ma_slow,
                vol_ma=params.vol_ma,
                macd_fast=params.macd_fast,
                macd_slow=params.macd_slow,
                macd_signal=params.macd_signal,
            )

            # ✅ Add EMA columns directly into df
            df = add_ema_columns(df, spans=ema_spans, price_col="close")

            # Ensure required indicator cols exist
            req_cols = ["date", "close", "MACD", "SIGNAL", "HIST"]
            if any(c not in df.columns for c in req_cols):
                empty += 1
                continue

            # Crosses calculated on full df (so shift uses real previous rows)
            macd = pd.to_numeric(df["MACD"], errors="coerce")
            sig = pd.to_numeric(df["SIGNAL"], errors="coerce")
            golden = (macd > sig) & (macd.shift(1) <= sig.shift(1))
            dead = (macd < sig) & (macd.shift(1) >= sig.shift(1))
            macd_diff = macd - sig

            # Slice to final range
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            df2 = df.loc[mask].copy()
            if df2.empty:
                empty += 1
                continue

            # Vectorized attach for the slice
            df2["macd_diff"] = macd_diff.loc[df2.index].astype(float)
            df2["golden_cross"] = golden.loc[df2.index].fillna(False).astype(bool)
            df2["dead_cross"] = dead.loc[df2.index].fillna(False).astype(bool)

            # Build output rows (vectorized)
            out_cols = ["date", "close", "MACD", "SIGNAL", "HIST", "macd_diff", "golden_cross", "dead_cross"]
            for span in ema_spans:
                c = f"EMA_{span}"
                if c in df2.columns:
                    out_cols.append(c)

            sub = df2[out_cols].copy()

            # Normalize types
            sub["date"] = sub["date"].apply(lambda x: x.isoformat() if isinstance(x, date) else str(x))
            sub["symbol"] = sym
            sub["close"] = pd.to_numeric(sub["close"], errors="coerce")

            for c in ["MACD", "SIGNAL", "HIST", "macd_diff"]:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")

            for span in ema_spans:
                c = f"EMA_{span}"
                if c in sub.columns:
                    sub[c] = pd.to_numeric(sub[c], errors="coerce")

            # Move symbol to front-ish
            # (最终顺序你可以随意调整)
            front = ["date", "symbol", "close", "MACD", "SIGNAL", "HIST", "macd_diff", "golden_cross", "dead_cross"]
            ema_cols = [f"EMA_{span}" for span in ema_spans if f"EMA_{span}" in sub.columns]
            sub = sub[front + ema_cols]

            all_rows.extend(sub.to_dict(orient="records"))
            ok += 1

        except Exception:
            err += 1
            continue

        # periodic prints if no tqdm
        if (not show_progress) and (processed % print_every == 0):
            elapsed = time.time() - t0
            print(
                f"[PROG] {processed:,}/{len(symbols):,} ok={ok:,} empty={empty:,} err={err:,} elapsed={fmt_secs(elapsed)}",
                flush=True,
            )

    if not all_rows:
        elapsed = time.time() - t0
        print(f"[WARN] No rows generated. elapsed={fmt_secs(elapsed)}")
        return

    out_df = pd.DataFrame(all_rows)

    # Sort for stable output
    out_df = out_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Save
    out_df.to_parquet(out_file, index=False)

    elapsed = time.time() - t0
    print()
    print(f"[OK] Saved rows: {len(out_df):,}")
    print(f"[OK] Symbols ok : {ok:,} | empty={empty:,} | err={err:,}")
    print(f"[OK] Output     : {out_file.resolve()}")
    print(f"[OK] Elapsed    : {fmt_secs(elapsed)}")


# -------------------------
# CLI
# -------------------------
def main():
    repo_root = get_repo_root()

    parser = argparse.ArgumentParser(description="Build daily MACD+EMA table from curated monthly files.")
    parser.add_argument("--timeframe", default="1day", help="timeframe folder name (default: 1day)")
    parser.add_argument("--start", default="2025-07-01", help="start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-16", help="end date (YYYY-MM-DD)")
    parser.add_argument("--lookback", type=int, default=120, help="lookback days for MACD (default: 120)")

    # ✅ EMA
    parser.add_argument(
        "--ema-spans",
        default="12,26",
        help="EMA spans comma-separated (default: 12,26). Example: 9,21,50",
    )

    parser.add_argument("--no-progress", action="store_true", help="disable tqdm")
    args = parser.parse_args()

    curated_root = repo_root / "data" / "curated" / "polygon" / args.timeframe
    out_file = repo_root / "data" / "features" / "polygon" / args.timeframe / "indicators_macd_daily.parquet"

    start_date = parse_iso_date(args.start)
    end_date = parse_iso_date(args.end)

    params = MacdScanParams(lookback_days=int(args.lookback))

    # parse ema spans
    ema_spans: List[int] = []
    for x in str(args.ema_spans).split(","):
        x = x.strip()
        if x:
            ema_spans.append(int(x))
    if not ema_spans:
        ema_spans = [12, 26]

    build_macd_daily_table_from_curated(
        curated_root=curated_root,
        out_file=out_file,
        start_date=start_date,
        end_date=end_date,
        params=params,
        ema_spans=ema_spans,
        show_progress=(not args.no_progress),
    )


if __name__ == "__main__":
    main()
