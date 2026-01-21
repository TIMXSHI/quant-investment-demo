from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


# -------------------------
# Helpers
# -------------------------
def try_get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def add_months(d: date, n: int) -> date:
    # safe month add
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)


def iter_month_starts(start_ym: date, end_ym: date):
    cur = month_start(start_ym)
    end = month_start(end_ym)
    while cur <= end:
        yield cur
        cur = add_months(cur, 1)


def list_symbols(raw_dir: Path) -> list[str]:
    syms = []
    for p in raw_dir.glob("symbol=*"):
        if p.is_dir():
            syms.append(p.name.split("symbol=")[-1])
    syms.sort()
    return syms


def raw_day_file(raw_dir: Path, symbol: str, d: date) -> Path:
    return (
        raw_dir
        / f"symbol={symbol}"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}"
        / "bars.parquet"
    )


def curated_month_file(curated_dir: Path, symbol: str, ym: date) -> Path:
    return (
        curated_dir
        / f"symbol={symbol}"
        / f"year={ym.year:04d}"
        / f"month={ym.month:02d}"
        / "bars.parquet"
    )


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "date" not in df.columns:
        raise ValueError("bars parquet must contain column 'date'")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def days_in_month(ym: date) -> list[date]:
    # ym is month start
    next_m = add_months(ym, 1)
    d = ym
    days = []
    while d < next_m:
        days.append(d)
        d += timedelta(days=1)
    return days


@dataclass
class Stats:
    months_done: int = 0
    files_read: int = 0
    files_missing: int = 0
    symbols_written: int = 0
    rows_written: int = 0


# -------------------------
# Core
# -------------------------
def curate_symbol_month(
    raw_dir: Path,
    curated_dir: Path,
    symbol: str,
    ym: date,
    skip_existing: bool,
) -> tuple[bool, int, int, int]:
    """
    Returns:
      (written?, rows_written, files_read, files_missing)
    """
    out_path = curated_month_file(curated_dir, symbol, ym)
    if skip_existing and out_path.exists():
        return False, 0, 0, 0

    frames = []
    read_cnt = 0
    miss_cnt = 0

    for d in days_in_month(ym):
        f = raw_day_file(raw_dir, symbol, d)
        if not f.exists():
            miss_cnt += 1
            continue
        try:
            df = pd.read_parquet(f)
            if df is not None and not df.empty:
                frames.append(df)
            read_cnt += 1
        except Exception:
            # treat unreadable as missing
            miss_cnt += 1

    if not frames:
        return False, 0, read_cnt, miss_cnt

    month_df = normalize_bars(pd.concat(frames, ignore_index=True))
    if month_df.empty:
        return False, 0, read_cnt, miss_cnt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    month_df.to_parquet(out_path, index=False)
    return True, len(month_df), read_cnt, miss_cnt


def run_backfill(
    raw_dir: Path,
    curated_dir: Path,
    start_month: date,
    end_month: date,
    skip_existing: bool,
    print_every: int,
):
    tqdm = try_get_tqdm()
    symbols = list_symbols(raw_dir)
    if not symbols:
        raise ValueError(f"No symbols found under: {raw_dir}")

    months = list(iter_month_starts(start_month, end_month))
    if not months:
        print("[SKIP] No months to process.")
        return

    print(f"[INFO] raw_dir     : {raw_dir.resolve()}")
    print(f"[INFO] curated_dir : {curated_dir.resolve()}")
    print(f"[INFO] symbols     : {len(symbols):,}")
    print(f"[INFO] months      : {months[0].strftime('%Y-%m')} .. {months[-1].strftime('%Y-%m')} ({len(months)} months)")
    print(f"[INFO] skip_existing: {skip_existing}")

    stats = Stats()
    t_all = time.time()

    month_iter = months if tqdm is None else tqdm(months, desc="Months", unit="mo")
    for ym in month_iter:
        t_m = time.time()
        ym_label = ym.strftime("%Y-%m")
        print(f"\n[MONTH] {ym_label} start")

        sym_iter = symbols if tqdm is None else tqdm(symbols, desc=f"Symbols {ym_label}", unit="sym", leave=False)
        processed = 0
        written_this_month = 0
        rows_this_month = 0

        for sym in sym_iter:
            processed += 1
            written, rows, read_cnt, miss_cnt = curate_symbol_month(
                raw_dir=raw_dir,
                curated_dir=curated_dir,
                symbol=sym,
                ym=ym,
                skip_existing=skip_existing,
            )

            stats.files_read += read_cnt
            stats.files_missing += miss_cnt

            if written:
                written_this_month += 1
                rows_this_month += rows
                stats.symbols_written += 1
                stats.rows_written += rows

            if tqdm is None and processed % print_every == 0:
                elapsed = time.time() - t_m
                speed = processed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[MONTH {ym_label}] {processed:,}/{len(symbols):,} "
                    f"speed={speed:.2f} sym/s written={written_this_month:,} rows={rows_this_month:,}",
                    flush=True,
                )

        stats.months_done += 1
        elapsed_m = time.time() - t_m
        print(f"[MONTH DONE] {ym_label} written_symbols={written_this_month:,} rows={rows_this_month:,} elapsed={elapsed_m/60:.1f} min")

    elapsed_all = time.time() - t_all
    print("\n[OK] Backfill complete")
    print(f"  months_done     : {stats.months_done:,}")
    print(f"  files_read      : {stats.files_read:,}")
    print(f"  files_missing   : {stats.files_missing:,}")
    print(f"  symbols_written : {stats.symbols_written:,}")
    print(f"  rows_written    : {stats.rows_written:,}")
    print(f"  elapsed_total   : {elapsed_all/60:.1f} min")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Initial backfill: raw daily -> curated monthly per symbol.")
    parser.add_argument("--timeframe", default="1day")
    parser.add_argument("--start-month", default=None, help="YYYY-MM (month start). default: 12 months ago (month start)")
    parser.add_argument("--end-month", default=None, help="YYYY-MM (month start). default: last month (month start)")
    parser.add_argument("--skip-existing", action="store_true", help="skip if curated month file already exists")
    parser.add_argument("--print-every", type=int, default=200, help="if tqdm not installed, print every N symbols")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw" / "polygon" / args.timeframe
    curated_dir = repo_root / "data" / "curated" / "polygon" / args.timeframe

    today = date.today()
    default_end = month_start(add_months(today, -1))     # last month start
    default_start = month_start(add_months(default_end, -11))  # 12 months window

    if args.start_month is None:
        start_m = default_start
    else:
        start_m = month_start(datetime.strptime(args.start_month, "%Y-%m").date())

    if args.end_month is None:
        end_m = default_end
    else:
        end_m = month_start(datetime.strptime(args.end_month, "%Y-%m").date())

    if start_m > end_m:
        raise ValueError("start-month must be <= end-month")

    run_backfill(
        raw_dir=raw_dir,
        curated_dir=curated_dir,
        start_month=start_m,
        end_month=end_m,
        skip_existing=bool(args.skip_existing),
        print_every=int(args.print_every),
    )


if __name__ == "__main__":
    main()