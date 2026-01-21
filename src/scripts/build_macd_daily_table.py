from __future__ import annotations

# =========================
# Bootstrap (NO PYTHONPATH needed)
# MUST be before importing quantdemo.*
# =========================
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.insert(0, str(SRC_DIR))

# =========================
# Standard imports
# =========================
import argparse
import time
from datetime import date, datetime, timedelta

import pandas as pd

from quantdemo.strategy.macd_scanner import list_symbols_from_raw, MacdScanParams
from quantdemo.research.features import load_symbol_daily_range
from quantdemo.research.indicators import add_indicators


# =========================
# Helpers
# =========================
def is_business_day(d: date) -> bool:
    return d.weekday() < 5


def next_business_day(d: date) -> date:
    d = d + timedelta(days=1)
    while not is_business_day(d):
        d = d + timedelta(days=1)
    return d


def last_business_day(d: date) -> date:
    while not is_business_day(d):
        d = d - timedelta(days=1)
    return d


def parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def try_get_tqdm():
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm
    except Exception:
        return None


def get_repo_root() -> Path:
    # src/scripts/*.py -> parents[2] => repo root
    return Path(__file__).resolve().parents[2]


def read_existing_max_date(out_file: Path) -> date | None:
    if not out_file.exists():
        return None
    try:
        df = pd.read_parquet(out_file, columns=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().date()
    except Exception:
        # If parquet is corrupted, treat as none (or you can raise)
        return None


# =========================
# Core builder
# =========================
def build_macd_daily_table_incremental(
    raw_dir: Path,
    out_file: Path,
    initial_start_date: date,
    end_date: date,
    params: MacdScanParams,
    show_progress: bool = True,
    print_every: int = 100,
) -> None:
    """
    Incrementally compute daily MACD metrics for all symbols and append to a long-table parquet:
      date, symbol, close, MACD, SIGNAL, HIST, golden_cross, dead_cross

    It will:
      - if out_file exists: start = max(date)+1 business day
      - else: start = initial_start_date
      - process business days only (weekend rollback/skip)
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {raw_dir}")

    symbols = list_symbols_from_raw(raw_dir)
    if not symbols:
        raise ValueError(f"No symbols found under: {raw_dir} (expected folders like symbol=AAPL)")

    last_done = read_existing_max_date(out_file)
    if last_done is None:
        start_date = initial_start_date
        print(f"[INFO] Output not found. Initial build starting from {start_date.isoformat()}")
    else:
        start_date = next_business_day(last_done)
        print(f"[INFO] Found existing output. Last done date = {last_done.isoformat()}, next start = {start_date.isoformat()}")

    # Normalize to business days
    if not is_business_day(start_date):
        start_date = last_business_day(start_date)
    if not is_business_day(end_date):
        end_date = last_business_day(end_date)

    if start_date > end_date:
        print(f"[SKIP] No new trading days to process. start={start_date.isoformat()} end={end_date.isoformat()}")
        return

    tqdm = try_get_tqdm() if show_progress else None

    all_new_rows: list[dict] = []
    total_days = 0
    d = start_date
    while d <= end_date:
        if is_business_day(d):
            total_days += 1
        d = d + timedelta(days=1)

    print(f"[INFO] Symbols={len(symbols):,} | Days to process={total_days:,} | Lookback={params.lookback_days}d")
    print(f"[INFO] Writing to: {out_file.resolve()}")

    day_counter = 0
    d = start_date
    overall_t0 = time.time()

    while d <= end_date:
        if not is_business_day(d):
            d = d + timedelta(days=1)
            continue

        day_counter += 1
        iso_d = d.isoformat()
        print(f"\n[DAY {day_counter}/{total_days}] Processing {iso_d}")

        # We only need enough history to compute MACD up to target day
        start_iso = (d - timedelta(days=params.lookback_days)).isoformat()

        rows_today: list[dict] = []
        t0 = time.time()

        # progress wrapper
        iterator = symbols
        if tqdm is not None:
            iterator = tqdm(symbols, desc=f"Symbols {iso_d}", unit="sym")

        processed = 0
        kept = 0
        errors = 0

        for sym in iterator:
            processed += 1
            try:
                df = load_symbol_daily_range(
                    symbol=sym,
                    start_date=start_iso,
                    end_date=iso_d,
                    data_dir=raw_dir,
                    strict=False,
                )
                if df is None or df.empty:
                    continue
                # ensure target day exists (trading day)
                if pd.Timestamp(iso_d) not in df.index:
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

                loc = df.index.get_loc(pd.Timestamp(iso_d))
                if isinstance(loc, slice) or loc == 0:
                    continue

                today = df.iloc[loc]
                prev = df.iloc[loc - 1]

                macd_t, sig_t = float(today["MACD"]), float(today["SIGNAL"])
                macd_p, sig_p = float(prev["MACD"]), float(prev["SIGNAL"])

                golden = (macd_t > sig_t) and (macd_p <= sig_p)
                dead = (macd_t < sig_t) and (macd_p >= sig_p)

                rows_today.append(
                    {
                        "date": iso_d,
                        "symbol": sym,
                        "close": float(today["close"]),
                        "MACD": macd_t,
                        "SIGNAL": sig_t,
                        "HIST": float(today["HIST"]),
                        "golden_cross": bool(golden),
                        "dead_cross": bool(dead),
                    }
                )
                kept += 1

            except Exception:
                errors += 1
                continue

            # if no tqdm, print periodic progress
            if tqdm is None and (processed % print_every == 0):
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed > 0 else 0.0
                # rough eta for the current day
                remain = len(symbols) - processed
                eta = remain / speed if speed > 0 else 0.0
                print(
                    f"[{iso_d}] {processed:,}/{len(symbols):,} "
                    f"kept={kept:,} err={errors:,} speed={speed:.2f} sym/s eta={fmt_secs(eta)}",
                    flush=True,
                )

        elapsed_day = time.time() - t0
        speed_day = processed / elapsed_day if elapsed_day > 0 else 0.0
        print(
            f"[DAY DONE] {iso_d} processed={processed:,} kept={kept:,} err={errors:,} "
            f"elapsed={fmt_secs(elapsed_day)} speed={speed_day:.2f} sym/s"
        )

        if rows_today:
            all_new_rows.extend(rows_today)

        d = d + timedelta(days=1)

    if not all_new_rows:
        elapsed_total = time.time() - overall_t0
        print(f"\n[INFO] No rows generated. Total elapsed={fmt_secs(elapsed_total)}")
        return

    new_df = pd.DataFrame(all_new_rows)

    # Append/update (dedupe on date+symbol)
    if out_file.exists():
        old_df = pd.read_parquet(out_file)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df["date"] = final_df["date"].astype(str)
        final_df = final_df.drop_duplicates(subset=["date", "symbol"], keep="last")
    else:
        final_df = new_df.copy()

    final_df = final_df.sort_values(["date", "symbol"]).reset_index(drop=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_file, index=False)

    elapsed_total = time.time() - overall_t0
    print(f"\n[OK] Added rows: {len(new_df):,} | Total rows: {len(final_df):,} | Total elapsed: {fmt_secs(elapsed_total)}")
    print(f"[OK] Saved to: {out_file.resolve()}")


# =========================
# CLI
# =========================
def main():
    repo_root = get_repo_root()

    parser = argparse.ArgumentParser(description="Build/Update daily MACD indicator table (incremental).")
    parser.add_argument("--timeframe", default="1day", help="raw timeframe folder under data/raw/polygon (default: 1day)")
    parser.add_argument("--start", default="2026-01-16", help="initial start date if output table doesn't exist (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="end date (YYYY-MM-DD). default: last business day of yesterday")
    parser.add_argument("--lookback", type=int, default=120, help="lookback days per symbol (default: 120)")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bar")
    parser.add_argument("--print-every", type=int, default=100, help="when tqdm not available, print every N symbols")
    args = parser.parse_args()

    raw_dir = repo_root / "data" / "raw" / "polygon" / args.timeframe
    out_file = repo_root / "data" / "features" / "polygon" / args.timeframe / "indicators_macd_daily.parquet"

    initial_start_date = parse_iso_date(args.start)

    if args.end is None:
        end_date = last_business_day(date.today() - timedelta(days=1))
    else:
        end_date = parse_iso_date(args.end)

    params = MacdScanParams(lookback_days=int(args.lookback))

    build_macd_daily_table_incremental(
        raw_dir=raw_dir,
        out_file=out_file,
        initial_start_date=initial_start_date,
        end_date=end_date,
        params=params,
        show_progress=(not args.no_progress),
        print_every=int(args.print_every),
    )


if __name__ == "__main__":
    main()
