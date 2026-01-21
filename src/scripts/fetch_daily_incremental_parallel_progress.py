from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

# -------------------------
# Config
# -------------------------
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment (.env).")

BASE_URL = "https://api.polygon.io"
UNIVERSE_PATH = Path("config/universe_daily.yml")

DATA_ROOT = Path("data/raw/polygon/1day")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_SLEEP_SEC = 0.15
MAX_RETRIES = 6


# -------------------------
# Helpers
# -------------------------
def last_business_day_simple(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return s


def _get_json(session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(params or {})
    params["apiKey"] = API_KEY

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)

            if r.status_code in (429, 500, 502, 503, 504):
                backoff = (2 ** (attempt - 1)) * 0.8 + random.random() * 0.3
                time.sleep(backoff)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            backoff = (2 ** (attempt - 1)) * 0.8 + random.random() * 0.3
            time.sleep(backoff)

    raise RuntimeError(f"GET failed after {MAX_RETRIES} retries: {url}") from last_err


def load_universe_symbols(path: Path) -> List[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or not symbols:
        raise ValueError(f"Universe file has no symbols list: {path}")
    out = []
    for s in symbols:
        s = str(s).strip().upper()
        if s:
            out.append(s)
    return sorted(set(out))


def shard_symbols(symbols: List[str], shard_index: int, shard_count: int) -> List[str]:
    if shard_count <= 1:
        return symbols
    return [s for i, s in enumerate(symbols) if (i % shard_count) == shard_index]


def _out_dir_for(symbol: str, d: date) -> Path:
    return (
        DATA_ROOT
        / f"symbol={symbol}"
        / f"year={d.year:04d}"
        / f"month={d.month:02d}"
        / f"day={d.day:02d}"
    )


def _out_file_for(symbol: str, d: date) -> Path:
    return _out_dir_for(symbol, d) / "bars.parquet"


def _done_marker(symbol: str) -> Path:
    return DATA_ROOT / f"symbol={symbol}" / "_DONE.marker"


def read_marker_last_date(symbol: str) -> Optional[date]:
    p = _done_marker(symbol)
    if not p.exists():
        return None
    try:
        meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        last_date = meta.get("last_date")
        if last_date:
            return datetime.fromisoformat(str(last_date)).date()
        window = meta.get("window") or {}
        end_s = window.get("end")
        if end_s:
            return datetime.fromisoformat(str(end_s)).date()
    except Exception:
        return None
    return None


def scan_fs_last_date(symbol: str) -> Optional[date]:
    sym_root = DATA_ROOT / f"symbol={symbol}"
    if not sym_root.exists():
        return None

    max_d: Optional[date] = None
    for p in sym_root.rglob("bars.parquet"):
        try:
            day_part = p.parent.name
            month_part = p.parent.parent.name
            year_part = p.parent.parent.parent.name
            dd = int(day_part.split("=")[1])
            mm = int(month_part.split("=")[1])
            yy = int(year_part.split("=")[1])
            d = date(yy, mm, dd)
            if (max_d is None) or (d > max_d):
                max_d = d
        except Exception:
            continue
    return max_d


def get_last_local_date(symbol: str) -> Optional[date]:
    d = read_marker_last_date(symbol)
    if d is not None:
        return d
    return scan_fs_last_date(symbol)


def write_marker(symbol: str, meta: Dict[str, Any]) -> None:
    p = _done_marker(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


# -------------------------
# Fetch + Save
# -------------------------
def fetch_daily_bars(session: requests.Session, ticker: str, start: date, end: date) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    data = _get_json(session, url, params=params)

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
    df["symbol"] = ticker
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "transactions",
        }
    )
    keep = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"]
    df = df[keep].sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def fetch_daily_bars_with_fallback(session: requests.Session, ticker: str, start: date, end: date) -> Tuple[pd.DataFrame, str]:
    df = fetch_daily_bars(session, ticker, start, end)
    if not df.empty:
        return df, ticker

    if "." in ticker:
        alt = ticker.replace(".", "-")
        df2 = fetch_daily_bars(session, alt, start, end)
        if not df2.empty:
            df2["symbol"] = ticker
            return df2, alt

    return df, ticker


def save_partitioned_by_day(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    written = 0
    for (symbol, d), g in df.groupby(["symbol", "date"], sort=False):
        out_file = _out_file_for(symbol, d)
        if out_file.exists():
            continue
        out_file.parent.mkdir(parents=True, exist_ok=True)

        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])
        g.to_parquet(out_file, index=False)
        written += 1

    return written


# -------------------------
# Worker: logging helpers
# -------------------------
def _wlog(log_path: Path, msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _append_failed_worker(failed_path: Path, symbol: str, error: str) -> None:
    new_file = not failed_path.exists()
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts_utc", "symbol", "error"])
        w.writerow([pd.Timestamp.utcnow().isoformat(), symbol, error])


# -------------------------
# Worker logic (runs in a separate process)
# -------------------------
def run_shard(
    worker_id: int,
    shard_index: int,
    shard_count: int,
    end_date: date,
    only_failed: bool,
    max_symbols: Optional[int],
    lookback_if_missing: int,
    print_every: int,
) -> Dict[str, Any]:
    """
    Process a shard of symbols. Return summary + shard preview for main process.
    """
    session = _session()

    log_path = LOG_DIR / f"fetch_worker_{worker_id}.log"
    failed_path = LOG_DIR / f"daily_fetch_failed.worker{worker_id}.csv"

    universe = load_universe_symbols(UNIVERSE_PATH)
    if only_failed:
        # main process will also pass only_failed; we still filter here
        # (each worker reads same CSV list; OK)
        base_failed = set()
        # if previous merged file exists, use it; else ignore
        merged = LOG_DIR / "daily_fetch_failed.csv"
        if merged.exists():
            try:
                df = pd.read_csv(merged)
                base_failed = set(df["symbol"].astype(str).str.upper().str.strip().tolist())
            except Exception:
                base_failed = set()
        universe = [s for s in universe if s in base_failed]

    shard = shard_symbols(universe, shard_index, shard_count)

    if max_symbols is not None:
        shard = shard[: max_symbols]

    # shard summary
    preview = {
        "count": len(shard),
        "head": shard[:5],
        "tail": shard[-5:] if len(shard) > 5 else shard,
    }
    _wlog(log_path, f"[W{worker_id}] shard_index={shard_index}/{shard_count} symbols={preview['count']} head={preview['head']} tail={preview['tail']}")

    written_partitions = 0
    up_to_date = 0
    no_data = 0
    failed_cnt = 0

    t0 = time.time()

    for idx, sym in enumerate(shard, start=1):
        prefix = f"[W{worker_id} {idx:,}/{len(shard):,}] {sym}"
        try:
            last_local = get_last_local_date(sym)

            if last_local is None:
                start_date = end_date - timedelta(days=lookback_if_missing)
                mode = "bootstrap"
            else:
                start_date = last_local + timedelta(days=1)
                mode = "incremental"

            if start_date > end_date:
                up_to_date += 1
                if (idx % print_every) == 0 or idx == 1:
                    _wlog(log_path, f"{prefix}: SKIP up-to-date last_local={last_local}")
                continue

            df, used_ticker = fetch_daily_bars_with_fallback(session, sym, start_date, end_date)

            if df.empty:
                no_data += 1
                write_marker(
                    sym,
                    {
                        "status": "no_data",
                        "symbol": sym,
                        "used_ticker": used_ticker,
                        "mode": mode,
                        "window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                        "last_date": last_local.isoformat() if last_local else None,
                        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
                    },
                )
                _wlog(log_path, f"{prefix}: no new bars ({start_date} -> {end_date}) mode={mode} used={used_ticker}")

            else:
                written = save_partitioned_by_day(df)
                written_partitions += written
                new_last = max(df["date"].tolist()) if "date" in df.columns else last_local

                write_marker(
                    sym,
                    {
                        "status": "ok",
                        "symbol": sym,
                        "used_ticker": used_ticker,
                        "mode": mode,
                        "rows": int(len(df)),
                        "partitions_written": int(written),
                        "window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                        "last_date": str(new_last),
                        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
                    },
                )

                _wlog(
                    log_path,
                    f"{prefix}: OK mode={mode} used={used_ticker} window={start_date}->{end_date} rows={len(df)} written={written} last_date={new_last}",
                )

        except Exception as e:
            failed_cnt += 1
            _append_failed_worker(failed_path, sym, repr(e))
            _wlog(log_path, f"{prefix}: ERROR -> {e}")

        time.sleep(BASE_SLEEP_SEC)

    elapsed = time.time() - t0
    _wlog(log_path, f"[W{worker_id}] DONE symbols={len(shard)} written_partitions={written_partitions} up_to_date={up_to_date} no_data={no_data} failed={failed_cnt} elapsed={elapsed:.1f}s")

    return {
        "worker_id": worker_id,
        "preview": preview,
        "written_partitions": written_partitions,
        "up_to_date": up_to_date,
        "no_data": no_data,
        "failed": failed_cnt,
        "symbols": len(shard),
        "failed_path": str(failed_path),
    }


# -------------------------
# CLI
# -------------------------
@dataclass
class Args:
    only_failed: bool
    max_symbols: Optional[int]
    lookback_if_missing: int
    end: Optional[str]
    workers: int
    print_every: int


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--only-failed", action="store_true", help="Only fetch symbols listed in logs/daily_fetch_failed.csv")
    p.add_argument("--max-symbols", type=int, default=None, help="Optional cap for safety (applies per worker shard)")
    p.add_argument("--lookback-if-missing", type=int, default=365, help="If symbol has no local data, fetch past N days")
    p.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD). Default: last business day of yesterday")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers (e.g., 1/2/4)")
    p.add_argument("--print-every", type=int, default=1, help="Print SKIP lines every N symbols (default 1 = always)")
    a = p.parse_args()
    return Args(
        only_failed=a.only_failed,
        max_symbols=a.max_symbols,
        lookback_if_missing=int(a.lookback_if_missing),
        end=a.end,
        workers=int(a.workers),
        print_every=int(a.print_every),
    )


def merge_failed_csvs(worker_failed_paths: List[str], merged_path: Path) -> None:
    rows = []
    for p in worker_failed_paths:
        pp = Path(p)
        if not pp.exists():
            continue
        try:
            df = pd.read_csv(pp)
            if df.empty:
                continue
            rows.append(df)
        except Exception:
            continue

    if not rows:
        return

    merged = pd.concat(rows, ignore_index=True)
    merged.to_csv(merged_path, index=False)


def main() -> None:
    args = parse_args()

    if args.end:
        end_date = datetime.fromisoformat(args.end).date()
    else:
        end_date = last_business_day_simple(date.today() - timedelta(days=1))

    workers = max(1, int(args.workers))
    shard_count = workers

    print(f"End date target: {end_date}")
    print(f"Workers: {workers}")
    print(f"Output root: {DATA_ROOT.resolve()}")
    print(f"Worker logs: {LOG_DIR.resolve()}\\fetch_worker_*.log")
    print()

    t0 = time.time()

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                run_shard,
                worker_id=i,
                shard_index=i,
                shard_count=shard_count,
                end_date=end_date,
                only_failed=args.only_failed,
                max_symbols=args.max_symbols,
                lookback_if_missing=args.lookback_if_missing,
                print_every=args.print_every,
            )
            for i in range(workers)
        ]

        for fut in as_completed(futures):
            results.append(fut.result())

    total_written = sum(r["written_partitions"] for r in results)
    total_up_to_date = sum(r["up_to_date"] for r in results)
    total_no_data = sum(r["no_data"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_symbols = sum(r["symbols"] for r in results)
    failed_paths = [r["failed_path"] for r in results]

    # Merge failed logs
    merged_failed = LOG_DIR / "daily_fetch_failed.csv"
    merge_failed_csvs(failed_paths, merged_failed)

    elapsed = time.time() - t0

    print("\n--- Shard allocation summary ---")
    for r in sorted(results, key=lambda x: x["worker_id"]):
        prev = r["preview"]
        print(f"W{r['worker_id']}: symbols={prev['count']} head={prev['head']} tail={prev['tail']}")

    print(
        f"\n[DONE] symbols={total_symbols} | written_partitions={total_written} | up_to_date={total_up_to_date} "
        f"| no_data={total_no_data} | failed={total_failed} | elapsed={elapsed:.1f}s"
    )
    print(f"[FAILED] merged failed csv: {merged_failed}")


if __name__ == "__main__":
    main()
