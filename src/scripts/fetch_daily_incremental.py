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
FAILED_CSV = LOG_DIR / "daily_fetch_failed.csv"

BASE_SLEEP_SEC = 0.15
MAX_RETRIES = 6


# -------------------------
# Helpers
# -------------------------
def last_business_day_simple(d: date) -> date:
    """
    Simple weekend rollback (no holiday calendar).
    For US holidays, Polygon will simply return no bar for that date anyway.
    """
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


def append_failed(symbol: str, error: str) -> None:
    new_file = not FAILED_CSV.exists()
    with FAILED_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts_utc", "symbol", "error"])
        w.writerow([pd.Timestamp.utcnow().isoformat(), symbol, error])


def load_failed_symbols() -> List[str]:
    if not FAILED_CSV.exists():
        return []
    df = pd.read_csv(FAILED_CSV)
    if "symbol" not in df.columns:
        return []
    return sorted(set(df["symbol"].astype(str).str.upper().str.strip().tolist()))


def read_marker_last_date(symbol: str) -> Optional[date]:
    p = _done_marker(symbol)
    if not p.exists():
        return None
    try:
        meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        last_date = meta.get("last_date")
        if last_date:
            return datetime.fromisoformat(str(last_date)).date()
        # fallback: if old marker only has window.end
        window = meta.get("window") or {}
        end_s = window.get("end")
        if end_s:
            return datetime.fromisoformat(str(end_s)).date()
    except Exception:
        return None
    return None


def scan_fs_last_date(symbol: str) -> Optional[date]:
    """
    Fallback: scan data/raw partitions for this symbol and find max date.
    Assumes folders:
      symbol=XXX/year=YYYY/month=MM/day=DD/bars.parquet
    """
    sym_root = DATA_ROOT / f"symbol={symbol}"
    if not sym_root.exists():
        return None

    # Fast-ish: only scan day folders that actually have bars.parquet
    # (365 files per symbol is OK)
    max_d: Optional[date] = None
    for p in sym_root.rglob("bars.parquet"):
        try:
            # .../year=YYYY/month=MM/day=DD/bars.parquet
            day_part = p.parent.name  # day=DD
            month_part = p.parent.parent.name  # month=MM
            year_part = p.parent.parent.parent.name  # year=YYYY
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
    """
    Prefer marker (cheap), else scan filesystem.
    """
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
            # keep original symbol for folder partitioning
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
# CLI
# -------------------------
@dataclass
class Args:
    only_failed: bool
    shard_index: int
    shard_count: int
    max_symbols: Optional[int]
    lookback_if_missing: int
    end: Optional[str]


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--only-failed", action="store_true", help="Only fetch symbols listed in logs/daily_fetch_failed.csv")
    p.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    p.add_argument("--shard-count", type=int, default=1, help="Total shards")
    p.add_argument("--max-symbols", type=int, default=None, help="Optional cap for safety")
    p.add_argument("--lookback-if-missing", type=int, default=365, help="If symbol has no local data, fetch past N days")
    p.add_argument("--end", default=None, help="Override end date (YYYY-MM-DD). Default: last business day of yesterday")
    a = p.parse_args()
    return Args(
        only_failed=a.only_failed,
        shard_index=a.shard_index,
        shard_count=a.shard_count,
        max_symbols=a.max_symbols,
        lookback_if_missing=int(a.lookback_if_missing),
        end=a.end,
    )


def main() -> None:
    args = parse_args()
    session = _session()

    if args.end:
        end_date = datetime.fromisoformat(args.end).date()
    else:
        # Run after market close: usually "yesterday" is safe
        end_date = last_business_day_simple(date.today() - timedelta(days=1))

    universe = load_universe_symbols(UNIVERSE_PATH)

    if args.only_failed:
        failed = set(load_failed_symbols())
        universe = [s for s in universe if s in failed]
        print(f"Mode=only_failed | symbols={len(universe)}")
    else:
        print(f"Mode=universe | symbols={len(universe)}")

    universe = shard_symbols(universe, args.shard_index, args.shard_count)
    print(f"Shard {args.shard_index}/{args.shard_count} -> {len(universe)} symbols")

    if args.max_symbols is not None:
        universe = universe[: args.max_symbols]
        print(f"Cap max_symbols={args.max_symbols} -> {len(universe)} symbols")

    print(f"End date target: {end_date}")
    print(f"Output root: {DATA_ROOT.resolve()}")

    total_written = 0
    total_no_new = 0
    total_no_data = 0
    total_failed = 0

    for i, sym in enumerate(universe, start=1):
        try:
            last_local = get_last_local_date(sym)

            if last_local is None:
                # first-time / missing symbol data -> backfill lookback window
                start_date = end_date - timedelta(days=args.lookback_if_missing)
                mode = "bootstrap"
            else:
                start_date = last_local + timedelta(days=1)
                mode = "incremental"

            if start_date > end_date:
                total_no_new += 1
                print(f"[{i}/{len(universe)}] {sym}: SKIP (up-to-date) last_local={last_local}")
                continue

            df, used_ticker = fetch_daily_bars_with_fallback(session, sym, start_date, end_date)

            if df.empty:
                total_no_data += 1
                # update marker anyway, keep last_local
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
                print(f"[{i}/{len(universe)}] {sym}: no new bars returned ({start_date} -> {end_date})")
            else:
                written = save_partitioned_by_day(df)
                total_written += written
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
                print(f"[{i}/{len(universe)}] {sym}: mode={mode} rows={len(df)} written={written} last_date={new_last}")

        except Exception as e:
            total_failed += 1
            append_failed(sym, repr(e))
            print(f"[{i}/{len(universe)}] {sym}: ERROR -> {e}")

        time.sleep(BASE_SLEEP_SEC)

    print(
        f"\n[DONE] written_partitions={total_written} | up_to_date={total_no_new} | no_data={total_no_data} | failed={total_failed} | failed_log={FAILED_CSV}"
    )


if __name__ == "__main__":
    main()
