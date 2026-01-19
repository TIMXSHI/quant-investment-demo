from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import date, timedelta
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

# Output partition style (matches your screenshot)
DATA_ROOT = Path("data/raw/polygon/1day")

# Progress + logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
FAILED_CSV = LOG_DIR / "daily_fetch_failed.csv"

LOOKBACK_DAYS = 365  # past 12 months
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)

# Global pacing (base)
BASE_SLEEP_SEC = 0.15

# HTTP retry
MAX_RETRIES = 6


# -------------------------
# Helpers
# -------------------------
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


def _get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Robust GET with retries/backoff for 429/5xx and network errors.
    """
    params = dict(params or {})
    params["apiKey"] = API_KEY

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)

            # Retryable status codes
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
    """
    Deterministic sharding: take every Nth symbol.
    shard_index: 0..N-1
    """
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


def is_done(symbol: str) -> bool:
    return _done_marker(symbol).exists()


def mark_done(symbol: str, meta: Dict[str, Any]) -> None:
    p = _done_marker(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


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
    """
    Returns (df, used_ticker)
    """
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
        g["date"] = pd.to_datetime(g["date"])  # parquet-friendly
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


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--only-failed", action="store_true", help="Only fetch symbols listed in logs/daily_fetch_failed.csv")
    p.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    p.add_argument("--shard-count", type=int, default=1, help="Total shards")
    p.add_argument("--max-symbols", type=int, default=None, help="Optional cap for safety")
    a = p.parse_args()
    return Args(
        only_failed=a.only_failed,
        shard_index=a.shard_index,
        shard_count=a.shard_count,
        max_symbols=a.max_symbols,
    )


def main() -> None:
    args = parse_args()
    session = _session()

    universe = load_universe_symbols(UNIVERSE_PATH)

    if args.only_failed:
        failed = set(load_failed_symbols())
        universe = [s for s in universe if s in failed]
        print(f"Mode=only_failed | symbols={len(universe)}")
    else:
        print(f"Mode=universe | symbols={len(universe)}")

    # shard
    universe = shard_symbols(universe, args.shard_index, args.shard_count)
    print(f"Shard {args.shard_index}/{args.shard_count} -> {len(universe)} symbols")

    if args.max_symbols is not None:
        universe = universe[: args.max_symbols]
        print(f"Cap max_symbols={args.max_symbols} -> {len(universe)} symbols")

    print(f"Date window: {START_DATE} -> {END_DATE}")
    print(f"Output root: {DATA_ROOT.resolve()}")

    total_written = 0
    total_done_skipped = 0
    total_empty = 0

    for i, sym in enumerate(universe, start=1):
        if is_done(sym):
            total_done_skipped += 1
            print(f"[{i}/{len(universe)}] {sym}: SKIP (done)")
            continue

        try:
            df, used_ticker = fetch_daily_bars_with_fallback(session, sym, START_DATE, END_DATE)

            if df.empty:
                total_empty += 1
                mark_done(
                    sym,
                    {
                        "status": "no_data",
                        "symbol": sym,
                        "used_ticker": used_ticker,
                        "window": {"start": START_DATE.isoformat(), "end": END_DATE.isoformat()},
                        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
                    },
                )
                print(f"[{i}/{len(universe)}] {sym}: no data (marked done)")
            else:
                written = save_partitioned_by_day(df)
                total_written += written
                mark_done(
                    sym,
                    {
                        "status": "ok",
                        "symbol": sym,
                        "used_ticker": used_ticker,
                        "rows": int(len(df)),
                        "partitions_written": int(written),
                        "window": {"start": START_DATE.isoformat(), "end": END_DATE.isoformat()},
                        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
                    },
                )
                print(f"[{i}/{len(universe)}] {sym}: rows={len(df)} written={written}")

        except Exception as e:
            append_failed(sym, repr(e))
            print(f"[{i}/{len(universe)}] {sym}: ERROR -> {e}")

        # gentle pacing
        time.sleep(BASE_SLEEP_SEC)

    print(
        f"\n[DONE] written_partitions={total_written} | skipped_done={total_done_skipped} | no_data={total_empty} | failed_log={FAILED_CSV}"
    )


if __name__ == "__main__":
    main()
