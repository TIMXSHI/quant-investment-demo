from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

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
DATA_DIR = Path("data/raw/polygon")  # provider root
UNIVERSE_PATH = Path("config/universe.yml")

# For demo & mid/low frequency: daily bars
DEFAULT_MULTIPLIER = 1
DEFAULT_TIMESPAN = "day"  # minute/hour/day/week/month

# Rate limiting safety (adjust if your plan allows more)
SLEEP_BETWEEN_CALLS_SEC = 0.25

@dataclass
class FetchSpec:
    symbol: str
    multiplier: int = DEFAULT_MULTIPLIER
    timespan: str = DEFAULT_TIMESPAN
    start: str = "2020-01-01"   # YYYY-MM-DD
    end: str = None            # None = today (we'll set)


# -------------------------
# Helpers: cache
# -------------------------
def cache_path(symbol: str, multiplier: int, timespan: str) -> Path:
    # e.g. data/raw/polygon/1day/AAPL.parquet
    interval = f"{multiplier}{timespan}"
    return DATA_DIR / interval / f"{symbol}.parquet"


def load_cached(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_parquet(path)
        if not df.empty:
            df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def save_cached(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure sorted + unique index
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path, index=True)


def merge_incremental(old: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    df = pd.concat([old, new], axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# -------------------------
# Polygon API: aggregates
# -------------------------
def _request_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(params or {})
    params["apiKey"] = API_KEY

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Polygon API error {r.status_code}: {r.text[:300]}")
    return r.json()


def fetch_aggs(symbol: str, multiplier: int, timespan: str, start: str, end: str,
               adjusted: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV aggregate bars from Polygon.
    Endpoint style widely used: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    (example shown in many Polygon docs/tutorials).:contentReference[oaicite:2]{index=2}
    Handles pagination via next_url when result set is large.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000
    }

    all_rows: List[Dict[str, Any]] = []

    while True:
        data = _request_json(url, params=params)
        results = data.get("results", [])
        if results:
            all_rows.extend(results)

        # Polygon may return pagination link when > limit
        next_url = data.get("next_url")
        if not next_url:
            break

        # next_url is usually a full URL without apiKey; keep using our helper
        url = next_url if next_url.startswith("http") else f"{BASE_URL}{next_url}"
        params = {}  # next_url already encodes most params; only apiKey will be injected
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    if not all_rows:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Polygon aggs fields (common): o,h,l,c,v,t (ms epoch)
    # normalize
    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp_ms"
    })

    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]].copy()

    # dtype optimization for speed/storage
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype("float32")
    df["volume"] = df["volume"].astype("int64")

    return df.sort_index()


# -------------------------
# Universe
# -------------------------
def load_universe(path: Path) -> List[str]:
    if not path.exists():
        # default demo
        return ["AAPL", "TSLA", "INTC"]
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    syms = obj.get("symbols", [])
    if not syms:
        return ["AAPL", "TSLA", "INTC"]
    return [str(s).strip().upper() for s in syms if str(s).strip()]


# -------------------------
# Main
# -------------------------
def main():
    symbols = load_universe(UNIVERSE_PATH)

    # end date: today in YYYY-MM-DD (Polygon accepts date strings in many examples)
    today = pd.Timestamp.utcnow().date().isoformat()

    multiplier = DEFAULT_MULTIPLIER
    timespan = DEFAULT_TIMESPAN

    print(f"Provider=polygon | interval={multiplier}{timespan} | symbols={symbols}")

    for sym in symbols:
        p = cache_path(sym, multiplier, timespan)
        old = load_cached(p)

        # incremental start: last date + 1 day (for daily bars)
        if old is not None and not old.empty:
            last_dt = old.index.max()
            # start next day (safe)
            start = (last_dt + pd.Timedelta(days=1)).date().isoformat()
        else:
            start = "2020-01-01"

        # if already up-to-date
        if start >= today:
            print(f"[SKIP] {sym} already up-to-date ({start} >= {today})")
            continue

        print(f"[FETCH] {sym} {start} -> {today}")
        df_new = fetch_aggs(sym, multiplier, timespan, start, today, adjusted=True)
        if df_new.empty:
            print(f"[WARN] No new data for {sym}")
            continue

        df_all = merge_incremental(old, df_new)
        save_cached(df_all, p)
        print(f"[OK] {sym} rows={len(df_all)} saved -> {p}")

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)


if __name__ == "__main__":
    main()