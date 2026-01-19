from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import requests
from dotenv import load_dotenv


# -------------------------
# Config
# -------------------------
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
if not API_KEY:
    raise RuntimeError("Missing POLYGON_API_KEY in environment (.env).")

BASE_URL = "https://api.polygon.io"

# Only test with 2 symbols
SYMBOLS = ["QQQ", "IVV"]

# Hourly bars
MULTIPLIER = 1
TIMESPAN = "hour"

LOOKBACK_YEARS = 1
SLEEP_BETWEEN_CALLS_SEC = 0.25

DATA_ROOT = Path("data/raw/polygon/1hour")


# -------------------------
# Polygon API
# -------------------------
def _request_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(params or {})
    params["apiKey"] = API_KEY

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Polygon API error {r.status_code}: {r.text[:300]}")
    return r.json()


def fetch_aggs(
    symbol: str,
    multiplier: int,
    timespan: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    rows: List[Dict[str, Any]] = []

    while True:
        data = _request_json(url, params)
        rows.extend(data.get("results", []))

        next_url = data.get("next_url")
        if not next_url:
            break

        url = next_url if next_url.startswith("http") else f"{BASE_URL}{next_url}"
        params = {}
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "timestamp_ms",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["symbol"] = symbol

    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("int64")

    return df.sort_values("timestamp")


# -------------------------
# Partitioned write
# -------------------------
def write_partitioned(df: pd.DataFrame) -> None:
    """
    data/raw/polygon/1hour/
      symbol=XXX/year=YYYY/month=MM/day=DD/part_*.parquet
    """
    if df.empty:
        return

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day

    for (symbol, y, m, d), g in df.groupby(["symbol", "year", "month", "day"]):
        out_dir = (
            DATA_ROOT
            / f"symbol={symbol}"
            / f"year={y}"
            / f"month={m:02d}"
            / f"day={d:02d}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"part_{int(time.time())}.parquet"
        g.drop(columns=["year", "month", "day"]).to_parquet(out_path, index=False)


# -------------------------
# Main (INITIAL ONLY)
# -------------------------
def main():
    end_dt = pd.Timestamp.utcnow().date()
    start_dt = end_dt - pd.DateOffset(years=LOOKBACK_YEARS)

    start = start_dt.date().isoformat()
    end = end_dt.isoformat()

    print(f"[INIT] 1hour bars | {start} -> {end} | symbols={SYMBOLS}")

    for sym in SYMBOLS:
        print(f"[FETCH] {sym}")
        df = fetch_aggs(sym, MULTIPLIER, TIMESPAN, start, end)

        if df.empty:
            print(f"[WARN] No data for {sym}")
            continue

        write_partitioned(df)
        print(f"[OK] {sym} rows={len(df)} written (partitioned)")

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)


if __name__ == "__main__":
    main()
