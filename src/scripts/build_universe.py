from __future__ import annotations

import os
import time
import random
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Set

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
OUT_PATH = Path("config/universe_daily.yml")

# You said: SP500 + Nasdaq common stocks + QQQ/IVV
ETF_SYMBOLS = {"QQQ", "IVV"}

# Pagination / safety
SLEEP_SEC = 0.25
MAX_RETRIES = 5


# -------------------------
# HTTP Helpers
# -------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = dict(params or {})
    params["apiKey"] = API_KEY

    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code in (429, 500, 502, 503, 504):
                backoff = (2 ** (attempt - 1)) * 0.6 + random.random() * 0.2
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            backoff = (2 ** (attempt - 1)) * 0.6 + random.random() * 0.2
            time.sleep(backoff)

    raise RuntimeError(f"GET failed after {MAX_RETRIES} retries: {url}") from last_err


# -------------------------
# Universe Sources
# -------------------------
def fetch_sp500_symbols(session: requests.Session) -> Set[str]:
    """
    Pull S&P 500 constituents from Wikipedia table.
    Uses requests+UA to avoid 403, then parse HTML via pandas.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = session.get(url, timeout=30)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    df = tables[0]  # first table is constituents

    syms = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Basic cleanup
    out = {s for s in syms if s and s != "NAN"}
    return out


def fetch_nasdaq_common_stocks(session: requests.Session) -> Set[str]:
    """
    Pull ALL Nasdaq (XNAS) common stocks (type=CS) from Polygon reference tickers.
    """
    url = f"{BASE_URL}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "exchange": "XNAS",
        "type": "CS",
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
    }

    out: Set[str] = set()
    while True:
        data = _get_json(session, url, params=params)

        for row in data.get("results", []):
            t = str(row.get("ticker", "")).strip().upper()
            if t:
                out.add(t)

        next_url = data.get("next_url")
        if not next_url:
            break

        # next_url already encodes params (except apiKey)
        url = next_url
        params = {}
        time.sleep(SLEEP_SEC)

    return out


# -------------------------
# Main
# -------------------------
def main() -> None:
    session = _session()

    print("Building universe (Daily MACD scope)...")

    sp500 = fetch_sp500_symbols(session)
    print(f"S&P 500 symbols: {len(sp500)}")

    nasdaq_cs = fetch_nasdaq_common_stocks(session)
    print(f"Nasdaq common stocks (CS): {len(nasdaq_cs)}")

    universe = sorted(set().union(sp500, nasdaq_cs, ETF_SYMBOLS))

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "notes": "Universe = S&P 500 (Wikipedia) + Nasdaq common stocks (Polygon XNAS + CS + active) + {QQQ, IVV}",
        "counts": {
            "sp500": len(sp500),
            "nasdaq_common_stocks": len(nasdaq_cs),
            "etfs": len(ETF_SYMBOLS),
            "total_unique": len(universe),
        },
        "symbols": universe,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    print(f"[OK] Saved universe -> {OUT_PATH} | total symbols={len(universe)}")


if __name__ == "__main__":
    main()
