from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -------------------------
# Repo import path
# -------------------------
SRC_DIR = Path(__file__).resolve().parents[1]  # .../src
sys.path.insert(0, str(SRC_DIR))


# -------------------------
# Helpers
# -------------------------
def parse_iso_date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def fmt_secs(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def daterange(d0: date, d1: date) -> Iterable[date]:
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def last_business_day(d: date) -> date:
    # simple Mon-Fri fallback; real trading days are determined by "has data file"
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def get_repo_root() -> Path:
    # src/scripts/*.py -> parents[2] => repo root
    return Path(__file__).resolve().parents[2]


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def list_symbols_from_raw(raw_root: Path) -> List[str]:
    """
    raw_root example:
      data/raw/polygon/1day
        symbol=AAPL/year=2025/month=07/day=01/*.parquet
    """
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    syms: List[str] = []
    for p in raw_root.glob("symbol=*"):
        if p.is_dir():
            syms.append(p.name.split("=", 1)[1])
    return sorted(set(syms))


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize date + OHLCV naming.
    Expect at least date + close.
    """
    if "date" not in df.columns:
        for c in ["Date", "datetime", "time", "timestamp"]:
            if c in df.columns:
                df = df.rename(columns={c: "date"})
                break

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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    return df


def read_one_raw_day(raw_root: Path, symbol: str, d: date) -> Optional[pd.DataFrame]:
    """
    Read raw daily parquet(s) for one symbol and one day.
    Path:
      raw_root/symbol=XXX/year=YYYY/month=MM/day=DD/*.parquet
    """
    day_dir = raw_root / f"symbol={symbol}" / f"year={d.year:04d}" / f"month={d.month:02d}" / f"day={d.day:02d}"
    if not day_dir.exists():
        return None

    files = sorted(day_dir.glob("*.parquet"))
    if not files:
        return None

    # Usually one file; if multiple, concat
    parts = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df = normalize_ohlcv_columns(df)
            parts.append(df)
        except Exception:
            continue

    if not parts:
        return None

    out = pd.concat(parts, ignore_index=True)
    # keep the row for that date (defensive)
    if "date" in out.columns:
        out = out[out["date"] == d]
    if out.empty:
        return None

    # If there are multiple rows for same day (unlikely), keep last
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out


def load_raw_daily_range(raw_root: Path, symbol: str, start: date, end: date) -> pd.DataFrame:
    """
    Load raw daily bars for [start..end] by reading each day folder.
    This is efficient when the range is small (incremental).
    """
    parts: List[pd.DataFrame] = []
    for d in daterange(start, end):
        df = read_one_raw_day(raw_root, symbol, d)
        if df is not None and not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = normalize_ohlcv_columns(out)
    if "date" not in out.columns:
        return pd.DataFrame()

    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    out.index = pd.to_datetime(out["date"])
    return out


# -------------------------
# MACD EMA incremental math
# -------------------------
@dataclass
class MacdEmaParams:
    ema_fast: int = 12
    ema_slow: int = 26
    signal: int = 9


def alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def ema_next(prev_ema: float, price: float, span: int) -> float:
    a = alpha(span)
    return a * price + (1.0 - a) * prev_ema


def get_symbol_state_from_features(features_df: pd.DataFrame, symbol: str) -> Optional[dict]:
    """
    Pull last known state from existing features table for one symbol:
      last_date, EMA_fast, EMA_slow, SIGNAL
    """
    sub = features_df[features_df["symbol"] == symbol]
    if sub.empty:
        return None

    # date column might be str in file; normalize
    if sub["date"].dtype == object:
        # handle iso string
        dser = pd.to_datetime(sub["date"], errors="coerce").dt.date
    else:
        dser = pd.to_datetime(sub["date"], errors="coerce").dt.date

    sub = sub.copy()
    sub["_d"] = dser
    sub = sub.dropna(subset=["_d"]).sort_values("_d")
    if sub.empty:
        return None

    last = sub.iloc[-1]
    state = {
        "last_date": last["_d"],
        "EMA_fast": safe_float(last.get(f"EMA_{int(last.get('ema_fast', 0) or 0)}", last.get("EMA_fast", last.get("EMA_12")))),  # legacy safe
    }
    # We won't rely on this messy key above; we will read explicit EMA_{span} names later.
    return None  # we will use a cleaner state getter below


def ensure_features_schema(df: pd.DataFrame, params: MacdEmaParams) -> pd.DataFrame:
    """
    Ensure required columns exist in features df.
    """
    needed = [
        "date",
        "symbol",
        "close",
        "MACD",
        "SIGNAL",
        "HIST",
        "macd_diff",
        "golden_cross",
        "dead_cross",
        f"EMA_{params.ema_fast}",
        f"EMA_{params.ema_slow}",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def load_existing_features(features_file: Path) -> pd.DataFrame:
    if not features_file.exists():
        return pd.DataFrame()
    df = pd.read_parquet(features_file)
    return df


def get_clean_state(features_df: pd.DataFrame, symbol: str, params: MacdEmaParams) -> Optional[dict]:
    """
    Return:
      last_date, prev_ema_fast, prev_ema_slow, prev_signal
    Requires columns:
      EMA_{fast}, EMA_{slow}, SIGNAL, date
    """
    sub = features_df[features_df["symbol"] == symbol].copy()
    if sub.empty:
        return None

    # normalize date to python date
    sub["_d"] = pd.to_datetime(sub["date"], errors="coerce").dt.date
    sub = sub.dropna(subset=["_d"]).sort_values("_d")
    if sub.empty:
        return None

    last = sub.iloc[-1]
    ema_f_col = f"EMA_{params.ema_fast}"
    ema_s_col = f"EMA_{params.ema_slow}"

    if ema_f_col not in sub.columns or ema_s_col not in sub.columns or "SIGNAL" not in sub.columns:
        return None

    prev_ema_f = safe_float(last[ema_f_col])
    prev_ema_s = safe_float(last[ema_s_col])
    prev_sig = safe_float(last["SIGNAL"])

    # if any is nan, treat as missing state
    if pd.isna(prev_ema_f) or pd.isna(prev_ema_s) or pd.isna(prev_sig):
        return None

    return {
        "last_date": last["_d"],
        "ema_fast": float(prev_ema_f),
        "ema_slow": float(prev_ema_s),
        "signal": float(prev_sig),
    }


def compute_initial_state_from_history(
    raw_root: Path,
    symbol: str,
    params: MacdEmaParams,
    end_date: date,
    warmup_days: int = 300,
) -> Tuple[Optional[dict], pd.DataFrame]:
    """
    Fallback when no state exists in features:
    - Load up to warmup_days history ending at end_date
    - Compute EMA/MACD/SIGNAL sequentially
    - Return last state + full computed history DF (so we can also write missing ranges)
    """
    start = end_date - timedelta(days=warmup_days)
    raw = load_raw_daily_range(raw_root, symbol, start, end_date)
    if raw.empty or "close" not in raw.columns:
        return None, pd.DataFrame()

    closes = pd.to_numeric(raw["close"], errors="coerce").tolist()
    dates = raw["date"].tolist()

    # need at least some points
    if len(closes) < max(params.ema_slow, params.signal) + 5:
        return None, pd.DataFrame()

    # seed EMA with first close (simple, stable for long warmup)
    ema_f = closes[0]
    ema_s = closes[0]
    macd = 0.0
    sig = 0.0  # seed signal with 0; will converge over warmup

    rows: List[dict] = []
    for i in range(len(closes)):
        c = closes[i]
        d = dates[i]
        if pd.isna(c):
            continue

        if i == 0:
            ema_f = c
            ema_s = c
            macd = ema_f - ema_s
            sig = macd  # seed signal with macd on first point
        else:
            ema_f = ema_next(ema_f, c, params.ema_fast)
            ema_s = ema_next(ema_s, c, params.ema_slow)
            macd = ema_f - ema_s
            sig = ema_next(sig, macd, params.signal)

        hist = macd - sig
        rows.append(
            {
                "date": d.isoformat() if isinstance(d, date) else str(d),
                "symbol": symbol,
                "close": float(c),
                f"EMA_{params.ema_fast}": float(ema_f),
                f"EMA_{params.ema_slow}": float(ema_s),
                "MACD": float(macd),
                "SIGNAL": float(sig),
                "HIST": float(hist),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return None, pd.DataFrame()

    # compute cross flags inside this computed history
    out["_d"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.sort_values("_d").reset_index(drop=True)

    macd_ser = pd.to_numeric(out["MACD"], errors="coerce")
    sig_ser = pd.to_numeric(out["SIGNAL"], errors="coerce")
    golden = (macd_ser > sig_ser) & (macd_ser.shift(1) <= sig_ser.shift(1))
    dead = (macd_ser < sig_ser) & (macd_ser.shift(1) >= sig_ser.shift(1))

    out["macd_diff"] = (macd_ser - sig_ser).astype(float)
    out["golden_cross"] = golden.fillna(False).astype(bool)
    out["dead_cross"] = dead.fillna(False).astype(bool)

    # last state
    last = out.iloc[-1]
    state = {
        "last_date": last["_d"],
        "ema_fast": float(last[f"EMA_{params.ema_fast}"]),
        "ema_slow": float(last[f"EMA_{params.ema_slow}"]),
        "signal": float(last["SIGNAL"]),
    }
    out = out.drop(columns=["_d"], errors="ignore")
    return state, out


def compute_incremental_for_symbol(
    raw_root: Path,
    symbol: str,
    params: MacdEmaParams,
    state: dict,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Using existing EMA/MACD signal state, compute new rows for [start_date..end_date]
    ONLY for days that actually exist in raw data (trading days).
    """
    raw = load_raw_daily_range(raw_root, symbol, start_date, end_date)
    if raw.empty or "close" not in raw.columns:
        return pd.DataFrame()

    closes = pd.to_numeric(raw["close"], errors="coerce").tolist()
    dates = raw["date"].tolist()

    ema_f = float(state["ema_fast"])
    ema_s = float(state["ema_slow"])
    sig = float(state["signal"])

    rows: List[dict] = []
    for i in range(len(closes)):
        c = closes[i]
        d = dates[i]
        if pd.isna(c):
            continue

        ema_f = ema_next(ema_f, float(c), params.ema_fast)
        ema_s = ema_next(ema_s, float(c), params.ema_slow)
        macd = ema_f - ema_s
        sig = ema_next(sig, macd, params.signal)
        hist = macd - sig

        rows.append(
            {
                "date": d.isoformat() if isinstance(d, date) else str(d),
                "symbol": symbol,
                "close": float(c),
                f"EMA_{params.ema_fast}": float(ema_f),
                f"EMA_{params.ema_slow}": float(ema_s),
                "MACD": float(macd),
                "SIGNAL": float(sig),
                "HIST": float(hist),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame()

    # crosses need previous day values; compute crosses inside the incremental chunk,
    # but we should include the last existing record to compute first day's cross correctly.
    out["_d"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    out = out.sort_values("_d").reset_index(drop=True)

    macd_ser = pd.to_numeric(out["MACD"], errors="coerce")
    sig_ser = pd.to_numeric(out["SIGNAL"], errors="coerce")
    golden = (macd_ser > sig_ser) & (macd_ser.shift(1) <= sig_ser.shift(1))
    dead = (macd_ser < sig_ser) & (macd_ser.shift(1) >= sig_ser.shift(1))

    out["macd_diff"] = (macd_ser - sig_ser).astype(float)
    out["golden_cross"] = golden.fillna(False).astype(bool)
    out["dead_cross"] = dead.fillna(False).astype(bool)

    out = out.drop(columns=["_d"], errors="ignore")
    return out


# -------------------------
# Main update routine
# -------------------------
def update_macd_incremental_raw_daily(
    raw_root: Path,
    features_file: Path,
    params: MacdEmaParams,
    symbols: Optional[List[str]] = None,
    end_date: Optional[date] = None,
    warmup_days: int = 300,
    show_progress: bool = True,
) -> None:
    """
    Incremental updater:
    - Load existing features parquet (if exists)
    - For each symbol:
        - If state exists: compute from last_date+1 to end_date (only trading days with raw files)
        - If missing state: compute warmup history ending at end_date; then append only what features lacks
    - Deduplicate and save
    """
    t0 = time.time()
    features_df = load_existing_features(features_file)

    # End date default: today -> last_business_day
    if end_date is None:
        end_date = last_business_day(date.today())

    # Determine symbol list
    if symbols is None:
        symbols = list_symbols_from_raw(raw_root)
    if not symbols:
        raise ValueError(f"No symbols found under raw root: {raw_root}")

    # Ensure schema columns exist (so later concat is consistent)
    features_df = ensure_features_schema(features_df, params)

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(symbols, desc="Symbols", unit="sym") if show_progress else symbols
    except Exception:
        iterator = symbols

    appended = 0
    skipped = 0
    errors = 0

    print(f"[INFO] Raw root     : {raw_root.resolve()}")
    print(f"[INFO] Features file: {features_file.resolve()}")
    print(f"[INFO] Params       : EMA({params.ema_fast},{params.ema_slow}) SIGNAL={params.signal}")
    print(f"[INFO] Warmup days  : {warmup_days}")
    print(f"[INFO] End date     : {end_date}")
    print(f"[INFO] Symbols      : {len(symbols):,}")
    print()

    new_parts: List[pd.DataFrame] = []

    for sym in iterator:
        try:
            state = get_clean_state(features_df, sym, params)

            if state is not None:
                # continue from last_date+1
                start = state["last_date"] + timedelta(days=1)
                if start > end_date:
                    skipped += 1
                    continue

                inc = compute_incremental_for_symbol(
                    raw_root=raw_root,
                    symbol=sym,
                    params=params,
                    state=state,
                    start_date=start,
                    end_date=end_date,
                )
                if inc.empty:
                    skipped += 1
                    continue

                new_parts.append(inc)
                appended += len(inc)
            else:
                # No usable state -> fallback compute warmup history and append missing rows
                state2, hist = compute_initial_state_from_history(
                    raw_root=raw_root,
                    symbol=sym,
                    params=params,
                    end_date=end_date,
                    warmup_days=warmup_days,
                )
                if hist.empty:
                    skipped += 1
                    continue

                # Append only dates that are not already in features_df for this symbol
                existing_dates = set(
                    pd.to_datetime(features_df.loc[features_df["symbol"] == sym, "date"], errors="coerce")
                    .dt.date.dropna()
                    .tolist()
                )
                hist["_d"] = pd.to_datetime(hist["date"], errors="coerce").dt.date
                hist2 = hist[~hist["_d"].isin(existing_dates)].drop(columns=["_d"], errors="ignore")
                if hist2.empty:
                    skipped += 1
                    continue

                new_parts.append(hist2)
                appended += len(hist2)

        except Exception:
            errors += 1
            continue

    if not new_parts:
        elapsed = time.time() - t0
        print(f"[OK] No new rows to append. skipped={skipped:,} errors={errors:,} elapsed={fmt_secs(elapsed)}")
        return

    new_df = pd.concat(new_parts, ignore_index=True)

    # Ensure schema in new_df too
    new_df = ensure_features_schema(new_df, params)

    # Combine and dedupe
    combined = pd.concat([features_df, new_df], ignore_index=True)

    # Normalize date
    combined["_d"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
    combined = combined.dropna(subset=["_d", "symbol"])
    combined = combined.sort_values(["symbol", "_d"])

    # Deduplicate (keep last)
    combined = combined.drop_duplicates(subset=["symbol", "_d"], keep="last")

    # Write back date as iso string for stable parquet
    combined["date"] = combined["_d"].apply(lambda x: x.isoformat())
    combined = combined.drop(columns=["_d"], errors="ignore")

    # Final sort
    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    features_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(features_file, index=False)

    elapsed = time.time() - t0
    print()
    print(f"[OK] Appended rows : {appended:,}")
    print(f"[OK] Skipped symbols: {skipped:,}")
    print(f"[OK] Errors        : {errors:,}")
    print(f"[OK] Saved         : {features_file.resolve()}")
    print(f"[OK] Elapsed       : {fmt_secs(elapsed)}")


# -------------------------
# CLI
# -------------------------
def main():
    repo_root = get_repo_root()

    parser = argparse.ArgumentParser(description="Incremental MACD updater from RAW daily parquet (EMA-based).")
    parser.add_argument("--timeframe", default="1day", help="raw timeframe folder (default: 1day)")
    parser.add_argument("--end", default=None, help="end date (YYYY-MM-DD). default=today(last business day)")
    parser.add_argument("--ema-fast", type=int, default=12, help="EMA fast span (default: 12)")
    parser.add_argument("--ema-slow", type=int, default=26, help="EMA slow span (default: 26)")
    parser.add_argument("--signal", type=int, default=9, help="Signal EMA span (default: 9)")
    parser.add_argument("--warmup-days", type=int, default=300, help="fallback warmup history days (default: 300)")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm")
    args = parser.parse_args()

    raw_root = repo_root / "data" / "raw" / "polygon" / args.timeframe
    features_file = repo_root / "data" / "features" / "polygon" / args.timeframe / "indicators_macd_daily.parquet"

    end_date = parse_iso_date(args.end) if args.end else None

    params = MacdEmaParams(
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        signal=int(args.signal),
    )

    update_macd_incremental_raw_daily(
        raw_root=raw_root,
        features_file=features_file,
        params=params,
        symbols=None,
        end_date=end_date,
        warmup_days=int(args.warmup_days),
        show_progress=(not args.no_progress),
    )


if __name__ == "__main__":
    main()
