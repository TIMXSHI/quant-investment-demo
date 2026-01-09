from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# -------------------------
# Make imports work reliably
# (so "from quantdemo..." works no matter how you run this file)
# -------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]   # repo root
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now this import will work
from quantdemo.research.features import macd, rsi, atr  # noqa: E402


# -------------------------
# Config
# -------------------------
RAW_DIR = ROOT_DIR / "data" / "raw" / "polygon" / "1day"
FEAT_DIR = ROOT_DIR / "data" / "features" / "polygon" / "1day"

# You can customize this list any time
SYMBOLS: List[str] = ["AAPL", "TSLA", "INTC"]

# Optional: keep only useful columns in output (set to None to keep all)
KEEP_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi", "atr",
    "macd", "macd_signal", "macd_hist"
]


def _load_raw(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw data for {symbol}: {path}\n"
            f"Run fetch_ohlcv.py first to create parquet files."
        )

    df = pd.read_parquet(path)

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Sort + de-dup (safe guard)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    # MACD returns 3 columns: macd, macd_signal, macd_hist
    macd_df = macd(df["close"])

    # RSI, ATR
    df = df.copy()
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)

    feat_df = pd.concat([df, macd_df], axis=1)

    # dtype optimization (optional)
    for c in ["open", "high", "low", "close", "macd", "macd_signal", "macd_hist", "rsi", "atr"]:
        if c in feat_df.columns:
            feat_df[c] = feat_df[c].astype("float32")
    if "volume" in feat_df.columns:
        # keep volume as int
        feat_df["volume"] = feat_df["volume"].astype("int64")

    # Keep only selected cols if provided
    if KEEP_COLS is not None:
        cols = [c for c in KEEP_COLS if c in feat_df.columns]
        feat_df = feat_df[cols]

    return feat_df


def _save_features(symbol: str, feat_df: pd.DataFrame) -> Path:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEAT_DIR / f"{symbol}.parquet"

    # Always sort + unique index
    feat_df = feat_df[~feat_df.index.duplicated(keep="last")].sort_index()

    feat_df.to_parquet(out_path, index=True)
    return out_path


def main():
    print(f"RAW_DIR : {RAW_DIR}")
    print(f"FEAT_DIR: {FEAT_DIR}")
    print(f"SYMBOLS : {SYMBOLS}")

    for sym in SYMBOLS:
        print(f"\n[BUILD] {sym}")
        df = _load_raw(sym)
        feat_df = _build_features(df)
        out_path = _save_features(sym, feat_df)

        # Quick sanity output
        # (show last 3 rows with key columns)
        preview_cols = [c for c in ["close", "macd", "macd_signal", "macd_hist", "rsi", "atr"] if c in feat_df.columns]
        print(feat_df[preview_cols].tail(3))
        print(f"[OK] Saved -> {out_path} | rows={len(feat_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
