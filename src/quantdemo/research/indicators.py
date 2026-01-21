from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def add_moving_averages(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    vol_ma: int = 20,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Adds:
      MA{ma_fast}, MA{ma_slow}, VMA{vol_ma}
    """
    out = df.copy()

    if price_col not in out.columns:
        raise ValueError(f"Missing price column: {price_col}")
    if volume_col not in out.columns:
        raise ValueError(f"Missing volume column: {volume_col}")

    out[f"MA{ma_fast}"] = out[price_col].rolling(ma_fast).mean()
    out[f"MA{ma_slow}"] = out[price_col].rolling(ma_slow).mean()
    out[f"VMA{vol_ma}"] = out[volume_col].rolling(vol_ma).mean()
    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Adds MACD columns:
      {prefix}MACD, {prefix}SIGNAL, {prefix}HIST
    """
    out = df.copy()

    if price_col not in out.columns:
        raise ValueError(f"Missing price column: {price_col}")

    macd_line = ema(out[price_col], fast) - ema(out[price_col], slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line

    out[f"{prefix}MACD"] = macd_line
    out[f"{prefix}SIGNAL"] = signal_line
    out[f"{prefix}HIST"] = hist
    return out


def add_indicators(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 50,
    vol_ma: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    price_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Convenience wrapper to add MA/VMA + MACD.
    """
    out = add_moving_averages(
        df,
        ma_fast=ma_fast,
        ma_slow=ma_slow,
        vol_ma=vol_ma,
        price_col=price_col,
        volume_col=volume_col,
    )
    out = add_macd(
        out,
        fast=macd_fast,
        slow=macd_slow,
        signal=macd_signal,
        price_col=price_col,
    )
    return out
