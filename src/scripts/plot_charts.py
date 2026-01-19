from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data/raw/polygon/1day")
SYMBOL = "AAPL"

START_DATE = "2025-01-01"
END_DATE   = "2026-01-16"

# Moving averages
MA_FAST = 20
MA_SLOW = 50

VOL_MA = 20

# MACD params
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# -------------------------
# Loader (fast): only read needed day partitions
# -------------------------
def load_symbol_daily_range(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    sym_dir = DATA_DIR / f"symbol={symbol}"

    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()

    files = []
    d = start
    while d <= end:
        f = (
            sym_dir
            / f"year={d.year:04d}"
            / f"month={d.month:02d}"
            / f"day={d.day:02d}"
            / "bars.parquet"
        )
        if f.exists():
            files.append(f)
        d += timedelta(days=1)

    if not files:
        raise FileNotFoundError(f"No bars.parquet found for {symbol} in {start_date} ~ {end_date}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )

    # numeric safety
    for c in ["open", "high", "low", "close", "volume", "vwap", "transactions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.loc[start_date:end_date].copy()
    return df


# -------------------------
# Indicators
# -------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Price MAs
    df[f"MA{MA_FAST}"] = df["close"].rolling(MA_FAST).mean()
    df[f"MA{MA_SLOW}"] = df["close"].rolling(MA_SLOW).mean()

    # Volume MA
    df[f"VMA{VOL_MA}"] = df["volume"].rolling(VOL_MA).mean()

    # MACD
    macd_line = ema(df["close"], MACD_FAST) - ema(df["close"], MACD_SLOW)
    signal_line = ema(macd_line, MACD_SIGNAL)
    hist = macd_line - signal_line

    df["MACD"] = macd_line
    df["SIGNAL"] = signal_line
    df["HIST"] = hist
    return df


# -------------------------
# Plot
# -------------------------
df = load_symbol_daily_range(SYMBOL, START_DATE, END_DATE)
if df.empty:
    raise ValueError(f"{SYMBOL} has no data within {START_DATE} ~ {END_DATE}")

df = add_indicators(df)

# Build x-axis values as strings -> treat axis as "category" => no gaps for weekends/holidays
x = df.index.strftime("%Y-%m-%d").tolist()

# Volume colors by up/down
up = df["close"] >= df["open"]
vol_colors = ["green" if u else "red" for u in up]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.60, 0.20, 0.20],
    subplot_titles=(
        f"{SYMBOL} Candlestick + MA{MA_FAST}/MA{MA_SLOW}",
        f"Volume + VMA{VOL_MA}",
        f"MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})",
    ),
)

# --- Row 1: Candlestick
fig.add_trace(
    go.Candlestick(
        x=x,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=SYMBOL,
        increasing_line_color="green",
        decreasing_line_color="red",
        increasing_fillcolor="green",
        decreasing_fillcolor="red",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Open: %{open:.2f}<br>"
            "High: %{high:.2f}<br>"
            "Low: %{low:.2f}<br>"
            "Close: %{close:.2f}<br>"
            "<extra></extra>"
        ),
    ),
    row=1, col=1
)

# MA lines (Row 1)
fig.add_trace(
    go.Scatter(
        x=x, y=df[f"MA{MA_FAST}"],
        mode="lines",
        name=f"MA{MA_FAST}",
        hovertemplate="<b>%{x}</b><br>MA: %{y:.2f}<extra></extra>",
    ),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(
        x=x, y=df[f"MA{MA_SLOW}"],
        mode="lines",
        name=f"MA{MA_SLOW}",
        hovertemplate="<b>%{x}</b><br>MA: %{y:.2f}<extra></extra>",
    ),
    row=1, col=1
)

# --- Row 2: Volume + Volume MA
fig.add_trace(
    go.Bar(
        x=x,
        y=df["volume"],
        name="Volume",
        marker_color=vol_colors,
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>",
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=df[f"VMA{VOL_MA}"],
        mode="lines",
        name=f"VMA{VOL_MA}",
        hovertemplate="<b>%{x}</b><br>VMA: %{y:,.0f}<extra></extra>",
    ),
    row=2, col=1
)

# --- Row 3: MACD (MACD line + SIGNAL purple + histogram)
fig.add_trace(
    go.Bar(
        x=x,
        y=df["HIST"],
        name="HIST",
        hovertemplate="<b>%{x}</b><br>HIST: %{y:.4f}<extra></extra>",
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=x,
        y=df["MACD"],
        mode="lines",
        name="MACD",
        hovertemplate="<b>%{x}</b><br>MACD: %{y:.4f}<extra></extra>",
    ),
    row=3, col=1
)

# ✅ 紫色信号线
fig.add_trace(
    go.Scatter(
        x=x,
        y=df["SIGNAL"],
        mode="lines",
        name="SIGNAL",
        line=dict(color="purple"),
        hovertemplate="<b>%{x}</b><br>SIGNAL: %{y:.4f}<extra></extra>",
    ),
    row=3, col=1
)

# Layout: IMPORTANT - make x-axis categorical => no weekend/holiday gaps
fig.update_layout(
    title=f"{SYMBOL} (No gaps) | {START_DATE} ~ {END_DATE}",
    hovermode="x unified",
    height=950,
)

fig.update_xaxes(type="category")  # 🔥 去掉缺失日期空档（周末/节假日）
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)

fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="MACD", row=3, col=1)

# Write HTML and open (no localhost)
OUT_DIR = Path("reports/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = (OUT_DIR / f"{SYMBOL}_candles_MA_MACD_{START_DATE}_{END_DATE}.html").resolve()

fig.write_html(str(out_file), include_plotlyjs="cdn", full_html=True)
print(f"[OK] Chart saved: {out_file}")
os.startfile(str(out_file))
