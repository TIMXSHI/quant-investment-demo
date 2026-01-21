from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_candles_volume_macd_figure(
    df: pd.DataFrame,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ma_fast: int = 20,
    ma_slow: int = 50,
    vol_ma: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    no_gaps: bool = True,
    price_cols: tuple[str, str, str, str] = ("open", "high", "low", "close"),
    volume_col: str = "volume",
) -> go.Figure:
    """
    Plot:
      Row1: candlestick + MA
      Row2: volume bars + VMA
      Row3: MACD hist + MACD line + SIGNAL line (purple)

    Assumes df index is datetime-like and sorted.
    """
    if df is None or df.empty:
        raise ValueError("df is empty")

    o, h, l, c = price_cols
    for col in [o, h, l, c, volume_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # indicator column names
    ma_fast_col = f"MA{ma_fast}"
    ma_slow_col = f"MA{ma_slow}"
    vma_col = f"VMA{vol_ma}"
    macd_col = "MACD"
    sig_col = "SIGNAL"
    hist_col = "HIST"

    # category x-axis to remove gaps (weekends/holidays)
    x = df.index.strftime("%Y-%m-%d").tolist()

    # volume color up/down
    up = df[c] >= df[o]
    vol_colors = ["green" if u else "red" for u in up]

    title_range = ""
    if start_date and end_date:
        title_range = f" | {start_date} ~ {end_date}"

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            f"{symbol} Candlestick + MA{ma_fast}/MA{ma_slow}",
            f"Volume + VMA{vol_ma}",
            f"MACD({macd_fast},{macd_slow},{macd_signal})",
        ),
    )

    # --- Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df[o],
            high=df[h],
            low=df[l],
            close=df[c],
            name=symbol,
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

    # MA lines (optional if exists)
    if ma_fast_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[ma_fast_col],
                mode="lines",
                name=ma_fast_col,
                hovertemplate="<b>%{x}</b><br>MA: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

    if ma_slow_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df[ma_slow_col],
                mode="lines",
                name=ma_slow_col,
                hovertemplate="<b>%{x}</b><br>MA: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

    # --- Row 2: Volume + VMA
    fig.add_trace(
        go.Bar(
            x=x,
            y=df[volume_col],
            name="Volume",
            marker_color=vol_colors,
            hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>",
        ),
        row=2, col=1
    )

    if vma_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[vma_col],
                mode="lines",
                name=vma_col,
                hovertemplate="<b>%{x}</b><br>VMA: %{y:,.0f}<extra></extra>",
            ),
            row=2, col=1
        )

    # --- Row 3: MACD
    if hist_col in df.columns:
        fig.add_trace(
            go.Bar(
                x=x,
                y=df[hist_col],
                name="HIST",
                hovertemplate="<b>%{x}</b><br>HIST: %{y:.4f}<extra></extra>",
            ),
            row=3, col=1
        )

    if macd_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[macd_col],
                mode="lines",
                name="MACD",
                hovertemplate="<b>%{x}</b><br>MACD: %{y:.4f}<extra></extra>",
            ),
            row=3, col=1
        )

    if sig_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[sig_col],
                mode="lines",
                name="SIGNAL",
                line=dict(color="purple"),
                hovertemplate="<b>%{x}</b><br>SIGNAL: %{y:.4f}<extra></extra>",
            ),
            row=3, col=1
        )

    fig.update_layout(
        title=f"{symbol}{title_range}",
        hovermode="x unified",
        height=950,
    )

    if no_gaps:
        fig.update_xaxes(type="category")

    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", showline=True)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig
