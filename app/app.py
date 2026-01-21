from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# -------------------------
# Repo root detection (robust)
# -------------------------
def find_repo_root(current_file: Path) -> Path:
    """
    Try to locate repo root by searching upwards for 'src' folder.
    """
    p = current_file.resolve().parent
    for _ in range(6):
        if (p / "src").exists():
            return p
        p = p.parent
    # fallback: previous convention (parents[1])
    return current_file.resolve().parents[1]


FILE_PATH = Path(__file__)
ROOT_DIR = find_repo_root(FILE_PATH)
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from quantdemo.research.features import load_symbol_daily_range
from quantdemo.research.indicators import add_indicators
from quantdemo.research.plots import make_candles_volume_macd_figure


# -------------------------
# Helpers
# -------------------------
def last_business_day(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def iso(d: date) -> str:
    return d.isoformat()


def ensure_session_defaults():
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "1day"

    if "selected_symbol" not in st.session_state:
        st.session_state.selected_symbol = "AAPL"

    if "scan_date" not in st.session_state:
        st.session_state.scan_date = last_business_day(date.today() - timedelta(days=1))

    if "chart_start" not in st.session_state:
        st.session_state.chart_start = date(2025, 1, 1)

    if "chart_end" not in st.session_state:
        st.session_state.chart_end = last_business_day(date.today() - timedelta(days=1))

    if "auto_show_chart" not in st.session_state:
        st.session_state.auto_show_chart = False


# -------------------------
# Data loading (features parquet)
# -------------------------
@st.cache_data(show_spinner=False)
def load_macd_feature_table(feature_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(feature_file)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Ensure expected columns exist
    needed = {"symbol", "date", "close", "MACD", "SIGNAL", "golden_cross", "dead_cross"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Features file missing columns: {sorted(missing)}")
    return df


def build_home_tables_from_features(
    feat: pd.DataFrame, scan_date: date
) -> tuple[pd.DataFrame, pd.DataFrame, date | None]:
    """
    From precomputed feature table, return:
      - golden_df: symbol, latest_close, macd_diff
      - dead_df:   symbol, latest_close, macd_diff
      - latest_date_in_features

    Notes:
      - latest_close is the close at the latest date in the features table (same for both lists)
      - macd_diff = MACD - SIGNAL on scan_date (keeps sign, e.g., +0.12 or -0.08)
    """
    if feat.empty:
        empty = pd.DataFrame(columns=["symbol", "latest_close", "macd_diff"])
        return empty, empty, None

    latest_date = max(feat["date"])

    # Map: symbol -> latest_close (from latest_date)
    latest_day = feat[feat["date"] == latest_date][["symbol", "close"]].copy()
    latest_day = latest_day.rename(columns={"close": "latest_close"})

    # Slice scan date
    day_df = feat[feat["date"] == scan_date].copy()
    if day_df.empty:
        empty = pd.DataFrame(columns=["symbol", "latest_close", "macd_diff"])
        return empty, empty, latest_date

    # macd diff on scan date
    day_df["macd_diff"] = day_df["MACD"].astype(float) - day_df["SIGNAL"].astype(float)

    # Filter crosses
    golden = day_df[day_df["golden_cross"] == True][["symbol", "macd_diff"]].copy()
    dead = day_df[day_df["dead_cross"] == True][["symbol", "macd_diff"]].copy()

    # Join latest close
    golden = golden.merge(latest_day, on="symbol", how="left")
    dead = dead.merge(latest_day, on="symbol", how="left")

    # Reorder + sort
    golden = golden[["symbol", "latest_close", "macd_diff"]].sort_values("symbol").reset_index(drop=True)
    dead = dead[["symbol", "latest_close", "macd_diff"]].sort_values("symbol").reset_index(drop=True)

    return golden, dead, latest_date


def goto_search_with_symbol(sym: str, scan_date: date, latest_date: date | None):
    """
    Jump to Search page and prefill:
    - symbol
    - date window: scan_date - 1 year ... latest_date (from features)
    - auto show chart
    """
    st.session_state.selected_symbol = sym
    st.session_state.page = "Search Stock Price"

    start = scan_date - timedelta(days=365)
    end_ = latest_date if latest_date is not None else last_business_day(date.today() - timedelta(days=1))

    st.session_state.chart_start = start
    st.session_state.chart_end = end_

    st.session_state.auto_show_chart = True
    st.rerun()


# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="Quant Dashboard", layout="wide")
ensure_session_defaults()

st.title("📊 Quant Dashboard")
st.caption("Home 使用 **预计算 features parquet** 过滤信号；Search Stock Price 用 raw 数据出图。")


# -------------------------
# Sidebar: Navigation
# -------------------------
st.sidebar.title("📌 Functions")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Search Stock Price"],
    index=0 if st.session_state.page == "Home" else 1,
)
st.session_state.page = page

timeframe = st.sidebar.selectbox("Timeframe", ["1day"], index=0)
st.session_state.timeframe = timeframe

FEATURE_FILE = ROOT_DIR / "data" / "features" / "polygon" / timeframe / "indicators_macd_daily.parquet"
RAW_DIR = ROOT_DIR / "data" / "raw" / "polygon" / timeframe

st.sidebar.divider()
st.sidebar.caption("Home 不重新计算 MACD；只读 features parquet。")


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "Home":
    st.subheader("🏠 Home — MACD Cross Signals")

    if not FEATURE_FILE.exists():
        st.error(f"Features file not found: {FEATURE_FILE}")
        st.info("请先运行增量构建脚本生成 indicators_macd_daily.parquet。")
        st.stop()

    feat = load_macd_feature_table(FEATURE_FILE)
    latest_in_features = max(feat["date"]) if not feat.empty else None

    # Default scan date fallback (cannot exceed latest available in features)
    default_scan_date = st.session_state.scan_date
    if latest_in_features is not None and default_scan_date > latest_in_features:
        default_scan_date = latest_in_features
        st.session_state.scan_date = default_scan_date

    scan_date = st.date_input("Scan date (daily close)", value=default_scan_date)
    st.session_state.scan_date = scan_date

    golden_df, dead_df, latest_date = build_home_tables_from_features(feat, scan_date)

    if latest_date is None:
        st.warning("Features 表为空。")
        st.stop()

    st.caption(f"Features 最新交易日：**{latest_date.isoformat()}**（表中 latest_close 来自这个日期）")

    # ---- tables + counts
    left, right = st.columns(2)

    with left:
        st.markdown("### ✅ Golden Cross")
        st.caption(f"Count: **{len(golden_df):,}**")
        st.caption("列：symbol / latest_close / macd_diff (MACD - SIGNAL)。点击行跳转到 Search 并自动出图。")

        event = st.dataframe(
            golden_df,
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
        )
        if event and getattr(event, "selection", None) and event.selection.get("rows"):
            idx = event.selection["rows"][0]
            sym = str(golden_df.iloc[idx]["symbol"])
            goto_search_with_symbol(sym, scan_date, latest_date)

    with right:
        st.markdown("### ❌ Dead Cross")
        st.caption(f"Count: **{len(dead_df):,}**")
        st.caption("列：symbol / latest_close / macd_diff (MACD - SIGNAL)。点击行跳转到 Search 并自动出图。")

        event2 = st.dataframe(
            dead_df,
            width="stretch",
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
        )
        if event2 and getattr(event2, "selection", None) and event2.selection.get("rows"):
            idx = event2.selection["rows"][0]
            sym = str(dead_df.iloc[idx]["symbol"])
            goto_search_with_symbol(sym, scan_date, latest_date)


# =========================
# SEARCH STOCK PRICE PAGE
# =========================
else:
    st.subheader("🔎 Search Stock Price")

    # Sidebar inputs ONLY for this page
    st.sidebar.header("✅ Check Stock Price")
    manual_symbol = st.sidebar.text_input("Ticker", value=st.session_state.selected_symbol).strip().upper()
    manual_start = st.sidebar.date_input("Start", value=st.session_state.chart_start)
    manual_end = st.sidebar.date_input("End", value=st.session_state.chart_end)
    no_gaps = st.sidebar.checkbox("No gaps (remove weekends)", value=True)

    st.sidebar.subheader("Indicators")
    ma_fast = st.sidebar.number_input("MA fast", min_value=2, max_value=400, value=20, step=1)
    ma_slow = st.sidebar.number_input("MA slow", min_value=2, max_value=400, value=50, step=1)
    vol_ma = st.sidebar.number_input("Volume MA", min_value=2, max_value=400, value=20, step=1)

    macd_fast = st.sidebar.number_input("MACD fast", min_value=2, max_value=100, value=12, step=1)
    macd_slow = st.sidebar.number_input("MACD slow", min_value=2, max_value=200, value=26, step=1)
    macd_signal = st.sidebar.number_input("MACD signal", min_value=2, max_value=100, value=9, step=1)

    btn_show_chart = st.sidebar.button("Show chart", type="primary")

    # persist
    if manual_symbol:
        st.session_state.selected_symbol = manual_symbol
    st.session_state.chart_start = manual_start
    st.session_state.chart_end = manual_end

    # trigger logic
    trigger = False
    if btn_show_chart:
        trigger = True
        st.session_state.auto_show_chart = False  # manual overrides

    if st.session_state.auto_show_chart:
        trigger = True
        st.session_state.auto_show_chart = False  # consume once

    st.divider()
    st.subheader(f"📈 Chart — {st.session_state.selected_symbol}")

    if not RAW_DIR.exists():
        st.error(f"RAW_DIR not found: {RAW_DIR}")
        st.info("Search 页出图依赖 data/raw/polygon/... 目录存在并有对应 symbol 的分区数据。")
        st.stop()

    chart_start = st.session_state.chart_start
    chart_end = st.session_state.chart_end

    if chart_start > chart_end:
        st.error("Chart start date must be <= end date.")
        st.stop()

    if not trigger:
        st.info("在左侧点 **Show chart**，或从 Home 表格点击某一行会自动跳转并出图。")
        st.stop()

    symbol = st.session_state.selected_symbol
    start_date = iso(chart_start)
    end_date = iso(chart_end)

    try:
        with st.spinner("Loading data + calculating indicators..."):
            df = load_symbol_daily_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_dir=RAW_DIR,
                strict=True,
            )
            df = add_indicators(
                df,
                ma_fast=int(ma_fast),
                ma_slow=int(ma_slow),
                vol_ma=int(vol_ma),
                macd_fast=int(macd_fast),
                macd_slow=int(macd_slow),
                macd_signal=int(macd_signal),
            )
            fig = make_candles_volume_macd_figure(
                df=df,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                ma_fast=int(ma_fast),
                ma_slow=int(ma_slow),
                vol_ma=int(vol_ma),
                macd_fast=int(macd_fast),
                macd_slow=int(macd_slow),
                macd_signal=int(macd_signal),
                no_gaps=no_gaps,
            )

        st.plotly_chart(fig, width="stretch")

        with st.expander("Show data (tail)"):
            st.dataframe(df.tail(200), width="stretch")

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("通常表示该 symbol 在你本地 raw 目录没有对应日期分区数据。")
    except Exception as e:
        st.exception(e)
