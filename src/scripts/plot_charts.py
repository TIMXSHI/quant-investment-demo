from pathlib import Path
import os

from quantdemo.research.features import load_symbol_daily_range
from quantdemo.research.indicators import add_indicators
from quantdemo.research.plots import make_candles_volume_macd_figure

symbol = "AAPL"
start_date = "2025-01-01"
end_date = "2026-01-16"

df = load_symbol_daily_range(symbol, start_date, end_date, data_dir=Path("data/raw/polygon/1day"))
df = add_indicators(df)
fig = make_candles_volume_macd_figure(df, symbol, start_date, end_date)

out = Path("reports/charts") / f"{symbol}_{start_date}_{end_date}.html"
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
os.startfile(str(out.resolve()))