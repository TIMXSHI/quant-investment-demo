import pandas as pd
df = pd.read_parquet("data/features/polygon/1day/indicators_macd_daily.parquet", columns=["date"])
print(df["date"].min(), df["date"].max(), df["date"].nunique())