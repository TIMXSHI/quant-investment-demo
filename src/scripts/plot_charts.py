from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data/raw/polygon/1day")
SYMBOLS = ["AAPL", "TSLA", "INTC"]

START_DATE = "2025-01-01"
END_DATE   = "2025-01-08"

# -------------------------
# Load & normalize
# -------------------------
norm_df = pd.DataFrame()

for sym in SYMBOLS:
    path = DATA_DIR / f"{sym}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    df = df.loc[START_DATE:END_DATE]

    # ⭐ 核心：归一化到 1
    norm_df[sym] = df["close"] / df["close"].iloc[0]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 5))

for sym in SYMBOLS:
    plt.plot(norm_df.index, norm_df[sym], marker="o", label=sym)

plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
plt.title("Normalized Stock Prices (Start = 1)")
plt.xlabel("Date")
plt.ylabel("Indexed Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
