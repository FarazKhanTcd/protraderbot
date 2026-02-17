import yfinance as yf
import pandas as pd
from pathlib import Path

# -------- CONFIG --------
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
START_DATE = "2002-01-01"
END_DATE = "2026-01-01"
# ------------------------

output_dir = Path("data/raw-stocks")
output_dir.mkdir(parents=True, exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    df.reset_index(inplace=True)
    df["Ticker"] = ticker

    file_path = output_dir / f"{ticker}.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path}")

print("All downloads completed.")
