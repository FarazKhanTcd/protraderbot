from pathlib import Path

import pandas as pd
import yfinance as yf


# -------- CONFIG --------
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
START_DATE = "2002-01-01"
END_DATE = "2026-01-01"
OUTPUT_DIR = Path("data/raw")
# ------------------------


def download_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print(f"No data found for {ticker}, skipping.")
            continue

        df.reset_index(inplace=True)
        df["Ticker"] = ticker

        file_path = output_dir / f"{ticker}.csv"
        df.to_csv(file_path, index=False)

        print(f"Saved {file_path}")

    print("All downloads completed.")


def main() -> None:
    download_stock_data(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()