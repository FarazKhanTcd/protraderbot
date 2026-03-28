import yfinance as yf

TICKERS = [
    # Tech / Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AVGO", "ORCL", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS",
    "BLK", "AXP", "V", "MA", "COF",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK",
    "PFE", "TMO", "ABT", "MDT", "CVS",
    # Consumer / Retail
    "WMT", "COST", "HD", "TGT", "MCD",
    "SBUX", "NKE", "PG", "KO", "PEP",
    # Energy / Industrials / Other
    "XOM", "CVX", "NEE", "LIN", "CAT",
    "HON", "RTX", "UPS", "GE", "SPG",
]

PERIOD   = "30y"
INTERVAL = "1d"

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    data = yf.download(ticker, period=PERIOD, interval=INTERVAL)
    filename = f"{ticker}.csv"
    data.to_csv(filename)
    print(f"Saved {filename}")

print(f"Done. {len(TICKERS)} files saved.")