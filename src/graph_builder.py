import torch
from torch_geometric.data import Data
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")


def load_prices():
    dfs = []

    for file in DATA_DIR.glob("*.csv"):

        df = pd.read_csv(file, skiprows=[1])

        ticker = file.stem
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

        df = df[["Date", "Close"]]
        df = df.rename(columns={"Close": ticker})

        dfs.append(df)

    prices = dfs[0]

    for df in dfs[1:]:
        prices = prices.merge(df, on="Date")

    prices = prices.sort_values("Date")
    prices = prices.set_index("Date")

    return prices


def compute_returns(prices):
    return prices.pct_change().dropna()


def build_edges(returns, corr_threshold=0.5):

    corr = returns.corr()

    edges = []

    n = corr.shape[0]

    for i in range(n):
        for j in range(n):

            if i != j and abs(corr.iloc[i, j]) > corr_threshold:
                edges.append([i, j])

    edge_index = torch.tensor(edges).t().contiguous()

    return edge_index


def build_dataset(window=30):

    prices = load_prices()
    returns = compute_returns(prices)

    edge_index = build_edges(returns)

    dataset = []

    for t in range(window, len(returns) - 1):

        window_data = returns.iloc[t-window:t]

        features = []

        for stock in window_data.columns:

            s = window_data[stock]

            features.append([
                s.mean(),
                s.std(),
                s.min(),
                s.max()
            ])

        x = torch.tensor(features, dtype=torch.float)

        next_day = returns.iloc[t + 1]

        y = torch.tensor((next_day > 0).astype(int).values)

        data = Data(x=x, edge_index=edge_index, y=y)

        dataset.append(data)

    return dataset