import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data


DATA_DIR = Path("data/raw")


def load_prices():
    dfs = []

    for file in DATA_DIR.glob("*.csv"):
        df = pd.read_csv(file)

        ticker = file.stem
        df["Date"] = pd.to_datetime(df["Date"])
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
    returns = prices.pct_change().dropna()
    return returns


def build_graph(returns, corr_threshold=0.5):
    corr = returns.corr()

    num_nodes = corr.shape[0]
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):

            if i != j and abs(corr.iloc[i, j]) > corr_threshold:
                edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index


def build_features(returns, window=30):
    features = []

    last_window = returns.tail(window)

    for col in last_window.columns:
        series = last_window[col]

        features.append([
            series.mean(),
            series.std(),
            series.min(),
            series.max(),
        ])

    x = torch.tensor(features, dtype=torch.float)

    return x


def build_labels(returns):
    next_return = returns.iloc[-1]

    labels = (next_return > 0).astype(int)

    y = torch.tensor(labels.values, dtype=torch.long)

    return y


def build_dataset():
    prices = load_prices()
    returns = compute_returns(prices)

    x = build_features(returns)
    edge_index = build_graph(returns)
    y = build_labels(returns)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data