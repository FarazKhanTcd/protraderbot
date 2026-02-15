import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from transformer_single_stock import SingleStockTransformer


# -----------------------------
# Dataset
# -----------------------------
class StockDataset(Dataset):
    def __init__(self, csv_path, T=60):
        self.T = T

        df = pd.read_csv(csv_path)

        # Use closing price
        prices = df["Close"].values

        # Compute log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        # Remove first row due to return calculation
        df = df.iloc[1:].copy()
        df["log_return"] = log_returns

        # Features (you can add more later)
        features = df[["log_return"]].values

        # Normalize
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)

        self.X = []
        self.y = []

        for i in range(T, len(features) - 1):
            self.X.append(features[i-T:i])
            self.y.append(log_returns[i])

        self.X = torch.FloatTensor(np.array(self.X))
        self.y = torch.FloatTensor(np.array(self.y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Training
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "../data/raw-stocks/AAPL.csv"

    dataset = StockDataset(csv_path, T=60)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SingleStockTransformer(F=1, T=60).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")


if __name__ == "__main__":
    train()
