import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from transformer_single_stock import SingleStockTransformer


# -----------------------------
# Dataset
# -----------------------------
class StockDataset(Dataset):
    def __init__(self, csv_path, T=60):
        self.T = T

        df = pd.read_csv(csv_path)

        # Be robust to column name variations
        close_col = None
        for c in ["Close", "close", "Adj Close", "AdjClose", "adj_close", "adjclose"]:
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            raise ValueError(f"Could not find a Close/Adj Close column in: {list(df.columns)}")

        # Force numeric + clean
        close = pd.to_numeric(df[close_col], errors="coerce")
        df = df.copy()
        df["close_clean"] = close

        # Drop NaNs and non-positive prices (log requires > 0)
        df = df.dropna(subset=["close_clean"])
        df = df[df["close_clean"] > 0].reset_index(drop=True)

        prices = df["close_clean"].to_numpy(dtype=np.float64)

        # Need at least T+2 points to make one sample
        if len(prices) < (T + 2):
            raise ValueError(f"Not enough rows after cleaning: {len(prices)} rows, need at least {T+2}")

        # Compute log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        # Align df with returns (drop first row)
        df = df.iloc[1:].reset_index(drop=True)
        df["log_return"] = log_returns

        # Features (start simple)
        features = df[["log_return"]].to_numpy(dtype=np.float64)

        # Normalize features
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)

        # Build windows
        X_list, y_list = [], []
        # y is the next-day log_return at time i (using window ending at i-1)
        for i in range(T, len(features)):
            X_list.append(features[i - T:i])   # [T, F]
            y_list.append(log_returns[i])      # predict log_return at day i (next day)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def directional_accuracy(y_pred, y_true):
    return (torch.sign(y_pred) == torch.sign(y_true)).float().mean().item()


# -----------------------------
# Training
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "../data/raw-stocks/AAPL.csv"
    T = 60

    dataset = StockDataset(csv_path, T=T)

    # Time-based split
    n = len(dataset)
    n_train = int(0.8 * n)

    X_train = dataset.X[:n_train]
    y_train = dataset.y[:n_train]
    X_val = dataset.X[n_train:]
    y_val = dataset.y[n_train:]

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=256, shuffle=False)

    model = SingleStockTransformer(F=1, T=T).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.HuberLoss()

    epochs = 20
    best_val = float("inf")

    train_losses = []
    val_losses = []

    best_preds = None
    best_true = None

    for epoch in range(epochs):
        # ---- train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- validate
        model.eval()
        val_loss = 0.0
        preds_all, y_all = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)

                pred = model(Xb)
                loss = criterion(pred, yb)

                val_loss += loss.item()
                preds_all.append(pred.cpu())
                y_all.append(yb.cpu())

        val_loss /= len(val_loader)
        preds_all = torch.cat(preds_all)
        y_all = torch.cat(y_all)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

        # ---- save best
        if val_loss < best_val:
            best_val = val_loss
            best_preds = preds_all.clone()
            best_true = y_all.clone()

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scaler_mean": dataset.scaler.mean_,
                    "scaler_scale": dataset.scaler.scale_,
                    "T": T
                },
                "best_single_stock_transformer.pt"
            )
            print("  saved: best_single_stock_transformer.pt")

    # -----------------------------
    # Plot 1: Loss curves
    # -----------------------------
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    # -----------------------------
    # Plot 2: Predictions vs True (returns)
    # -----------------------------
    plt.figure()
    plt.plot(best_true.numpy(), label="true next-day log return")
    plt.plot(best_preds.numpy(), label="pred next-day log return")
    plt.xlabel("validation time index")
    plt.ylabel("log return")
    plt.legend()
    plt.title("Validation: Predicted vs True Next-Day Returns")
    plt.show()



if __name__ == "__main__":
    train()
