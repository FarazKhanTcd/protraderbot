import math
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
SAVE_PATH = BASE_DIR / "gat_portfolio_weights50.pth"

# ── Universe: 50 diversified stocks ──
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

# ── Rotational training settings ──
N_STOCKS_PER_ROUND = 10
N_ROUNDS = 5
EPOCHS_PER_ROUND = 50

# ── Data settings ──
LOOKBACK = 52
REBALANCE_CANDLES = 1
FEATURES_PER_BAR = 6

# ── Model hyperparameters ──
D_MODEL = 64
N_LAYERS = 2
N_HEADS_GAT = 4
GAT_LAYERS = 2
DROPOUT = 0.1
MEMORY_LEN = 8

# ── Training ──
LR = 3e-4
WEIGHT_DECAY = 5e-3
SHARPE_WINDOW = 12
HHI_LAMBDA = 0.3

# ── Split ──
TRAIN_FRAC = 0.80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_daily_data(data_dir, tickers):
    frames = {}
    for ticker in tickers:
        path = Path(data_dir) / f"{ticker}.csv"
        df = pd.read_csv(path, skiprows=3)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            frames[(col, ticker)] = df[col].astype(float)

    result = pd.DataFrame(frames)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result.sort_index()


def resample_to_weekly(daily_df, tickers):
    resampled = {}
    for ticker in tickers:
        ohlcv = daily_df.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
        rs = ohlcv.resample("W-FRI").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            resampled[(col, ticker)] = rs[col]

    result = pd.DataFrame(resampled)
    result.columns = pd.MultiIndex.from_tuples(result.columns)
    return result.sort_index()


def prepare_data():
    daily_df = load_daily_data(DATA_DIR, TICKERS)
    daily_df = daily_df.loc["2013-01-01":].ffill().dropna()  # META IPO'd 2012
    weekly_df = resample_to_weekly(daily_df, TICKERS).ffill().dropna()

    split_idx = int(len(weekly_df) * TRAIN_FRAC)
    train_df = weekly_df.iloc[:split_idx]
    test_df = weekly_df.iloc[split_idx:]

    return weekly_df, train_df, test_df, split_idx


def build_feature_tensor(df, tickers, end_idx, lookback):
    closes = df["Close"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    volumes = df["Volume"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    highs = df["High"][tickers].iloc[end_idx - lookback - 4 : end_idx]
    lows = df["Low"][tickers].iloc[end_idx - lookback - 4 : end_idx]

    assert (end_idx - lookback - 4) >= 0, f"end_idx={end_idx} too small"

    feat_list = []
    for ticker in tickers:
        c = closes[ticker].values.astype(np.float32)
        v = volumes[ticker].values.astype(np.float32)
        h = highs[ticker].values.astype(np.float32)
        l = lows[ticker].values.astype(np.float32)

        weekly_ret = np.diff(c) / (c[:-1] + 1e-8)
        mom_4w = (c[4:] - c[:-4]) / (c[:-4] + 1e-8)
        weekly_ret = weekly_ret[-lookback:]
        mom_4w = mom_4w[-lookback:]

        vol = np.array([weekly_ret[max(0, i - 3) : i + 1].std() for i in range(len(weekly_ret))])
        hl_ratio = ((h[1:] - l[1:]) / (c[1:] + 1e-8))[-lookback:]
        v_slice = v[1:][-lookback:]
        v_zscore = (v_slice - v_slice.mean()) / (v_slice.std() + 1e-8)
        c_tail = c[1:][-lookback:]
        roll_high = np.maximum.accumulate(c_tail)
        drawdown = (c_tail - roll_high) / (roll_high + 1e-8)

        feats = np.stack([weekly_ret, mom_4w, vol, hl_ratio, v_zscore, drawdown], axis=1)
        feat_list.append(feats)

    arr = np.stack(feat_list)
    mean = arr.mean(axis=0, keepdims=True)
    std = np.where(arr.std(axis=0, keepdims=True) < 1e-4, 1.0, arr.std(axis=0, keepdims=True))
    arr = np.clip(np.nan_to_num((arr - mean) / std), -3.0, 3.0)
    return torch.tensor(arr, dtype=torch.float32)


def build_correlation_graph(df, tickers, end_idx, lookback, top_k=3):
    returns = df["Close"][tickers].iloc[end_idx - lookback : end_idx].pct_change().dropna()
    corr = returns.corr().values.astype(np.float32)
    np.fill_diagonal(corr, 0)
    adj = np.zeros_like(corr)
    for i in range(len(tickers)):
        top_k_idx = np.argsort(np.abs(corr[i]))[-top_k:]
        adj[i, top_k_idx] = np.abs(corr[i, top_k_idx])
    return dense_to_sparse(torch.tensor(adj, dtype=torch.float32))


class PortfolioMemory(nn.Module):
    def __init__(self, n_stocks, d_model, memory_len=MEMORY_LEN):
        super().__init__()
        self.memory_len = memory_len
        self.input_proj = nn.Linear(2 * n_stocks, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history, device="cpu"):
        if not history:
            return torch.zeros(self.gru.hidden_size, device=device)
        recent = history[-self.memory_len:]
        seq = torch.stack([
            self.input_proj(torch.cat([w, r]).to(device))
            for w, r in recent
        ]).unsqueeze(0)
        _, h_n = self.gru(seq)
        return self.norm(h_n.squeeze())


class StockTemporalEncoder(nn.Module):
    def __init__(self, n_features=FEATURES_PER_BAR, d_model=D_MODEL, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.input_proj(x)
        out, _ = self.gru(h)
        attn_w = torch.softmax(self.attn(out), dim=1)
        pooled = (out * attn_w).sum(dim=1)
        return self.norm(F.gelu(self.out(pooled)))


def project_weights(weights, max_w=0.25, min_w=0.02):
    for _ in range(10):
        weights = weights.clamp(min=min_w, max=max_w)
        weights = weights / weights.sum()
        if weights.max() <= max_w + 1e-5 and weights.min() >= min_w - 1e-5:
            break
    return weights


class GATPortfolioNet(nn.Module):
    def __init__(
        self,
        n_stocks,
        n_features=FEATURES_PER_BAR,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads_gat=N_HEADS_GAT,
        gat_layers=GAT_LAYERS,
        dropout=DROPOUT,
        memory_len=MEMORY_LEN,
    ):
        super().__init__()
        self.n_stocks = n_stocks
        self.temporal_encoder = StockTemporalEncoder(n_features, d_model, n_layers, dropout)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.gat_convs = nn.ModuleList()
        in_dim = d_model
        for _ in range(gat_layers):
            self.gat_convs.append(
                GATConv(
                    in_dim,
                    d_model // n_heads_gat,
                    heads=n_heads_gat,
                    dropout=dropout,
                    concat=True,
                )
            )
            in_dim = d_model
        self.gat_norm = nn.LayerNorm(d_model)
        self.memory = PortfolioMemory(n_stocks, d_model, memory_len)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, edge_index, edge_attr, history, device="cpu"):
        node_emb = self.temporal_encoder(x)
        if torch.isnan(node_emb).any():
            return torch.ones(self.n_stocks, device=x.device) / self.n_stocks

        h = node_emb
        for conv in self.gat_convs:
            h = F.elu(conv(h, edge_index))
            if torch.isnan(h).any():
                return torch.ones(self.n_stocks, device=x.device) / self.n_stocks

        h = self.gat_norm(h + node_emb)
        regret = self.memory(history, device=device).unsqueeze(0).expand(self.n_stocks, -1)
        scores = self.head(torch.cat([h, regret], dim=-1)).squeeze(-1)
        if torch.isnan(scores).any():
            return torch.ones(self.n_stocks, device=x.device) / self.n_stocks

        weights = F.softmax(scores / self.temperature.clamp(min=0.1), dim=0)
        return project_weights(weights)


def window_loss(window_weights, hhi_lambda=HHI_LAMBDA):
    port_rets = torch.stack([(w * fr).sum() for w, fr in window_weights])
    mean_r = port_rets.mean()
    variance = ((port_rets - mean_r.detach()) ** 2).mean()
    sharpe = mean_r / torch.sqrt(variance + 1e-4)
    hhi = torch.stack([(w ** 2).sum() for w, _ in window_weights]).mean()
    return -sharpe + hhi_lambda * hhi


def compute_forward_returns(df, tickers, idx, horizon=REBALANCE_CANDLES):
    close = df["Close"][tickers]
    cur = close.iloc[idx].values
    fut = close.iloc[min(idx + horizon, len(close) - 1)].values
    return torch.tensor((fut - cur) / (cur + 1e-8), dtype=torch.float32)


def train_model_rotational(
    model,
    df,
    all_tickers,
    n_rounds=N_ROUNDS,
    n_stocks_per_round=N_STOCKS_PER_ROUND,
    epochs_per_round=EPOCHS_PER_ROUND,
    device=DEVICE,
):
    total_epochs = n_rounds * epochs_per_round
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_epochs)

    rng = random.Random(SEED)
    global_epoch = 0

    for rnd in range(1, n_rounds + 1):
        subset = rng.sample(all_tickers, n_stocks_per_round)
        steps = list(range(LOOKBACK + 4, len(df) - REBALANCE_CANDLES, REBALANCE_CANDLES))
        print(f"\n=== Round {rnd}/{n_rounds} | tickers: {subset} ===")
        print(f"    {len(steps)} walk-forward steps x {epochs_per_round} epochs")

        for epoch in range(1, epochs_per_round + 1):
            global_epoch += 1
            model.train()
            epoch_loss, n_updates = 0.0, 0
            window_weights, history = [], []

            for idx in steps:
                x = build_feature_tensor(df, subset, idx, LOOKBACK).to(device)
                ei, ea = build_correlation_graph(df, subset, idx, LOOKBACK)
                ei, ea = ei.to(device), ea.to(device)
                x = x + 0.005 * torch.randn_like(x)

                weights = model(x, ei, ea, history=history, device=device)
                if torch.isnan(weights).any():
                    continue

                fut_ret = compute_forward_returns(df, subset, idx).to(device)
                if torch.isnan(fut_ret).any():
                    continue

                history.append((weights.detach().cpu(), fut_ret.detach().cpu()))
                if len(history) > MEMORY_LEN:
                    history.pop(0)

                window_weights.append((weights, fut_ret))

                if len(window_weights) >= SHARPE_WINDOW:
                    loss = window_loss(window_weights)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        optim.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optim.step()
                        epoch_loss += loss.item()
                        n_updates += 1
                    window_weights = []

            sched.step()
            if epoch % 10 == 0 or epoch == 1:
                avg = epoch_loss / max(n_updates, 1)
                print(
                    f"  Epoch {epoch:3d}/{epochs_per_round} "
                    f"(global {global_epoch:3d}/{total_epochs}) | "
                    f"avg loss: {avg:.4f} | "
                    f"lr: {sched.get_last_lr()[0]:.2e} | "
                    f"updates: {n_updates}"
                )

    return model


def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    print(f"Device: {DEVICE}")
    print(
        f"Universe: {len(TICKERS)} tickers | "
        f"{N_ROUNDS} rounds × {N_STOCKS_PER_ROUND} stocks × {EPOCHS_PER_ROUND} epochs"
    )

    weekly_df, train_df, test_df, split_idx = prepare_data()
    print(
        f"Weekly bars: {len(weekly_df)}  |  "
        f"{weekly_df.index[0].date()} -> {weekly_df.index[-1].date()}"
    )
    print(f"Tickers: {TICKERS}")
    print(
        f"Train: {len(train_df)} bars  |  "
        f"{train_df.index[0].date()} -> {train_df.index[-1].date()}"
    )
    print(
        f"Test:  {len(test_df)} bars  |  "
        f"{test_df.index[0].date()} -> {test_df.index[-1].date()}"
    )

    test_feat = build_feature_tensor(weekly_df, TICKERS, LOOKBACK + 4, LOOKBACK)
    print(f"Feature tensor: {test_feat.shape}  |  range: [{test_feat.min():.2f}, {test_feat.max():.2f}]")

    model = GATPortfolioNet(n_stocks=N_STOCKS_PER_ROUND).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = train_model_rotational(model, train_df, TICKERS)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Weights saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
