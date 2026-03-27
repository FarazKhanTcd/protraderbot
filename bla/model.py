"""
model.py  –  GATPortfolioNet definition
========================================
Place this file in the same folder as portfolio_rebalancer.py.
The app will prompt you to load it before loading the .pth checkpoint.

Constants below must match the values used during training.
Adjust them if you used different hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# ── Hyperparameter constants (match your training config) ─────────────────────
MEMORY_LEN    = 12      # how many recent (weights, returns) pairs to remember
FEATURES_PER_BAR = 20  # number of features per time-step per stock
D_MODEL       = 128     # internal embedding dimension
N_LAYERS      = 2       # GRU layers in StockTemporalEncoder
N_HEADS_GAT   = 4       # attention heads in each GATConv layer
GAT_LAYERS    = 2       # number of stacked GATConv layers
DROPOUT       = 0.1     # dropout probability


# ─────────────────────────────────────────────────────────────────────────────

class PortfolioMemory(nn.Module):
    """GRU over recent (weights, returns) pairs -> regret embedding."""
    def __init__(self, n_stocks, d_model, memory_len=MEMORY_LEN):
        super().__init__()
        self.memory_len = memory_len
        self.input_proj = nn.Linear(2 * n_stocks, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model,
                          num_layers=1, batch_first=True)
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
    """Bidirectional GRU with learned attention pooling."""
    def __init__(self, n_features=FEATURES_PER_BAR, d_model=D_MODEL,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.input_proj(x)
        out, _ = self.gru(h)
        attn_w = torch.softmax(self.attn(out), dim=1)
        pooled = (out * attn_w).sum(dim=1)
        return self.norm(F.gelu(self.out(pooled)))


def project_weights(weights, max_w=0.25, min_w=0.02):
    """Iterative clamp-and-renorm to enforce per-stock weight bounds."""
    for _ in range(10):
        weights = weights.clamp(min=min_w, max=max_w)
        weights = weights / weights.sum()
        if weights.max() <= max_w + 1e-5 and weights.min() >= min_w - 1e-5:
            break
    return weights


class GATPortfolioNet(nn.Module):
    def __init__(self, n_stocks, n_features=FEATURES_PER_BAR, d_model=D_MODEL,
                 n_layers=N_LAYERS, n_heads_gat=N_HEADS_GAT,
                 gat_layers=GAT_LAYERS, dropout=DROPOUT, memory_len=MEMORY_LEN):
        super().__init__()
        self.n_stocks = n_stocks
        self.temporal_encoder = StockTemporalEncoder(n_features, d_model,
                                                     n_layers, dropout)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.gat_convs = nn.ModuleList()
        in_dim = d_model
        for _ in range(gat_layers):
            self.gat_convs.append(
                GATConv(in_dim, d_model // n_heads_gat,
                        heads=n_heads_gat, dropout=dropout, concat=True))
            in_dim = d_model
        self.gat_norm = nn.LayerNorm(d_model)
        self.memory = PortfolioMemory(n_stocks, d_model, memory_len)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1))

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
        regret = self.memory(history, device=device).unsqueeze(0).expand(
            self.n_stocks, -1)
        scores = self.head(torch.cat([h, regret], dim=-1)).squeeze(-1)
        if torch.isnan(scores).any():
            return torch.ones(self.n_stocks, device=x.device) / self.n_stocks
        weights = F.softmax(scores / self.temperature.clamp(min=0.1), dim=0)
        return project_weights(weights)
