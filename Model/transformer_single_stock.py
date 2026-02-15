import torch
import torch.nn as nn
import numpy as np


class SingleStockTransformer(nn.Module):
    """
    Transformer model for single-stock next-day return prediction.

    Input shape:  [batch_size, T, F]
    Output shape: [batch_size]
    """

    def __init__(
        self,
        F,                  # number of features per day
        T=60,               # lookback window
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    ):
        super().__init__()

        self.T = T
        self.d_model = d_model

        # Project features to model dimension
        self.input_projection = nn.Linear(F, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Final regression head (predict next-day log return)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, T, F]
        """

        # 1. Project input features
        x = self.input_projection(x)  # [B, T, d_model]

        # 2. Add positional encoding
        x = x + self.positional_encoding(x)

        # 3. Transformer
        x = self.transformer(x)  # [B, T, d_model]

        # 4. Pool over time (mean pooling)
        x = x.mean(dim=1)  # [B, d_model]

        # 5. Predict next-day return
        out = self.regressor(x)  # [B, 1]

        return out.squeeze(-1)

    def positional_encoding(self, x):
        """
        Sinusoidal positional encoding.
        """
        B, T, D = x.size()

        pe = torch.zeros(T, D, device=x.device)
        position = torch.arange(0, T, device=x.device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, D, 2, device=x.device) *
            (-np.log(10000.0) / D)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, T, D]


# -----------------------------
# Example usage (sanity check)
# -----------------------------
if __name__ == "__main__":
    batch_size = 16
    T = 60
    F = 10  # number of features per day

    model = SingleStockTransformer(F=F, T=T)

    dummy_input = torch.randn(batch_size, T, F)

    output = model(dummy_input)

    print("Output shape:", output.shape)  # should be [batch_size]
