import torch
import torch.nn.functional as F
from model import StockGNN
from graph_builder import build_dataset


def train():

    data = build_dataset()

    model = StockGNN(
        in_channels=data.x.shape[1],
        hidden_channels=32,
        out_channels=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(1, 101):

        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out, data.y)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:

            pred = out.argmax(dim=1)
            acc = (pred == data.y).sum().item() / data.y.size(0)

            print(f"Epoch {epoch} | Loss {loss:.4f} | Acc {acc:.4f}")


if __name__ == "__main__":
    train()