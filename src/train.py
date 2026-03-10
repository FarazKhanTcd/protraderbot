import torch
import torch.nn.functional as F
from model import StockGNN
from graph_builder import build_dataset


dataset = build_dataset()

train_size = int(len(dataset) * 0.8)

train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]


model = StockGNN(
    in_channels=4,
    hidden_channels=32,
    out_channels=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(1, 101):

    model.train()

    total_loss = 0

    for data in train_dataset:

        optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        loss = F.cross_entropy(out, data.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:

        model.eval()

        correct = 0
        total = 0

        for data in test_dataset:

            out = model(data.x, data.edge_index)

            pred = out.argmax(dim=1)

            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

        acc = correct / total

        print(f"Epoch {epoch} | Loss {total_loss:.2f} | Test Acc {acc:.3f}")