import torch
import torch.nn as nn
import torch.optim as optim

# Simple feed-forward eval net: input is 12×64 bitboard planes flattened
class EvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(12*64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Dummy dataset: replace with real positions + scalar evaluations
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # list of (bitboard_tensor, eval_score)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- Training loop ---
model = EvalNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Assume `train_data` is preloaded list of (tensor([12,64]), float)
loader = torch.utils.data.DataLoader(ChessDataset(train_data), batch_size=64, shuffle=True)

for epoch in range(10):
    for boards, targets in loader:
        preds = model(boards)[:,0]
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} loss {loss.item():.4f}")

# Export to TorchScript for C++
scripted = torch.jit.script(model)
scripted.save("eval_net.pt")
