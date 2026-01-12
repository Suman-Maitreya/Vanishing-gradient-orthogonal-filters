import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, csv

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

save_dir = "Results"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transform),
    batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=False, transform=transform),
    batch_size=128, shuffle=False)

# -------------------------
# Model
# -------------------------
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [784] + [128]*15 + [10]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = DeepMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Training
# -------------------------
loss_log, acc_log = [], []

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.view(x.size(0), -1).to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss_log.append(total_loss)

    # Test accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(test_loader.dataset)
    acc_log.append(acc)
    print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Acc={acc:.4f}")

# -------------------------
# Save CSV
# -------------------------
with open(os.path.join(save_dir, "loss.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["loss"]] + [[l] for l in loss_log])

with open(os.path.join(save_dir, "accuracy.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["accuracy"]] + [[a] for a in acc_log])
