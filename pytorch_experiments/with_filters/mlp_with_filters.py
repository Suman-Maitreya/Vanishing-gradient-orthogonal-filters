import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os, csv

# -------------------------
# Setup
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

save_dir = "G:\\SEM  - 4\\Deep Learning\\DL\\Vanishing_gradient_Project\\pytorch_experiments\\with_filters\\Results"
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
# Orthogonal Additive Layer
# -------------------------
class OrthoAddLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        Q = torch.empty(out_dim, in_dim)
        nn.init.orthogonal_(Q)
        self.register_buffer("Q", Q)

    def forward(self, x):
        return torch.relu(self.linear(x)) + torch.matmul(x, self.Q.t())

# -------------------------
# Model
# -------------------------
class DeepOrthoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [784] + [128]*15

        for i in range(len(dims)-1):
            layers.append(OrthoAddLayer(dims[i], dims[i+1]))

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.hidden(x)
        return self.output(x)

model = DeepOrthoMLP().to(device)
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
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    loss_log.append(total_loss)

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()

    acc = correct / len(test_loader.dataset)
    acc_log.append(acc)

    print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Accuracy={acc:.4f}")

# -------------------------
# Save CSV
# -------------------------
with open(os.path.join(save_dir, "loss.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["loss"]] + [[l] for l in loss_log])

with open(os.path.join(save_dir, "accuracy.csv"), "w", newline="") as f:
    csv.writer(f).writerows([["accuracy"]] + [[a] for a in acc_log])
