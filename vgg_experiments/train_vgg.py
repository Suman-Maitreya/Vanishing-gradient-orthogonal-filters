import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import csv

# --- 1. Setup & Environment ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'Results')
os.makedirs(results_dir, exist_ok=True)

# --- 2. Dataset (Bypassing Internet) ---
# We use batch_size=64 because VGG-16 takes up a lot of GPU memory
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Local MNIST Dataset from Phase 1...")
data_path = os.path.abspath(os.path.join(script_dir, '..', 'pytorch_experiments', 'with_filters'))

train_data = datasets.MNIST(root=data_path, train=True, download=False, transform=transform)
test_data = datasets.MNIST(root=data_path, train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# --- 3. The Orthogonal Additive Filter (OAF) Head ---
class OAFHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 512)
        nn.init.orthogonal_(self.layer1.weight)
        self.relu1 = nn.ReLU()
        
        self.layer2 = nn.Linear(512, 256)
        nn.init.orthogonal_(self.layer2.weight)
        self.relu2 = nn.ReLU()
        
        self.layer3 = nn.Linear(256, 128)
        nn.init.orthogonal_(self.layer3.weight)
        self.relu3 = nn.ReLU()
        
        self.layer4 = nn.Linear(128, num_classes)
        nn.init.orthogonal_(self.layer4.weight)

    def forward(self, x):
        # VGG outputs a complex tensor, so we must flatten it first
        x = torch.flatten(x, 1)
        x1 = self.relu1(self.layer1(x))
        x2 = self.relu2(self.layer2(x1)) + x1[:, :256] 
        x3 = self.relu3(self.layer3(x2)) + x2[:, :128]
        return self.layer4(x3)

# --- 4. Build Hybrid Model (VGG-16 + OAF) ---
print("Building VGG-16 with OAF Head...")
vgg = models.vgg16(weights=None)

# VGG's standard classifier is massive and prone to vanishing gradients. We replace it!
in_features = vgg.classifier[0].in_features
vgg.classifier = OAFHead(in_features, 10)
model = vgg.to(device)

criterion = nn.CrossEntropyLoss()
# We use a slightly smaller Learning Rate because VGG can be unstable early on
optimizer = optim.Adam(model.parameters(), lr=0.0005) 

# --- 5. Training Loop ---
epochs = 10
loss_log, acc_log = [], []

print("Starting GPU Training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    loss_log.append((epoch + 1, avg_loss))
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    acc_log.append((epoch + 1, accuracy))
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# --- 6. Save CSV Data ---
loss_file = os.path.join(results_dir, 'vgg_loss.csv')
acc_file = os.path.join(results_dir, 'vgg_accuracy.csv')

with open(loss_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Loss'])
    writer.writerows(loss_log)

with open(acc_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Accuracy'])
    writer.writerows(acc_log)

print(f"Training Complete! CSVs saved to {results_dir}")