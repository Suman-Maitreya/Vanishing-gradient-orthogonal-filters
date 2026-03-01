import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import csv

# --- 1. Setup & Environment ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")
os.makedirs('Results', exist_ok=True)

# --- 2. Dataset (Local MNIST to bypass internet errors) ---
# We resize to 224x224 and copy the grayscale to 3 channels to make ResNet happy
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Local MNIST Dataset...")
# Set download=False since you copied the folder. If you didn't copy it, change to True.
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

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
        x1 = self.relu1(self.layer1(x))
        x2 = self.relu2(self.layer2(x1)) + x1[:, :256] 
        x3 = self.relu3(self.layer3(x2)) + x2[:, :128]
        return self.layer4(x3)

# --- 4. Build Hybrid Model (ResNet-18 + OAF) ---
# Load standard ResNet-18, then replace its head with our custom OAF head
resnet = models.resnet18(weights=None) 
resnet.fc = OAFHead(resnet.fc.in_features, 10)
model = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. Training Loop ---
epochs = 10
loss_log, acc_log = [], []

print("Starting Training...")
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
    
    # Validation/Testing Pass
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
with open('Results/resnet_loss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Loss'])
    writer.writerows(loss_log)

with open('Results/resnet_accuracy.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Accuracy'])
    writer.writerows(acc_log)

print("Training Complete! CSVs saved to /Results.")