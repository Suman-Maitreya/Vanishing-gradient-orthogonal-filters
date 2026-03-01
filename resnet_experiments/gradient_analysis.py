import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure Results folder exists
os.makedirs('Results', exist_ok=True)

# --- 1. Standard Deep Head (The Problem) ---
class StandardHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        return self.layer4(x)

# --- 2. OAF Head (Your Solution) ---
class OAFHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 512)
        nn.init.orthogonal_(self.layer1.weight) # Orthogonal Matrix Q
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
        # Additive orthogonal pathway to protect gradients
        x1 = self.relu1(self.layer1(x))
        x2 = self.relu2(self.layer2(x1)) + x1[:, :256] 
        x3 = self.relu3(self.layer3(x2)) + x2[:, :128]
        return self.layer4(x3)

# --- 3. Run the Mathematical Analysis ---
def analyze_gradients():
    print("Running diagnostic gradient analysis...")
    in_features = 512 # Standard output size of ResNet-18
    num_classes = 10
    
    standard_head = StandardHead(in_features, num_classes)
    oaf_head = OAFHead(in_features, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    dummy_input = torch.randn(32, in_features)
    dummy_target = torch.randint(0, num_classes, (32,))
    
    # Trigger Backpropagation (Standard)
    loss_std = criterion(standard_head(dummy_input), dummy_target)
    loss_std.backward()
    
    # Trigger Backpropagation (OAF)
    loss_oaf = criterion(oaf_head(dummy_input), dummy_target)
    loss_oaf.backward()
    
    # Extract Gradient Magnitudes
    std_grads = [
        standard_head.layer4.weight.grad.abs().mean().item(),
        standard_head.layer3.weight.grad.abs().mean().item(),
        standard_head.layer2.weight.grad.abs().mean().item(),
        standard_head.layer1.weight.grad.abs().mean().item()
    ]
    
    oaf_grads = [
        oaf_head.layer4.weight.grad.abs().mean().item(),
        oaf_head.layer3.weight.grad.abs().mean().item(),
        oaf_head.layer2.weight.grad.abs().mean().item(),
        oaf_head.layer1.weight.grad.abs().mean().item()
    ]
    
    layers = ['Layer 4 (Output)', 'Layer 3', 'Layer 2', 'Layer 1 (Deepest)']
    
    # --- 4. Plot and Save the Proof ---
    plt.figure(figsize=(10, 6))
    plt.plot(layers, std_grads, marker='o', color='red', label='Standard Deep Head', linewidth=2)
    plt.plot(layers, oaf_grads, marker='s', color='green', label='OAF Head (Yours)', linewidth=2)
    
    plt.yscale('log')
    plt.title('Gradient Flow: Standard Head vs OAF Head')
    plt.xlabel('Backward Pass Direction (Output -> Input)')
    plt.ylabel('Average Gradient Magnitude (Log Scale)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join('Results', 'head_gradient_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"Success! Graph saved to {save_path}")

if __name__ == '__main__':
    analyze_gradients()