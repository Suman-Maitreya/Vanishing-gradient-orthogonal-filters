import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# -------------------------
# Create Result Folder
# -------------------------
save_dir = "Results/Numpy_No_filter"
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# Setup
# -------------------------
np.random.seed(42)

layers = 8
dim = 50

# -------------------------
# Functions
# -------------------------

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def orthogonal_matrix(dim):
    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    return Q

# -------------------------
# Initialize Parameters
# -------------------------

W = []
b = []
Q = []

for _ in range(layers):
    W.append(np.random.randn(dim, dim) * 0.1)
    b.append(np.zeros((dim, 1)))
    Q.append(orthogonal_matrix(dim))

# -------------------------
# Dummy Data
# -------------------------

x = np.random.randn(dim, 1)
y = np.random.randn(dim, 1)

# -------------------------
# Forward Passes
# -------------------------

def forward_normal(x, W, b):
    h = [x]
    z = []
    for i in range(len(W)):
        zi = W[i] @ h[-1] + b[i]
        hi = relu(zi)
        z.append(zi)
        h.append(hi)
    return h, z

def forward_ortho(x, W, b, Q):
    h = [x]
    z = []
    for i in range(len(W)):
        zi = W[i] @ h[-1] + b[i]
        hi = relu(zi) + Q[i] @ h[-1]
        z.append(zi)
        h.append(hi)
    return h, z

# -------------------------
# Backward Passes
# -------------------------

def backward_normal(h, z, W, y):
    grads = []
    delta = (h[-1] - y)
    for i in reversed(range(len(W))):
        dz = delta * relu_deriv(z[i])
        dW = dz @ h[i].T
        grads.append(np.linalg.norm(dW))
        delta = W[i].T @ dz
    return grads[::-1]

def backward_ortho(h, z, W, Q, y):
    grads = []
    delta = (h[-1] - y)
    for i in reversed(range(len(W))):
        dz = delta * relu_deriv(z[i])
        dW = dz @ h[i].T
        grads.append(np.linalg.norm(dW))
        delta = W[i].T @ dz + Q[i].T @ delta
    return grads[::-1]

# -------------------------
# Run Experiment
# -------------------------

h_n, z_n = forward_normal(x, W, b)
grads_n = backward_normal(h_n, z_n, W, y)

h_o, z_o = forward_ortho(x, W, b, Q)
grads_o = backward_ortho(h_o, z_o, W, Q, y)

# -------------------------
# Plot & Save Results
# -------------------------

plt.figure(figsize=(8,5))
plt.plot(grads_n, marker='o', label="Normal Deep MLP (No Filter)")
plt.plot(grads_o, marker='o', label="Orthogonal Additive MLP")
plt.xlabel("Layer Index")
plt.ylabel("Gradient Norm")
plt.title("Gradient Flow Comparison (NumPy)")
plt.legend()
plt.grid(True)

save_path = os.path.join(save_dir, "gradient_comparison.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# -------------------------
# Save CSV Results
# -------------------------

csv_path_normal = os.path.join(save_dir, "gradients_normal.csv")
csv_path_ortho = os.path.join(save_dir, "gradients_orthogonal.csv")

with open(csv_path_normal, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Gradient_Norm"])
    for i, g in enumerate(grads_n):
        writer.writerow([i, g])

with open(csv_path_ortho, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Gradient_Norm"])
    for i, g in enumerate(grads_o):
        writer.writerow([i, g])

print(f"Plot saved at: {save_path}")
print("CSV files saved at:")
print(csv_path_normal)
print(csv_path_ortho)
