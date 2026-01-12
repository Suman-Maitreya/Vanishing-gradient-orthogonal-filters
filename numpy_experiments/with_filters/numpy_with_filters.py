import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# -------------------------
# Create Result Folder
# -------------------------
save_dir = "Results/Numpy_With_Filter"
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
    Q, _ = np.linalg.qr(A)   # Q is orthogonal
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
# Forward Pass (WITH FILTER)
# -------------------------

def forward_ortho(x, W, b, Q):
    h = [x]
    z = []
    for i in range(len(W)):
        zi = W[i] @ h[-1] + b[i]
        hi = relu(zi) + Q[i] @ h[-1]   # orthogonal additive path
        z.append(zi)
        h.append(hi)
    return h, z

# -------------------------
# Backward Pass (WITH FILTER)
# -------------------------

def backward_ortho(h, z, W, Q, y):
    grads = []
    delta = (h[-1] - y)

    for i in reversed(range(len(W))):
        dz = delta * relu_deriv(z[i])
        dW = dz @ h[i].T
        grads.append(np.linalg.norm(dW))

        # gradient flows through both paths
        delta = W[i].T @ dz + Q[i].T @ delta

    return grads[::-1]

# -------------------------
# Run Experiment
# -------------------------

h, z = forward_ortho(x, W, b, Q)
grads = backward_ortho(h, z, W, Q, y)

# -------------------------
# Plot & Save Results
# -------------------------

plt.figure(figsize=(8,5))
plt.plot(grads, marker='o', label="Orthogonal Additive MLP")
plt.xlabel("Layer Index")
plt.ylabel("Gradient Norm")
plt.title("Gradient Flow with Orthogonal Filters (NumPy)")
plt.legend()
plt.grid(True)

save_path = os.path.join(save_dir, "gradient_with_filter.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

# -------------------------
# Save CSV Results
# -------------------------

csv_path = os.path.join(save_dir, "gradients_with_filter.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Layer", "Gradient_Norm"])
    for i, g in enumerate(grads):
        writer.writerow([i, g])

print(f"Plot saved at: {save_path}")
print(f"CSV saved at: {csv_path}")
