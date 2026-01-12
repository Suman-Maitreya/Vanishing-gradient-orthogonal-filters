import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# =========================
# Folder Setup
# =========================

base_dir = "Results"
os.makedirs(base_dir, exist_ok=True)

# =========================
# Parameters
# =========================

np.random.seed(42)

layers = 8
dim = 50
lr = 0.001
steps = 300
clip = 1.0

# =========================
# Functions
# =========================

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def mse(y_pred, y):
    return np.mean((y_pred - y) ** 2)

def orthogonal_matrix(dim):
    A = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(A)
    return Q

# =========================
# Initialization
# =========================

def init_params():
    W = []
    b = []
    for _ in range(layers):
        W.append(np.random.randn(dim, dim) * 0.01)
        b.append(np.zeros((dim, 1)))
    return W, b

W_nf, b_nf = init_params()
W_of, b_of = init_params()
Q = [orthogonal_matrix(dim) for _ in range(layers)]

# =========================
# Dummy Data
# =========================

x = np.random.randn(dim, 1)
y = np.random.randn(dim, 1)

# =========================
# Forward Passes
# =========================

def forward_no_filter(x, W, b):
    h = [x]
    z = []
    for i in range(len(W)):
        zi = W[i] @ h[-1] + b[i]
        hi = relu(zi)
        h.append(hi)
        z.append(zi)
    return h, z

def forward_with_filter(x, W, b, Q):
    h = [x]
    z = []
    for i in range(len(W)):
        zi = W[i] @ h[-1] + b[i]
        hi = relu(zi) + Q[i] @ h[-1]
        h.append(hi)
        z.append(zi)
    return h, z

# =========================
# Training Loop
# =========================

loss_nf, grad_nf = [], []
loss_of, grad_of = [], []

for step in range(steps):

    # ---- NO FILTER ----
    h, z = forward_no_filter(x, W_nf, b_nf)
    loss_nf.append(mse(h[-1], y))

    delta = h[-1] - y
    grads = []

    for i in reversed(range(layers)):
        dz = delta * relu_deriv(z[i])
        dW = np.clip(dz @ h[i].T, -clip, clip)
        W_nf[i] -= lr * dW
        b_nf[i] -= lr * dz
        grads.append(np.linalg.norm(dW))
        delta = W_nf[i].T @ dz

    grad_nf.append(np.mean(grads))

    # ---- WITH ORTHOGONAL FILTER ----
    h, z = forward_with_filter(x, W_of, b_of, Q)
    loss_of.append(mse(h[-1], y))

    delta = h[-1] - y
    grads = []

    for i in reversed(range(layers)):
        dz = delta * relu_deriv(z[i])
        dW = np.clip(dz @ h[i].T, -clip, clip)
        W_of[i] -= lr * dW
        b_of[i] -= lr * dz
        grads.append(np.linalg.norm(dW))
        delta = W_of[i].T @ dz + Q[i].T @ delta

    grad_of.append(np.mean(grads))

# =========================
# Save CSV Files
# =========================

def save_csv(filename, data):
    with open(os.path.join(base_dir, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Value"])
        for i, v in enumerate(data):
            writer.writerow([i, v])

save_csv("loss_no_filter.csv", loss_nf)
save_csv("loss_with_filter.csv", loss_of)
save_csv("grad_no_filter.csv", grad_nf)
save_csv("grad_with_filter.csv", grad_of)

# =========================
# Plots
# =========================

plt.figure()
plt.plot(loss_nf, label="No Filter")
plt.plot(loss_of, label="With Orthogonal Filter")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Loss Comparison (NumPy)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_dir, "loss_comparison.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(grad_nf, label="No Filter")
plt.plot(grad_of, label="With Orthogonal Filter")
plt.xlabel("Training Steps")
plt.ylabel("Average Gradient Norm")
plt.title("Gradient Comparison (NumPy)")
plt.legend()
plt.grid()
plt.savefig(os.path.join(base_dir, "gradient_comparison.png"), dpi=300)
plt.show()

print("Training comparison completed. Results saved.")
