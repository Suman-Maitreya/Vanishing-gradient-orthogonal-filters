import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------
# Correct CSV Paths
# -------------------------
base_dir = "G:\\SEM  - 4\\Deep Learning\\DL\\Vanishing_gradient_Project\\numpy_experiments\\training_comparision\\Results"

loss_nf = np.loadtxt(
    os.path.join(base_dir, "loss_no_filter.csv"),
    delimiter=",",
    skiprows=1,
    usecols=1
)

loss_of = np.loadtxt(
    os.path.join(base_dir, "loss_with_filter.csv"),
    delimiter=",",
    skiprows=1,
    usecols=1
)

grad_nf = np.loadtxt(
    os.path.join(base_dir, "grad_no_filter.csv"),
    delimiter=",",
    skiprows=1,
    usecols=1
)

grad_of = np.loadtxt(
    os.path.join(base_dir, "grad_with_filter.csv"),
    delimiter=",",
    skiprows=1,
    usecols=1
)

# -------------------------
# Final Loss Comparison
# -------------------------
plt.figure()
plt.plot(loss_nf, label="Baseline (No Filter)")
plt.plot(loss_of, label="Orthogonal Additive")
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("Final Loss Comparison")
plt.legend()
plt.grid()
plt.savefig("plots/final_loss_comparison.png", dpi=300)
plt.show()

# -------------------------
# Final Gradient Comparison
# -------------------------
plt.figure()
plt.plot(grad_nf, label="Baseline (No Filter)")
plt.plot(grad_of, label="Orthogonal Additive")
plt.xlabel("Training Iterations")
plt.ylabel("Average Gradient Norm")
plt.title("Final Gradient Stability Comparison")
plt.legend()
plt.grid()
plt.savefig("plots/final_gradient_comparison.png", dpi=300)
plt.show()

print("✅ Final comparison plots fixed and regenerated.")
