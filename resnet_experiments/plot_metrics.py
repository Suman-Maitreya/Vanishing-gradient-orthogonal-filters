import matplotlib.pyplot as plt
import csv
import os

epochs, acc, loss = [], [], []

# Ensure we are looking in the right place
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'Results')

# Read Accuracy
with open(os.path.join(results_dir, 'resnet_accuracy.csv'), 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        epochs.append(int(row[0]))
        acc.append(float(row[1]))

# Read Loss
with open(os.path.join(results_dir, 'resnet_loss.csv'), 'r') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        loss.append(float(row[1]))

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot Loss on the left Y-axis
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Training Loss', color='tab:red', fontweight='bold')
ax1.plot(epochs, loss, color='tab:red', marker='o', linewidth=2, label='Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot Accuracy on the right Y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Test Accuracy (%)', color='tab:blue', fontweight='bold')
ax2.plot(epochs, acc, color='tab:blue', marker='s', linewidth=2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('ResNet-18 + OAF Head: Training Progress', fontsize=14, fontweight='bold')
fig.tight_layout()

# Save the plot
save_path = os.path.join(results_dir, 'resnet_training_metrics.png')
plt.savefig(save_path, dpi=300)
print(f"Success! Graph saved to {save_path}")
plt.show() # This will pop up the graph on your screen too!