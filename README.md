# Preventing Vanishing Gradients Using Orthogonal Additive Filters

A research-oriented deep learning project that systematically analyzes and mitigates the vanishing gradient problem in deep neural networks using orthogonal additive filters, validated through mathematical intuition, NumPy simulations, and PyTorch experiments.

---

## Overview

Training very deep neural networks is challenging due to the vanishing gradient problem, where gradients shrink as they propagate backward through many layers. This prevents early layers from learning effectively.

This project proposes a simple architectural modification that improves gradient flow without increasing model complexity.

---

## Key Idea

Standard deep MLP layer:
h_i = ReLU(W_i h_{i-1})

Proposed orthogonal additive layer:
h_i = ReLU(W_i h_{i-1}) + Q h_{i-1}

Where Q is an orthogonal matrix that preserves vector magnitude.

---

## Methodology

The project is carried out in three stages:

1. NumPy-based gradient flow analysis  
2. NumPy-based training dynamics comparison  
3. PyTorch-based MNIST classification  

Each stage compares a normal deep network with a modified network using orthogonal additive filters.

---

## Experiments

### NumPy Experiments
- Gradient norm comparison across layers
- Loss and gradient tracking during training

### PyTorch Experiments
- Deep MLP trained on MNIST
- Comparison of convergence speed and accuracy

---

## Results Summary

| Model | Gradient Behavior | Convergence |
|------|------------------|------------|
| Normal Deep MLP | Vanishing gradients | Slow |
| Orthogonal Additive MLP | Stable gradients | Faster |

The orthogonal additive model shows improved training stability and faster convergence.

---

## Repository Structure

Vanishing_gradient_Project  
├── numpy_experiments  
├── pytorch_experiments  
├── plots  
└── README.md  

---

## How to Run

### NumPy
python numpy_experiments/no_filters/numpy_no_filter_baseline.py  
python numpy_experiments/with_filters/numpy_with_filters.py  
python numpy_experiments/training_comparison/numpy_training_comparison.py  

### PyTorch
python pytorch_experiments/no_filters/mlp_no_filter.py  
python pytorch_experiments/with_filters/mlp_with_filters.py  

---

## Technologies Used

Python  
NumPy  
Matplotlib  
PyTorch  

---

## Conclusion

Orthogonal additive filters provide an effective and simple solution to the vanishing gradient problem. By preserving gradient magnitude across layers, deep networks train faster and more reliably without complex architectural changes.

---

## Future Work

Extension to CNNs and Transformers  
Learnable orthogonal constraints  
Application to long-sequence modeling
