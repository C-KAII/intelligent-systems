# Intelligent Systems - Neural Network Classification

## Overview

This task involves implementing a neural network from scratch to classify images from the CIFAR-10 dataset. The neural network must be trained using stochastic gradient descent with specified hyperparameters.

## Objectives

- Implement a neural network with three layers.
- Train and evaluate using the CIFAR-10 dataset.
- Experiment with different learning rates and mini-batch sizes.

## Implementation Requirements

1. **Dataset**: CIFAR-10 (download from [CIFAR-10 dataset page](http://www.cs.toronto.edu/~kriz/cifar.html)).
2. **Neural Network Structure**: 
  - Input Layer: 3072 neurons.
  - Hidden Layer: 30 neurons.
  - Output Layer: 10 neurons.
3. **Training Parameters**:
  - Epochs: 20
  - Mini-batch sizes: 1, 5, 20, 100, 300
  - Learning rates: 0.001, 0.01, 1.0, 10, 100
4. **Loss Function**: Quadratic cost.
5. **Output**:
  - Accuracy per epoch.
  - Final weights after training.
  - Graphs for test accuracy vs. epochs and mini-batch size.

## Notes

- Ensure all hyperparameters are easy to adjust.
- Test with a small neural network for debugging.
- No external machine learning libraries allowed (e.g., sklearn, TensorFlow, PyTorch).
