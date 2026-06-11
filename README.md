# Neural Networks: Zero to Hero — Personal Implementations

My implementations following Andrej Karpathy's "Neural Networks: Zero to Hero" series.
Built from scratch, without copy-pasting, focusing on deep understanding over speed.

---

## 1. Micrograd — `micrograd_selfcoding.py`

A tiny scalar-valued autograd engine built from scratch.

- **Engine**: `Value` class with forward and backward pass
- **Layers**: `Neuron`, `Layer`, `MLP` classes
- **Backpropagation**: Manual chain rule implementation
- **Activation**: Tanh

This replicates the core mechanism behind PyTorch's autograd in ~100 lines.

---

## 2. Makemore — `main.ipynb`

A character-level bigram language model that generates human-like names.

- **Statistical Model**: Counts bigram frequencies across 32,000 names,
  normalizes to probabilities, and samples new names.
- **Neural Network Model**: Single-layer neural network trained with
  gradient descent to learn the same bigram distribution.
- **Loss Function**: Negative Log Likelihood (NLL) used to evaluate
  and minimize prediction error.
- **Techniques**: One-hot encoding, softmax, Laplace smoothing,
  backpropagation via PyTorch autograd.

**Key result:** Both models converge to the same NLL (~2.45), proving that a neural
network can learn what a counting table knows — purely through gradient descent.

### Dataset
`names.txt` — 32,033 human first names from [ssa.gov](https://www.ssa.gov/oact/babynames/)

---

## 3. Makemore Part 2: MLP — `03_makemore_part2/mlp_scratch.ipynb`

A Multi-Layer Perceptron (MLP) character-level language model, based on the architecture described by Bengio et al. (2003). Built completely from scratch using fundamental PyTorch tensor operations.

- **Architecture**: Transitions from a simple bigram model to an N-gram model using a sliding context window (e.g., predicting the next character based on the previous 3 characters).
- **Embeddings**: Implements a learnable embedding layer (lookup table) that maps discrete characters into a continuous multi-dimensional vector space.
- **Layers**: Features a hidden layer with `tanh` non-linearity and an output layer for logit generation.
- **Training**: Introduces Minibatches and Stochastic Gradient Descent (SGD) to efficiently train the network on the full dataset without performance bottlenecks.
- **Implementation Details**: Focuses on manual tensor dimension management, broadcasting rules, and understanding the mathematical flow from inputs to the final cross-entropy loss.

**Key result**: Significantly improves the quality and structure of the generated human-like names compared to the basic bigram model, effectively capturing deeper character relationships through the hidden layer.

---

## 4. Makemore Part 3: Activations & Gradients, BatchNorm — `04_makemore_part3/makemore_part3_self.ipynb`

An in-depth exploration of the internal dynamics of deep neural networks, focusing on initialization, activations, gradients, and batch normalization.

- **Initialization & Scaling**: Properly initializes network weights (Kaiming initialization) and scales output layers to prevent extreme initial loss (hockey stick loss curve) and saturated `tanh` neurons (dead neurons).
- **Batch Normalization**: Implements a custom `BatchNorm1d` layer from scratch to stabilize hidden layer distributions, improving training stability and speed.
- **Diagnostics & Monitoring**: Develops rich diagnostic plots to track activation distributions (forward pass), gradient distributions (backward pass), and the update-to-data ratio across training steps.
- **Refactoring**: Re-structures the code using PyTorch-like layer classes (`Linear`, `BatchNorm1d`, `Tanh`) to modularize forward and backward passes.

**Key result**: Achieves a robust, stable, and easily expandable neural network architecture where training dynamics can be visualized, avoiding common pitfalls like exploding/vanishing gradients and dead neurons.

---

## Based on

Andrej Karpathy — [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
