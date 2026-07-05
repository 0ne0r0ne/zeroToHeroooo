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

## 5. Makemore Part 5: Building a WaveNet — `05_makemore_part5/wavenet.ipynb`

An implementation of a WaveNet-like hierarchical architecture, extending the character-level language model to handle much larger context lengths without performance bottlenecks.

- **Architecture**: Replaces the flat context window with a deep, hierarchical structure using Dilated Causal Convolutions (simulated via reshaping tricks).
- **Hierarchical Processing**: Uses `FlattenConsecutive` layer to fuse character embeddings in pairs (e.g., 8 -> 4 -> 2 -> 1) through multiple layers, allowing the network to build abstract representations of long sequences efficiently.
- **Dimensionality**: Masters 3-dimensional tensor operations `(Batch, Time, Channels)` to maintain the spatial (time) relationship of characters across the network layers.
- **Modularity**: Organizes the complex architecture cleanly using a `Sequential` container class, closely mirroring the PyTorch `nn.Sequential` API.

**Key result**: Successfully scales the context window from 3 to 8 characters without exploding parameter counts or memory, setting the stage for even more advanced sequence models like Transformers.

---
## Based on

Andrej Karpathy — [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## 6. Let's Build GPT: from scratch — `06_lets_build_gpt/train.py`

A complete, decoder-only Transformer language model built from scratch, replicating the architecture of GPT.

- **Architecture**: Implements the full Transformer block including Token and Position Embeddings, Multi-Head Self-Attention, FeedForward networks, and Residual Connections.
- **Attention Mechanism**: Hand-coded self-attention mechanism showcasing the quadratic interaction between queries and keys, masked to enforce autoregressive behavior (preventing the model from 'seeing the future').
- **Optimization**: Incorporates LayerNorm and Dropout for training stability and regularization, ensuring robust optimization.
- **Dataset**: Trained on a dataset of 19,000 Turkish poems to demonstrate the model's ability to learn complex grammar, syntax, and stylistic structures strictly from character-level patterns.

**Key result**: Successfully builds the modern 'brain' behind Large Language Models, demonstrating how mathematical dot products and Softmax enable a neural network to 'attend' to relevant context and generate coherent text.

---