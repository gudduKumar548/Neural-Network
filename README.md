# Neural Networks — From Zero to Code

A complete beginner's guide covering the intuition, the math, and the code.

---

## Table of Contents

1. [Why Do We Need Neural Networks?](#1-why-do-we-need-neural-networks)
2. [The Building Block — A Single Neuron](#2-the-building-block--a-single-neuron)
3. [The Math](#3-the-math)
4. [Activation Functions](#4-activation-functions)
5. [Network Architecture — Stacking Neurons into Layers](#5-network-architecture--stacking-neurons-into-layers)
6. [How the Network Learns](#6-how-the-network-learns)
7. [Coding from Scratch (NumPy)](#7-coding-from-scratch-numpy)
8. [Using PyTorch (Real-World Way)](#8-using-pytorch-real-world-way)
9. [Summary Cheatsheet](#9-summary-cheatsheet)

---

## 1. Why Do We Need Neural Networks?

Think about how you recognise a dog in a photo. You don't follow a rulebook — your brain just *knows*. Traditional programming requires you to write those rules explicitly. But for complex problems like image recognition, language, or fraud detection, the rules are too many and too complex to write by hand.

| Approach | How it works |
|---|---|
| Traditional programming | You write rules → computer follows them |
| Machine learning | You give examples → computer learns the rules |
| Neural networks | A specific way to do ML, inspired by how brain neurons connect |

**We need neural networks for tasks like:**
- Image and speech recognition
- Language translation
- Recommendation systems (Netflix, YouTube)
- Medical diagnosis
- Anything where the pattern is too complex to hand-code

---

## 2. The Building Block — A Single Neuron

A single neuron does three things:

```
x₁ ──(w₁)──┐
x₂ ──(w₂)──┤──► [Σ + bias] ──► [Activation σ] ──► output
x₃ ──(w₃)──┘
```

1. Takes **inputs** (x₁, x₂, x₃ …)
2. Multiplies each by a **weight** (how important is this input?)
3. Sums them with a **bias**, then passes through an **activation function**

---

## 3. The Math

### Step 1 — Weighted sum (pre-activation)

```
z = w₁x₁ + w₂x₂ + w₃x₃ + b
```

In vector notation:

```
z = W · x + b
```

**Example with real numbers:**

| Symbol | Value |
|---|---|
| x₁, x₂, x₃ | 0.5, 0.8, 0.2 |
| w₁, w₂, w₃ | 0.4, 0.7, −0.3 |
| bias b | 0.1 |

```
z = (0.4 × 0.5) + (0.7 × 0.8) + (−0.3 × 0.2) + 0.1
  = 0.20 + 0.56 − 0.06 + 0.1
  = 0.80
```

### Step 2 — Activation function

Without an activation function, no matter how many layers you stack, everything collapses to a single linear equation — useless for complex patterns. Activation functions introduce **non-linearity**.

---

## 4. Activation Functions

### Sigmoid σ(z)

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

- Output range: **(0, 1)**
- Used in: output layer for **binary classification**
- Example: σ(0.8) ≈ 0.69

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

- Output range: **[0, ∞)**
- Most popular for **hidden layers** — simple and fast
- Example: ReLU(0.8) = 0.8, ReLU(−0.5) = 0

### Tanh

```
tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

- Output range: **(−1, 1)**
- Centred at 0, often better than Sigmoid for hidden layers

### When to use which?

| Layer | Recommended |
|---|---|
| Hidden layers | **ReLU** (default choice) |
| Output — binary classification | **Sigmoid** |
| Output — multi-class classification | **Softmax** |
| Output — regression | **None** (linear) |

---

## 5. Network Architecture — Stacking Neurons into Layers

```
Input Layer       Hidden Layer        Output Layer
  (3 neurons)      (4 neurons)         (2 neurons)

    x₁ ────────── h₁ ─────────── y₁
    x₂ ──────×──── h₂ ───×─────── y₂
    x₃ ────────── h₃ ─────────────
                   h₄

         ↑                ↑               ↑
    raw features     learned          final
                     patterns        prediction
```

- **Input layer** — raw data (pixels, numbers, words…)
- **Hidden layers** — learn intermediate patterns (can be many!)
- **Output layer** — final prediction

Every connection has a **weight**. Every neuron has a **bias**. These are the *learnable parameters* — what training adjusts.

> **Number of parameters** = (inputs × hidden) + hidden_biases + (hidden × outputs) + output_biases

---

## 6. How the Network Learns

Training repeats this 4-step loop:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│ Forward pass │ ──► │ Compute loss │ ──► │ Backpropagation  │ ──► │ Update weights│
│ (predict)    │     │ (how wrong?) │     │ (find gradients) │     │ (grad descent)│
└──────────────┘     └──────────────┘     └──────────────────┘     └───────┬───────┘
       ▲                                                                    │
       └────────────────────────────────────────────────────────────────────┘
                              repeat for every batch
```

### The Loss Function

Measures how wrong the network is. Lower = better.

**Mean Squared Error (regression):**
```
Loss = (1/n) × Σ (predicted − actual)²
```

**Binary Cross-Entropy (classification):**
```
Loss = −(1/n) × Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

### Gradient Descent

Nudges each weight in the direction that reduces the loss:

```
w = w − α × (∂Loss / ∂w)
```

Where **α (alpha)** is the **learning rate**:
- Too large → overshoot the minimum, training diverges
- Too small → training takes forever
- Typical values: 0.001 – 0.1

### Backpropagation

Backprop applies the **chain rule** from calculus *backwards* through the network — computing how much each weight contributed to the error.

```
∂Loss/∂W1 = ∂Loss/∂A2 × ∂A2/∂Z2 × ∂Z2/∂A1 × ∂A1/∂Z1 × ∂Z1/∂W1
```

You don't need to derive this by hand — PyTorch/TensorFlow handle it automatically via `loss.backward()`.

---

## 7. Coding from Scratch (NumPy)

A complete 2-input → 3-hidden → 1-output network, no libraries except NumPy.

### Setup

```python
import numpy as np

np.random.seed(42)

# W1: shape (3, 2)  — 3 hidden neurons, each takes 2 inputs
# W2: shape (1, 3)  — 1 output neuron, takes 3 hidden inputs
W1 = np.random.randn(3, 2) * 0.1   # small random weights
b1 = np.zeros((3, 1))

W2 = np.random.randn(1, 3) * 0.1
b2 = np.zeros((1, 1))
```

### Activation Functions

```python
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu_deriv(z):
    return (z > 0).astype(float)   # 1 if z > 0, else 0
```

### Forward Pass

```python
def forward(X):
    # X shape: (2, m) — 2 features, m examples

    # Hidden layer
    Z1 = W1 @ X + b1        # (3, m)
    A1 = relu(Z1)            # apply ReLU

    # Output layer
    Z2 = W2 @ A1 + b2        # (1, m)
    A2 = sigmoid(Z2)         # apply Sigmoid → output in (0, 1)

    cache = (Z1, A1, Z2, A2)
    return A2, cache
```

### Compute Loss

```python
def compute_loss(A2, Y):
    m = Y.shape[1]
    loss = -1/m * np.sum(
        Y * np.log(A2 + 1e-8) +
        (1 - Y) * np.log(1 - A2 + 1e-8)
    )
    return loss
```

### Backpropagation

```python
def backward(X, Y, cache):
    Z1, A1, Z2, A2 = cache
    m = X.shape[1]

    # Output layer gradients
    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden layer gradients (chain rule)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
```

### Training Loop

```python
def train(X, Y, learning_rate=0.1, epochs=1000):
    global W1, b1, W2, b2

    for epoch in range(epochs):
        # 1. Forward pass
        A2, cache = forward(X)

        # 2. Compute loss
        loss = compute_loss(A2, Y)

        # 3. Backward pass
        grads = backward(X, Y, cache)

        # 4. Gradient descent update
        W1 -= learning_rate * grads["dW1"]
        b1 -= learning_rate * grads["db1"]
        W2 -= learning_rate * grads["dW2"]
        b2 -= learning_rate * grads["db2"]

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# Training data: 4 samples, 2 features each
X = np.array([[0.5, 0.1, 0.9, 0.3],
              [0.8, 0.4, 0.2, 0.7]])
Y = np.array([[1, 0, 1, 0]])

train(X, Y, learning_rate=0.1, epochs=500)

# Final predictions
preds, _ = forward(X)
print("\nFinal predictions:", preds.round(2))
print("True labels:      ", Y)
```

**Expected output:**
```
Epoch    0 | Loss: 0.6931
Epoch  100 | Loss: 0.5812
Epoch  200 | Loss: 0.4703
Epoch  300 | Loss: 0.3891
Epoch  400 | Loss: 0.3214

Final predictions: [[0.82 0.18 0.79 0.21]]
True labels:       [[1 0 1 0]]
```

---

## 8. Using PyTorch (Real-World Way)

Once you understand the math, use a framework. PyTorch handles backprop automatically.

### Install

```bash
pip install torch
```

### Same network in PyTorch

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 3),   # input → hidden
            nn.ReLU(),
            nn.Linear(3, 1),   # hidden → output
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Setup
model = SimpleNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCELoss()

# Data
X = torch.tensor([[0.5, 0.8], [0.1, 0.4], [0.9, 0.2], [0.3, 0.7]])
Y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

# Training loop
for epoch in range(500):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # backprop — PyTorch handles all the calculus!
    optimizer.step()        # update weights

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

# Predict
with torch.no_grad():
    print("\nPredictions:", model(X).round())
```

### Key PyTorch concepts

| Method | What it does |
|---|---|
| `nn.Linear(in, out)` | Fully connected layer (weights + bias) |
| `nn.ReLU()` | ReLU activation |
| `nn.Sigmoid()` | Sigmoid activation |
| `optimizer.zero_grad()` | Clear gradients from last step |
| `loss.backward()` | Run backpropagation |
| `optimizer.step()` | Update weights with gradient descent |
| `torch.no_grad()` | Disable gradient tracking for inference |

---

## 9. Summary Cheatsheet

| Concept | What it is | Key formula |
|---|---|---|
| Neuron | Computes weighted sum + bias, applies activation | `z = W·x + b` |
| Weight | How important each input is — learned during training | adjusted each epoch |
| Bias | Allows the neuron to shift its output | part of `z = W·x + b` |
| Activation | Adds non-linearity so the network can learn complex patterns | ReLU, Sigmoid, Tanh |
| Forward pass | Data flows input → output, producing a prediction | layer by layer |
| Loss function | Measures how wrong the prediction is | MSE or Cross-Entropy |
| Backpropagation | Chain rule backwards — finds each weight's contribution to the error | `∂L/∂W` |
| Gradient descent | Nudges weights to reduce loss | `w = w − α·(∂L/∂w)` |
| Learning rate α | Controls step size — too high overshoots, too low is slow | typically 0.001–0.1 |
| Epoch | One full pass through the training data | repeat until loss converges |

---

## What's Next?

Once you're comfortable with the basics, explore:

- **Convolutional Neural Networks (CNNs)** — for images
- **Recurrent Neural Networks (RNNs / LSTMs)** — for sequences and text
- **Transformers** — the architecture behind GPT, BERT, and modern LLMs
- **Regularisation** (Dropout, L2) — prevent overfitting
- **Batch Normalisation** — stabilise and speed up training
- **Adam optimiser** — better than plain SGD in most cases

---

*Happy learning!*
