# Self-Pruning Neural Network on CIFAR-10

## Overview

This project implements a self-pruning neural network using learnable gates on weights. The model is trained on the CIFAR-10 dataset and automatically learns which connections are important by applying sparsity regularization.

Instead of manually pruning weights, the network learns to prune itself during training using a differentiable gating mechanism.

---

## Key Idea

Each weight is associated with a learnable gate:

$$
W' = W \cdot \sigma(g)
$$

- W: original weight
- g: learnable gate score
- sigma(g): sigmoid activation (values between 0 and 1)

During training:

- Important weights -> gate close to 1
- Unimportant weights -> gate close to 0

---

## Model Architecture

- Fully Connected Neural Network:
  - Input: 32 x 32 x 3 (flattened)
  - Layer 1: 512 neurons
  - Layer 2: 256 neurons
  - Output: 10 classes (CIFAR-10)

- Custom Layer: PrunableLinear
  - Includes:
    - weights
    - bias
    - learnable gate scores

---

## Dataset

- Dataset: CIFAR-10
- 60,000 images (32x32 RGB)
- 10 classes (airplane, car, bird, etc.)
- Normalized using standard mean and std

---

## Training Setup

- Loss Function:

$$
\text{Loss} = \text{CrossEntropy} + \lambda \cdot \text{SparsityLoss}
$$

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 20

---

## Sparsity Loss

```python
def get_sparsity_loss(model):
    total = 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            total += torch.sigmoid(layer.gate_scores).sum()
    return total
```

This encourages gates to shrink toward zero, effectively pruning weights.

---

## Experiments

We evaluate different values of lambda:

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-5   | ~53%     | ~0%      |
| 1e-4   | ~52-53%  | ~0%      |
| 1e-3   | ~51-53%  | ~0-1%    |

---

## Observations

- Small lambda values lead to almost no pruning
- Model behaves like a standard neural network
- Gates remain around 0.5 (no strong push to 0)

---

## Visualization

The distribution of gate values is plotted after training:

- X-axis: Gate value (0 = pruned, 1 = active)
- Y-axis: Count

Output saved as:

- gate_distribution.png

---

## Limitations

- Sparsity loss is not normalized, causing scale issues
- Sigmoid gating is smooth, so pruning pressure is weak
- No validation split in the baseline workflow
- Low lambda values provide insufficient pruning pressure

---

## Possible Improvements

- Normalize sparsity loss (mean instead of sum)
- Use stronger regularization (higher lambda)
- Add temperature scaling for sharper gating
- Introduce validation split
- Try L0 regularization for more binary pruning behavior
- Apply warmup before sparsity regularization

---

## How to Run

Install dependencies:

```bash
pip install torch torchvision matplotlib
```

Run your training file:

```bash
python your_script.py
```

If you are using the notebook, open start.ipynb and run the cells in order.

---

## Output

- Training logs (loss, accuracy, sparsity)
- Summary table of results
- Gate distribution plot

---

## Key Learning

This project demonstrates:

- Differentiable pruning using learnable gates
- Trade-off between accuracy and sparsity
- Importance of loss scaling and hyperparameter tuning

---

## Author

Manas Goyal  
B.E. Computer Engineering  
Thapar Institute of Engineering and Technology

---

## License

This project is for academic and learning purposes.
