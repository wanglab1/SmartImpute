# SmartImpute

**SmartImpute** A Targeted Imputation Framework for Single-cell Transcriptome Data


## Features

- **Imputation:** Using target gene panel for imputation. 
- **Zero Identification:** Identifies true biological zeros using probabilistic outputs.
- **Multi-Task Learning:** Combines imputation and zero classification into a unified framework.
- **Flexible Utilities:** Includes normalization, splitting, and evaluation functions.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/<your-username>/SmartImpute.git
    cd SmartImpute
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quickstart

Here’s an example:

```python
import pandas as pd
from smartimpute import (
    normalize_data,
    split_data,
    data_transformer,
    Generator,
    Discriminator,
    train_smart,
)

# 1) Load & rearrange
raw = pd.read_csv("examples/sample_data.csv", index_col=0)
data = raw.values              # shape [cells × genes]
genes = raw.columns.tolist()

target_genes = ["GeneA", "GeneB", "GeneC", "GeneX"]
X, gene_order, common_targets = data_transformer(data, genes, target_genes)

# 2) Normalize & split
X_norm = normalize_data(X)
train, test = split_data(X_norm, test_frac=0.2)

# 3) Build masks
#    M: 1 if observed >0, else 0  
#    F: here we simulate forced‐drop on train only
import numpy as np
M = (train > 0).astype(np.float32)
F = np.random.binomial(1, 0.1, size=M.shape).astype(np.float32) * (M==1)

# 4) To torch tensors
import torch
X_t = torch.tensor(train, dtype=torch.float32)
M_t = torch.tensor(M,     dtype=torch.float32)
F_t = torch.tensor(F,     dtype=torch.float32)

# 5) Train!
#    you can swap recon_loss="mse" | "poisson"
generator, history = train_smart(
    X_t, M_t, F_t,
    recon_loss="mse",
    hint_rate=0.9,
    hint2_rate=0.6,
    alpha=100,
    l1_lambda=1.0,
    beta=5.0,
    curriculum=50,
    epochs=100
)

# 6) Impute & extract probabilities
with torch.no_grad():
    Z = torch.rand_like(X_t)
    mu = generator(X_t * M_t + (1-M_t)*Z, M_t)
    imputed = (M_t * X_t + (1-M_t) * mu).cpu().numpy()

# 7) Zero‐scores: fraction of imputed values below threshold
zero_scores = (mu.cpu().numpy() < 0.2).astype(float)

print("Completed! Imputed data shape:", imputed.shape)
```
